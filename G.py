import argparse
import hashlib
import json
import logging
import os
import random
import re
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as thf
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from asteroid_filterbanks import make_enc_dec
    HAS_AFB = True
except ImportError:
    HAS_AFB = False

try:
    from torchmetrics.audio import (
        scale_invariant_signal_noise_ratio as si_snr,
        signal_noise_ratio as snr,
    )
    HAS_TM = True
except ImportError:
    try:
        from torchmetrics.functional import (
            scale_invariant_signal_noise_ratio as si_snr,
            signal_noise_ratio as snr,
        )
        HAS_TM = True
    except ImportError:
        HAS_TM = False

try:
    import soundfile as sf
    HAS_SF = True
except Exception:
    HAS_SF = False

try:
    import torchaudio
    HAS_TA = True
except Exception:
    HAS_TA = False

log = logging.getLogger(__name__)


def _parse_roots(data_root_arg) -> List[Path]:
    if isinstance(data_root_arg, (list, tuple)):
        raw = data_root_arg
    else:
        raw = str(data_root_arg).split(",")
    roots = []
    for r in raw:
        r = r.strip()
        if not r:
            continue
        roots.append(Path(r).expanduser().resolve())
    return roots


def _tqdm(iterable, **kwargs):
    if HAS_TQDM:
        return tqdm(iterable, **kwargs)
    return iterable


class _GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights):
        ctx.save_for_backward(weights)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        weights, = ctx.saved_tensors
        w = weights.view(-1, *([1] * (grad_output.dim() - 1)))
        return grad_output * w, None


class IndividualCurriculumTable:
    def __init__(self, decay=0.9, delta_decay=0.9,
                 delta_threshold=0.05, low=0.3, high=1.0,
                 cooldown_epochs=5):
        self.decay = decay
        self.delta_decay = delta_decay
        self.delta_threshold = delta_threshold
        self.low = low
        self.high = high
        self.cooldown_epochs = cooldown_epochs

        self.score = {}
        self.delta = {}
        self.weight = {}
        self.count = {}
        self.last_active_epoch = {}

    @torch.no_grad()
    def update(self, individual_id, si_sdr_val, current_epoch):
        iid = int(individual_id)
        prev = self.score.get(iid, None)

        if prev is None:
            self.score[iid] = float(si_sdr_val)
            self.delta[iid] = 0.0
            self.weight[iid] = self.high
            self.count[iid] = 1
            self.last_active_epoch[iid] = current_epoch
            return

        new_score = self.decay * prev + (1 - self.decay) * float(si_sdr_val)
        d = new_score - prev
        prev_delta = self.delta.get(iid, 0.0)
        new_delta = self.delta_decay * prev_delta + (1 - self.delta_decay) * d

        self.score[iid] = new_score
        self.delta[iid] = new_delta
        self.count[iid] = self.count.get(iid, 0) + 1

        if new_delta > self.delta_threshold:
            self.weight[iid] = self.high
            self.last_active_epoch[iid] = current_epoch
        else:
            self.weight[iid] = self.low

        if self.weight[iid] == self.low:
            last_active = self.last_active_epoch.get(iid, 0)
            if current_epoch - last_active >= self.cooldown_epochs:
                self.weight[iid] = self.high
                self.last_active_epoch[iid] = current_epoch

    def get(self, individual_id):
        return self.weight.get(int(individual_id), self.high)

    def get_batch_weights(self, individual_ids, device, dtype):
        vals = [self.get(i) for i in individual_ids]
        return torch.tensor(vals, dtype=dtype, device=device)

    def state_dict(self):
        return {
            "decay": self.decay,
            "delta_decay": self.delta_decay,
            "delta_threshold": self.delta_threshold,
            "low": self.low,
            "high": self.high,
            "cooldown_epochs": self.cooldown_epochs,
            "score": dict(self.score),
            "delta": dict(self.delta),
            "weight": dict(self.weight),
            "count": dict(self.count),
            "last_active_epoch": dict(self.last_active_epoch),
        }

    def load_state_dict(self, sd):
        self.decay = float(sd.get("decay", self.decay))
        self.delta_decay = float(sd.get("delta_decay", self.delta_decay))
        self.delta_threshold = float(sd.get("delta_threshold", self.delta_threshold))
        self.low = float(sd.get("low", self.low))
        self.high = float(sd.get("high", self.high))
        self.cooldown_epochs = int(sd.get("cooldown_epochs", self.cooldown_epochs))
        self.score = {int(k): float(v) for k, v in sd.get("score", {}).items()}
        self.delta = {int(k): float(v) for k, v in sd.get("delta", {}).items()}
        self.weight = {int(k): float(v) for k, v in sd.get("weight", {}).items()}
        self.count = {int(k): int(v) for k, v in sd.get("count", {}).items()}
        self.last_active_epoch = {
            int(k): int(v) for k, v in sd.get("last_active_epoch", {}).items()
        }

    def stats(self):
        if not self.weight:
            return {}
        vals = list(self.weight.values())
        n = len(vals)
        n_high = sum(1 for v in vals if v >= 0.99)
        n_low = sum(1 for v in vals if v <= self.low + 0.01)
        scores = list(self.score.values())
        deltas = list(self.delta.values())
        return {
            "n_individuals": n,
            "n_high": n_high,
            "n_low": n_low,
            "frac_high": n_high / n,
            "frac_low": n_low / n,
            "score_mean": sum(scores) / len(scores) if scores else 0.0,
            "delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
        }


class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5,
                 element_wise=True):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.element_wise = element_wise
        if self.element_wise:
            self.gamma = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
            self.beta = nn.Parameter(torch.Tensor(*param_size).to(torch.float32))
            nn.init.ones_(self.gamma)
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        mu_ = x.mean(dim=self.stat_dims, keepdim=True)
        std_ = torch.sqrt(x.var(dim=self.stat_dims, unbiased=False,
                                keepdim=True) + self.eps)
        if self.element_wise:
            return ((x - mu_) / std_) * self.gamma + self.beta
        return (x - mu_) / std_


class RMSNorm(nn.Module):
    def __init__(self, input_dims, stat_dims, p=-1., eps=1e-8, bias=False):
        super().__init__()
        self.eps = eps
        self.stat_dims = stat_dims
        self.d = 1
        param_size = [1] * 3
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
            self.d *= input_dim
        self.scale = nn.Parameter(torch.ones(param_size, dtype=torch.float32))
        if bias:
            self.offset = nn.Parameter(torch.zeros(param_size, dtype=torch.float32))

    def forward(self, x):
        norm_x = x.norm(2, dim=self.stat_dims, keepdim=True)
        rms_x = norm_x * self.d ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)
        return self.scale * x_normed


class RNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, dropout_p=0.1, bidirectional=False):
        super().__init__()
        self.rnn1 = nn.GRU(emb_dim // 2, hidden_dim // 2, 1, batch_first=True,
                           bidirectional=bidirectional)
        self.rnn2 = nn.GRU(emb_dim // 2, hidden_dim // 2, 1, batch_first=True,
                           bidirectional=bidirectional)
        self.norm = RMSNorm((emb_dim,), (2,))
        if bidirectional:
            self.dense = nn.Sequential(
                nn.Linear(hidden_dim * 2, emb_dim * 2), nn.GLU(-1))
        else:
            self.dense = nn.Sequential(
                nn.Linear(hidden_dim, emb_dim * 2), nn.GLU(-1))

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=-1)
        x1, _ = self.rnn1(x1)
        x2, _ = self.rnn2(x2)
        x = torch.cat((x1, x2), dim=-1)
        x = self.norm(x)
        return self.dense(x)


class DualPathRNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, n_freqs=32, dropout_p=0.1):
        super().__init__()
        self.intra_norm = nn.LayerNorm(emb_dim)
        self.intra_rnn_attn = RNN(emb_dim, hidden_dim // 2, dropout_p,
                                  bidirectional=True)
        self.inter_norm = nn.LayerNorm(emb_dim)
        self.inter_rnn_attn = RNN(emb_dim, hidden_dim, dropout_p,
                                  bidirectional=False)

    def forward(self, x):
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)
        x_res = x
        x = x.reshape(B * T, F, D)
        x = self.intra_norm(x)
        x = self.intra_rnn_attn(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res
        x_res = x
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B * F, T, D)
        x = self.inter_norm(x)
        x = self.inter_rnn_attn(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3)
        x = x + x_res
        return x.permute(0, 3, 1, 2)


class minGRUAttentionM(nn.Module):
    def __init__(self, d_model, scale=2, dropout=0.0, gruType="mingru"):
        super().__init__()
        self.gru = nn.LSTM(d_model, d_model, 1, batch_first=True)
        self.xconv = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.SiLU())
        self.norm = RMSNorm((d_model,), (2,))
        self.post_linear = nn.Linear(d_model, d_model)
        self.ou_conv = nn.Conv2d(d_model, d_model, 1)
        self.act = nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev=None):
        return_prev = prev is not None
        skip = x
        fx = torch.mean(thf.relu(x) ** 2, dim=-1).transpose(1, 2)
        skip2 = fx
        fx = self.norm(fx)
        fx, _ = self.gru(fx)
        fx = self.post_linear(fx) + skip2
        fx = fx.transpose(1, 2).unsqueeze(-1)
        x = self.act(fx) * self.xconv(x)
        x = self.dropout(x)
        x = self.ou_conv(x)
        out = x + skip
        if return_prev:
            return out, prev
        return out


class Film(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.w = nn.Conv2d(channel, channel, 1)
        self.b = nn.Conv2d(channel, channel, 1)
        self.norm = nn.LayerNorm(channel)

    def forward(self, x, e):
        w = self.w(e)
        b = self.b(e)
        o = w * x + b
        return self.norm(o.transpose(1, 3)).transpose(1, 3)


class DPR(nn.Module):
    def __init__(self, emb_dim=16, hidden_dim=24, n_freqs=32, dropout_p=0.1, r=2):
        super().__init__()
        self.film = Film(emb_dim)
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = minGRUAttentionM(emb_dim, 2, 0.1)

    def forward(self, x, spk=None):
        if spk is not None:
            x = self.film(x, spk)
        x = self.dp_rnn_attn(x)
        x = self.conv_glu(x)
        return x


class TCM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.GroupNorm(1, channel),
        )
        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1, groups=channel)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1, groups=channel),
            nn.Sigmoid()
        )
        self.ou_conv = nn.Conv2d(channel, channel, 1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.conv1(x) * self.conv2(x)
        return self.ou_conv(x)


class SimpleDownsample(nn.Module):
    def __init__(self, downsample: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(downsample))

    def forward(self, src):
        B, C, T, F = src.shape
        src = src.reshape(B, C, T, F // 2, 2)
        weights = self.bias.softmax(dim=0).view(1, 1, 1, 1, 2)
        return (src * weights).sum(dim=-1)


class SimpleUpsample(nn.Module):
    def __init__(self, upsample: int):
        super().__init__()
        self.upsample = upsample

    def forward(self, src):
        B, C, T, F = src.shape
        src = src.unsqueeze(-1).expand(B, C, T, F, self.upsample)
        return src.reshape(B, C, T, F * 2)


class TargetResidualBridge(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.x_norm = nn.LayerNorm(channel, elementwise_affine=False)
        self.p_norm = nn.LayerNorm(channel, elementwise_affine=False)
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, p):
        x_norm = self.x_norm(x.transpose(1, 3)).transpose(1, 3)
        p_norm = self.p_norm(p.transpose(1, 3)).transpose(1, 3)
        return x + self.beta * (x_norm * p_norm)


class FeatureDRC(nn.Module):
    def __init__(self, compress_factor=0.5):
        super().__init__()
        self.compress_factor = compress_factor
        self.mix = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        compressed = torch.sign(x) * torch.pow(
            torch.clamp(x.abs(), min=1e-8), self.compress_factor)
        return x + self.mix * (compressed - x)


class SpkEncoder(nn.Module):
    def __init__(self, channels, n_layer=6, fdrc_factor=0.5):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.in_conv = nn.Conv2d(channels, channels, 3, padding=(1, 0),
                                 stride=(1, 2))
        self.fdrc = FeatureDRC(fdrc_factor)
        for _ in range(n_layer):
            self.layers.append(TCM(channels))

    def forward(self, x):
        x = self.in_conv(x)
        x = self.fdrc(x)
        for layer in self.layers:
            x = layer(x)
        x = torch.mean(x, dim=-2, keepdim=True)
        return x


class MaskNet(nn.Module):
    def __init__(self, ch_dim, freq_dim, n_layer, r, n_head):
        super().__init__()
        self.freq_down = nn.Sequential(
            nn.ConstantPad2d((1, 0, 0, 0), 0.0),
            nn.Conv2d(ch_dim, ch_dim, (1, 3)),
            nn.ReLU(),
        )
        self.sd = SimpleDownsample(2)
        self.dpr1 = nn.ModuleList([])
        for i in range(6):
            self.dpr1.append(DPR(ch_dim, ch_dim, freq_dim // 2))
        self.b5_b6_bridge = TargetResidualBridge(ch_dim)
        self.freq_up = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), 0.0),
            nn.Conv2d(ch_dim, ch_dim, (1, 2)),
            nn.ReLU(),
        )
        self.su = SimpleUpsample(2)

    def forward(self, x, spk):
        x = self.freq_down(x)
        x = self.sd(x)
        for i in range(len(self.dpr1)):
            if i >= 2:
                x = self.dpr1[i](x, spk)
            else:
                x = self.dpr1[i](x)
            if i == 4:
                x = self.b5_b6_bridge(x, spk)
        x = self.su(x)
        x = self.freq_up(x)
        return x


class ACNTCN(nn.Module):
    def __init__(self, ch_dim, label_emb_dim=512, n_fft=256, stride=128,
                 window="hann", n_layers=6, eps=1.0e-5, compress_factor=None,
                 fdrc_factor=0.5):
        super().__init__()
        assert HAS_AFB
        self.n_srcs = 1
        self.n_layers = n_layers
        self.n_imics = 1
        n_freqs = n_fft // 2 + 1
        self.n_freqs = n_freqs
        self.eps = eps
        self.chunk_size = stride
        self.istft_pad = n_fft - stride
        self.istft_lookback = 1 + (self.istft_pad - 1) // self.istft_pad

        self.enc, self.dec = make_enc_dec(
            "stft", n_filters=n_fft, kernel_size=n_fft,
            stride=stride, window_type=window,
        )
        self.t_ksize = 3
        self.ch_dim = ch_dim
        self.half_freq_dim = n_freqs // 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.n_imics * 2, ch_dim, kernel_size=(3, 3), padding=(0, 1)),
        )
        self.mask_net = MaskNet(ch_dim, n_freqs, n_layers, 4, 8)
        self.spk_encoder = SpkEncoder(ch_dim, n_layer=n_layers,
                                      fdrc_factor=fdrc_factor)
        self.compress_factor = compress_factor
        self.ou_conv = nn.Sequential(
            nn.Conv2d(ch_dim, self.n_imics * 2, kernel_size=(3, 3), padding=(0, 1)),
        )

    def init_buffers(self, batch_size, device):
        return dict(
            conv_buf=torch.zeros(batch_size, self.n_imics * 2,
                                 self.t_ksize - 1, self.n_freqs, device=device),
            istft_buf=torch.zeros(batch_size, self.n_srcs,
                                  self.n_freqs * 2, self.istft_lookback,
                                  device=device),
            deconv_buf=torch.zeros(batch_size, self.ch_dim,
                                   self.t_ksize - 1, self.n_freqs,
                                   device=device),
        )

    def drc(self, x, dim=1):
        if self.compress_factor is None:
            return x
        out_dtype = x.dtype
        real, imag = torch.unbind(x.float(), dim=dim)
        comp = torch.complex(real, imag)
        mag = torch.abs(comp) ** self.compress_factor
        phase = torch.angle(comp)
        out = torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=dim)
        return out.to(out_dtype)

    def idrc(self, x, dim=1):
        if self.compress_factor is None:
            return x
        out_dtype = x.dtype
        real, imag = torch.unbind(x.float(), dim=dim)
        comp = torch.complex(real, imag)
        mag = torch.abs(comp) ** (1.0 / self.compress_factor)
        phase = torch.angle(comp)
        out = torch.stack((mag * torch.cos(phase), mag * torch.sin(phase)), dim=dim)
        return out.to(out_dtype)

    def forward(self, x, spk, input_state=None):
        if input_state is None:
            input_state = self.init_buffers(x.shape[0], x.device)
        conv_buf = input_state["conv_buf"]
        deconv_buf = input_state["deconv_buf"]
        istft_buf = input_state["istft_buf"]

        batch = self.enc(x)
        spk = self.enc(spk)

        batch = torch.stack(
            (batch[..., : self.n_freqs, :], batch[..., self.n_freqs:, :]), dim=1)
        batch = batch.transpose(2, 3).contiguous()

        spk = torch.stack(
            (spk[..., : self.n_freqs, :], spk[..., self.n_freqs:, :]), dim=1)
        spk = spk.transpose(2, 3).contiguous()

        batch = self.drc(batch)
        spk = self.drc(spk)

        batch = torch.cat((conv_buf, batch), dim=2)
        conv_buf = batch[:, :, -(self.t_ksize - 1):, :]

        batch = self.in_conv(batch)
        spk = self.in_conv(spk)
        spk = self.spk_encoder(spk)

        B, M, T, C = batch.shape
        mask = self.mask_net(batch, spk)
        batch = mask * batch

        batch = torch.cat((deconv_buf, batch), dim=2)
        deconv_buf = batch[:, :, -(self.t_ksize - 1):, :]
        batch = self.ou_conv(batch)
        batch = self.idrc(batch)

        batch = batch.view([B, self.n_srcs, 2, T, C]).transpose(3, 4)
        batch = torch.cat([batch[:, :, 0], batch[:, :, 1]], dim=2)

        batch = torch.cat([istft_buf, batch], dim=3)
        istft_buf = batch[..., -self.istft_lookback:]

        batch = self.dec(batch)
        batch = batch[..., self.istft_lookback * self.chunk_size:]

        input_state["conv_buf"] = conv_buf
        input_state["deconv_buf"] = deconv_buf
        input_state["istft_buf"] = istft_buf
        return batch, input_state


class Net(nn.Module):
    def __init__(self, label_len, ch_dim=64, n_fft=256, stride=128,
                 label_emb_dim=512, n_layers=6, compress_factor=0.3,
                 fdrc_factor=0.5):
        super().__init__()
        self.net = ACNTCN(
            ch_dim=ch_dim, label_emb_dim=label_emb_dim,
            n_fft=n_fft, stride=stride, n_layers=n_layers,
            compress_factor=compress_factor, fdrc_factor=fdrc_factor,
        )
        self.stride = stride
        self.spk_encoder = nn.Identity()

    def forward(self, x, spk=None, input_state=None, pad=True,
                writer=None, step=None, idx=None):
        if spk is None:
            spk = torch.randn((1, 8000), device=x.device)
        L = x.shape[-1]
        pad_n = 0
        if L % self.stride != 0:
            pad_n = (L // self.stride + 1) * self.stride - L
            x = torch.nn.functional.pad(x, pad=(0, pad_n))
        x, _ = self.net(x, spk)
        if pad_n != 0:
            x = x[..., :L]
        return x


def optimizer(model, data_parallel=False, **kwargs):
    return torch.optim.AdamW(model.parameters(), **kwargs)


def loss(est, tgt):
    p = si_snr(est, tgt).mean()
    sn = snr(est, tgt).mean()
    l = -0.5 * p - 0.5 * sn
    return l, p.detach(), sn.detach()


def metrics(mixed, output, gt):
    m = {}

    def metric_i(metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).cpu().item())
        return _vals

    for m_fn in [snr, si_snr]:
        m[m_fn.__name__] = metric_i(m_fn, mixed[:, : gt.shape[1], :], output, gt)
    return m


class ModelEMA:
    def __init__(self, model, decay=0.9999, unbias=True, device="cpu"):
        self.decay = decay
        self.unbias = unbias
        self.device = device
        self.count = 0
        self.ema_state = {}
        for k, v in model.state_dict().items():
            if v.dtype == torch.float32:
                self.ema_state[k] = v.detach().to(device, copy=True)

    @torch.no_grad()
    def update(self, model):
        if self.unbias:
            self.count = self.count * self.decay + 1
            w = 1.0 / self.count
        else:
            w = 1.0 - self.decay
        for k, v in model.state_dict().items():
            if k in self.ema_state:
                self.ema_state[k].mul_(1 - w).add_(
                    v.detach().to(self.device), alpha=w)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_state, strict=False)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        model.load_state_dict(backup)

    def state_dict(self):
        return {
            "ema_state": {k: v.detach().cpu() for k, v in self.ema_state.items()},
            "ema_count": self.count,
        }


_PAT = re.compile(
    r"^(?P<utt1>[^_]+)_(?P<g1>-?\d+\.?\d*)_(?P<utt2>[^_]+)_(?P<g2>-?\d+\.?\d*)\.wav$"
)


def _parse_name(name):
    m = _PAT.match(name)
    if not m:
        raise ValueError(f"bad name: {name}")
    d = m.groupdict()
    d["spk1"] = d["utt1"][:3]
    d["spk2"] = d["utt2"][:3]
    return d


def _build_enroll_table(enroll_roots: List[Path]):
    table = {}
    for enroll_root in enroll_roots:
        if not enroll_root.exists():
            log.warning(f"enroll_root not found: {enroll_root}")
            continue
        for si_dir in sorted(enroll_root.glob("si_*")):
            if not si_dir.is_dir():
                continue
            for spk_dir in sorted(si_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                wavs = sorted(str(w) for w in spk_dir.glob("*.wav"))
                if wavs:
                    table.setdefault(spk_dir.name, []).extend(wavs)
    return table


def _pad_or_crop(wav, target_length, start=None):
    length = wav.size(-1)
    pad = 0
    if length < target_length:
        pad = target_length - length
        wav = torch.nn.functional.pad(wav, (0, pad))
        start = 0
    else:
        if start is None:
            start = torch.randint(0, length - target_length + 1, ()).item()
        else:
            start = max(0, min(start, length - target_length))
        wav = wav[..., start:start + target_length]
    return wav, start, pad


def _find_first_speech(wav, sr, frame_ms=20.0, ratio=0.05, min_speech_ms=40.0):
    if wav.numel() == 0:
        return 0
    frame = max(int(sr * frame_ms / 1000.0), 1)
    n_frame = wav.shape[-1] // frame
    if n_frame <= 0:
        return 0
    x = wav[:n_frame * frame].reshape(n_frame, frame)
    rms = x.pow(2).mean(dim=-1).sqrt()
    peak = rms.max().item()
    if peak <= 0.0:
        return 0
    thr = peak * ratio
    active = (rms > thr).tolist()
    min_run = max(int(min_speech_ms / frame_ms), 1)
    run = 0
    for i, a in enumerate(active):
        if a:
            run += 1
            if run >= min_run:
                return (i - run + 1) * frame
        else:
            run = 0
    return 0


class FRNetDataset(Dataset):
    def __init__(self, mix_dirs, enroll_table, sample_rate=8000, training=False,
                 mix_seconds=4.0, enroll_seconds=4.0, vad_frame_ms=20.0,
                 vad_ratio=0.05, vad_min_speech_ms=40.0, norm_ref=True,
                 vad_enable=True):
        self.sample_rate = sample_rate
        self.training = training
        self.vad_frame_ms = vad_frame_ms
        self.vad_ratio = vad_ratio
        self.vad_min_speech_ms = vad_min_speech_ms
        self.norm_ref = norm_ref
        self.vad_enable = vad_enable

        self.enroll_table = enroll_table
        self.samples = []

        if isinstance(mix_dirs, (str, Path)):
            mix_dirs = [Path(mix_dirs)]
        else:
            mix_dirs = [Path(d) for d in mix_dirs]

        seen_names = set()
        stats = {"total": 0, "dup": 0, "bad_name": 0, "no_pool": 0,
                 "no_s1s2": 0, "kept": 0}

        for mix_dir_p in mix_dirs:
            if not mix_dir_p.exists():
                log.warning(f"mix_dir not found: {mix_dir_p}")
                continue
            s1_dir = mix_dir_p.parent / "s1"
            s2_dir = mix_dir_p.parent / "s2"
            for mix in sorted(mix_dir_p.glob("*.wav")):
                stats["total"] += 1
                if mix.name in seen_names:
                    stats["dup"] += 1
                    continue
                try:
                    info = _parse_name(mix.name)
                except ValueError:
                    stats["bad_name"] += 1
                    continue
                pool1 = self.enroll_table.get(info["spk1"], [])
                pool2 = self.enroll_table.get(info["spk2"], [])
                if not pool1 or not pool2:
                    stats["no_pool"] += 1
                    continue
                s1_p = s1_dir / mix.name
                s2_p = s2_dir / mix.name
                if not s1_p.exists() or not s2_p.exists():
                    stats["no_s1s2"] += 1
                    continue
                u1 = f"{info['utt1']}.wav"
                u2 = f"{info['utt2']}.wav"
                pool1f = [p for p in pool1 if Path(p).name != u1] or pool1
                pool2f = [p for p in pool2 if Path(p).name != u2] or pool2

                self.samples.append({
                    "mix": str(mix),
                    "s1": str(s1_p),
                    "s2": str(s2_p),
                    "pool1": pool1f,
                    "pool2": pool2f,
                    "name": mix.name,
                })
                seen_names.add(mix.name)
                stats["kept"] += 1

        log.info(f"  [FRNetDataset] mix_dirs={len(mix_dirs)}  "
                 f"total={stats['total']}  dup={stats['dup']}  "
                 f"bad_name={stats['bad_name']}  no_pool={stats['no_pool']}  "
                 f"no_s1s2={stats['no_s1s2']}  kept={stats['kept']}  "
                 f"premixed=True  single_source="
                 f"{'random' if training else 'alternate'}")

        self.mix_length = int(mix_seconds * sample_rate)
        self.enroll_len = int(enroll_seconds * sample_rate)

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        if HAS_SF:
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav).mean(axis=1)
        else:
            wav, sr = torchaudio.load(path)
            wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            if HAS_TA:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            else:
                t_new_len = int(wav.shape[-1] * self.sample_rate / sr)
                wav = torch.nn.functional.interpolate(
                    wav.view(1, 1, -1), size=t_new_len, mode="linear",
                    align_corners=False).view(-1)
        return wav

    def _load_enroll(self, pool):
        path = random.choice(pool) if self.training else pool[0]
        wav = self._load(path)
        if self.vad_enable:
            start0 = _find_first_speech(
                wav, self.sample_rate,
                frame_ms=self.vad_frame_ms, ratio=self.vad_ratio,
                min_speech_ms=self.vad_min_speech_ms)
            wav = wav[start0:]
        wav, _, _ = _pad_or_crop(wav, self.enroll_len, start=None)
        if self.norm_ref:
            mean = wav.mean(dim=-1, keepdim=True)
            std = wav.std(dim=-1, keepdim=True)
            wav = (wav - mean) / (std + 1e-8)
        return wav.unsqueeze(0)

    def __getitem__(self, idx):
        g = self.samples[idx]

        if self.training:
            spk_idx = random.randint(0, 1)
        else:
            spk_idx = idx % 2

        mix = self._load(g["mix"])

        if spk_idx == 0:
            target = self._load(g["s1"])
            pool = g["pool1"]
        else:
            target = self._load(g["s2"])
            pool = g["pool2"]

        mix, start, pad = _pad_or_crop(mix, self.mix_length)
        if pad > 0:
            target = F.pad(target, (0, pad))
        else:
            target = target[..., start:start + self.mix_length]

        maxval = torch.max(torch.abs(mix))
        if maxval >= 1:
            mix = mix / maxval
            target = target / maxval

        enroll = self._load_enroll(pool)

        individual_id = idx * 2 + spk_idx

        return (mix.unsqueeze(0), enroll, target.unsqueeze(0),
                torch.tensor(spk_idx),
                torch.tensor(individual_id, dtype=torch.long))


def _set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _save_checkpoint(ckpt_dir, name, model, meta, ema_state=None,
                     optimizer_state=None, scheduler_state=None,
                     curriculum_state=None):
    p = Path(ckpt_dir)
    p.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {"state": state, "model_state_dict": state, "meta": meta}
    if ema_state is not None:
        payload["ema_state_dict"] = ema_state
    if optimizer_state is not None:
        payload["optimizer_state_dict"] = optimizer_state
    if scheduler_state is not None:
        payload["scheduler_state_dict"] = scheduler_state
    if curriculum_state is not None:
        payload["curriculum_state"] = curriculum_state
    torch.save(payload, p / f"{name}.pt")


def _try_load_resume(ckpt_dir, model, optimizer_, scheduler, ema, device):
    last_path = Path(ckpt_dir) / "last.pt"
    if not last_path.exists():
        log.info("[resume] no last.pt found, training from scratch")
        return None
    try:
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
    except Exception as e:
        log.warning(f"[resume] failed to load {last_path}: {e}")
        return None

    meta = ckpt.get("meta", {})
    missing, unexpected = model.load_state_dict(
        {k: v.to(device) for k, v in ckpt["state"].items()}, strict=False)
    if missing:
        log.warning(f"[resume] missing {len(missing)} keys, e.g. {missing[:3]}")
    if unexpected:
        log.warning(f"[resume] unexpected {len(unexpected)} keys, "
                    f"e.g. {unexpected[:3]}")

    if optimizer_ is not None and "optimizer_state_dict" in ckpt:
        try:
            optimizer_.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as e:
            log.warning(f"[resume] failed to load optimizer: {e}")

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception as e:
            log.warning(f"[resume] failed to load scheduler: {e}")

    if ema is not None and "ema_state_dict" in ckpt:
        try:
            ema_saved = ckpt["ema_state_dict"]
            ema.ema_state = {
                k: v.to(device) for k, v in ema_saved["ema_state"].items()}
            ema.count = ema_saved.get("ema_count", 0)
        except Exception as e:
            log.warning(f"[resume] failed to load EMA: {e}")

    start_epoch = int(meta.get("epoch", -1)) + 1
    best_metric = float(meta.get("best_metric", -1e9))
    patience_count = int(meta.get("patience_count", 0))
    history = meta.get("history", [])

    curriculum_state = ckpt.get("curriculum_state", None)

    log.info(f"[resume] loaded {last_path}")
    log.info(f"[resume] start_epoch={start_epoch}  "
             f"best_metric={best_metric:.4f}  patience={patience_count}")
    return (start_epoch, best_metric, patience_count, history,
            curriculum_state)


def _summarize(records):
    if not records:
        return {}
    out = {"n": len(records)}
    for k in records[0].keys():
        vals = [r[k] for r in records if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else 0.0
    return out


def _fmt_summary(s):
    if not s:
        return "empty"
    parts = [f"n={s.get('n', 0)}"]
    for k, v in s.items():
        if k == "n":
            continue
        if isinstance(v, (int, float)):
            parts.append(f"{k}={v:.3f}")
    return "  ".join(parts)


def _train_one_epoch(model, loader, optimizer_, scaler, ema, device, amp,
                     clip_val=0.0, epoch=0,
                     curriculum=None, grad_scale_enable=True):
    model.train()
    total_loss, n = 0.0, 0
    metric_sums = {}
    p_sum = 0.0
    sn_sum = 0.0

    w_sum = 0.0
    w_n = 0

    use_clip = clip_val is not None and clip_val > 0.0

    pbar = _tqdm(loader, desc=f"  epoch {epoch:02d} train",
                 leave=False, dynamic_ncols=True, mininterval=1.0)

    for batch in pbar:
        mix, enroll, target, key, individual_ids = batch
        mix = mix.to(device, non_blocking=True)
        enroll = enroll.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        individual_ids = individual_ids.to(device)

        optimizer_.zero_grad(set_to_none=True)

        if curriculum is not None and grad_scale_enable:
            with torch.no_grad():
                w = curriculum.get_batch_weights(
                    individual_ids.tolist(), device, torch.float32)
            w_sum += w.mean().item()
            w_n += 1
        else:
            w = None

        if amp and scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                est = model(mix, enroll)
            est = est.float()

            if w is not None:
                est = _GradientScaler.apply(est, w)

            l, p, sn = loss(est, target)
            scaler.scale(l).backward()
            scaler.unscale_(optimizer_)
            if use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            scaler.step(optimizer_)
            scaler.update()
        else:
            est = model(mix, enroll)

            if w is not None:
                est = _GradientScaler.apply(est, w)

            l, p, sn = loss(est, target)
            l.backward()
            if use_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer_.step()

        if ema is not None:
            ema.update(model)

        if curriculum is not None:
            with torch.no_grad():
                s_batch = si_snr(est.detach(), target).view(-1)
                for iid, s_val in zip(individual_ids.tolist(), s_batch.tolist()):
                    curriculum.update(iid, s_val, epoch)

        p_sum += p.item()
        sn_sum += sn.item()

        with torch.no_grad():
            try:
                m = metrics(mix.detach(), est.detach(), target.detach())
                for k, v in m.items():
                    if isinstance(v, list) and len(v) > 0:
                        metric_sums[k] = metric_sums.get(k, 0.0) + float(sum(v) / len(v))
            except Exception as e:
                if n == 0:
                    log.warning(f"metrics failed: {e}")

        bs = mix.shape[0]
        total_loss += l.item() * bs
        n += bs

        if HAS_TQDM:
            cur_loss = total_loss / max(n, 1)
            postfix = {
                "loss": f"{cur_loss:.2f}",
                "p": f"{p.item():.2f}",
                "sn": f"{sn.item():.1f}",
                "clip": "on" if use_clip else "off",
            }
            if w is not None:
                postfix["w"] = f"{w.mean().item():.2f}"
            pbar.set_postfix(postfix)

    n = max(n, 1)
    n_b = max(len(loader), 1)
    out = {
        "loss": total_loss / n,
        "p": p_sum / n_b,
        "snr": sn_sum / n_b,
    }
    if w_n > 0:
        out["w_mean"] = w_sum / w_n
    for k, v in metric_sums.items():
        out[k] = v / n_b
    return out


@torch.no_grad()
def _evaluate(model, loader, device, amp=False):
    model.eval()
    records = []
    pbar = _tqdm(loader, desc="  eval ",
                 leave=False, dynamic_ncols=True, mininterval=1.0)
    for batch in pbar:
        if len(batch) == 5:
            mix, enroll, target, key, _ = batch
        else:
            mix, enroll, target, key = batch
        mix = mix.to(device)
        enroll = enroll.to(device)
        target = target.to(device)

        if amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                est = model(mix, enroll)
            est = est.float()
        else:
            est = model(mix, enroll)

        try:
            m = metrics(mix, est, target)
        except Exception:
            m = {}
        B = mix.shape[0]
        for j in range(B):
            rec = {}
            for k, v in m.items():
                if isinstance(v, list) and j < len(v):
                    rec[k] = float(v[j])
            rec["key"] = int(key[j].item())
            records.append(rec)
    return records


def run_standalone(args):
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _set_logger(str(ckpt_dir / "train.log"))
    _seed_everything(args.seed)

    if not HAS_AFB:
        log.error("需要安装 asteroid_filterbanks")
        return
    if not HAS_TM:
        log.error("需要安装 torchmetrics")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda") and args.amp
    log.info(f"device: {device}  amp: {amp}  tqdm: {HAS_TQDM}")

    roots = _parse_roots(args.data_root)
    log.info(f"data_roots ({len(roots)}):")
    for r in roots:
        exists_mix = (r / "wsj0-mix/2speakers/wav8k/min/tr/mix").exists()
        exists_enroll = (r / "wsj0").exists()
        log.info(f"  - {r}  [mix={exists_mix}  enroll={exists_enroll}]")

    def _find_split_mix_dirs(split):
        rel = f"wsj0-mix/2speakers/wav8k/min/{split}/mix"
        dirs = []
        for r in roots:
            cand = r / rel
            if cand.exists():
                dirs.append(cand)
        if not dirs:
            raise FileNotFoundError(
                f"找不到 {rel}，已在以下 root 搜索：{[str(r) for r in roots]}")
        return dirs

    tr_mix_dirs = _find_split_mix_dirs("tr")
    cv_mix_dirs = _find_split_mix_dirs("cv")
    tt_mix_dirs = _find_split_mix_dirs("tt")
    for split_name, dirs in [("tr", tr_mix_dirs), ("cv", cv_mix_dirs),
                             ("tt", tt_mix_dirs)]:
        log.info(f"{split_name}_mix_dirs ({len(dirs)}):")
        for d in dirs:
            n = len(list(d.glob("*.wav")))
            log.info(f"  - {d}  ({n} wav)")

    enroll_roots = [r / "wsj0" for r in roots]
    enroll_table = _build_enroll_table(enroll_roots)
    n_wavs = sum(len(v) for v in enroll_table.values())
    log.info(f"enroll: {len(enroll_table)} speakers, {n_wavs} wav files")
    if not enroll_table:
        log.error("!! 未找到 enroll 数据，请检查 data_root 路径")
        return

    common = dict(
        sample_rate=args.sample_rate,
        norm_ref=args.norm_ref,
        vad_enable=args.vad_enable,
        mix_seconds=args.mix_seconds,
        enroll_seconds=args.enroll_seconds,
    )
    tr_ds = FRNetDataset(tr_mix_dirs, enroll_table, training=True, **common)
    cv_ds = FRNetDataset(cv_mix_dirs, enroll_table, training=False, **common)
    tt_ds = FRNetDataset(tt_mix_dirs, enroll_table, training=False, **common)
    log.info(f"dataset sizes: tr={len(tr_ds)}  cv={len(cv_ds)}  tt={len(tt_ds)}")
    log.info(f"mode: vad={args.vad_enable}  amp={amp}  "
             f"premixed=True  train_source=random  val_source=alternate")
    log.info(f"★ loss (单源，与原初.py 完全相同): -0.5*si_snr - 0.5*snr")
    log.info(f"★ 个体 ID = idx * 2 + spk_idx；batch 里 A/B 个体随机混合")
    log.info(f"★ 个体课程表: enable={args.enable_curriculum}  "
             f"decay={args.curriculum_decay}  "
             f"delta_decay={args.curriculum_delta_decay}  "
             f"delta_threshold={args.curriculum_delta_threshold}  "
             f"low={args.curriculum_low}  high={args.curriculum_high}  "
             f"cooldown_epochs={args.curriculum_cooldown_epochs}")
    if len(tr_ds) == 0:
        log.error("训练集为空")
        return

    nw = args.num_workers
    pm = (device.type == "cuda")
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=nw, drop_last=True, pin_memory=pm)
    cv_loader = DataLoader(cv_ds, batch_size=args.eval_batch_size, shuffle=False,
                           num_workers=nw, pin_memory=pm)
    tt_loader = DataLoader(tt_ds, batch_size=args.eval_batch_size, shuffle=False,
                           num_workers=nw, pin_memory=pm)

    model_params = dict(
        label_len=args.label_len, ch_dim=args.ch_dim, n_fft=args.n_fft,
        stride=args.stride, label_emb_dim=args.label_emb_dim,
        n_layers=args.n_layers, compress_factor=args.compress_factor,
        fdrc_factor=args.fdrc_factor,
    )
    log.info(f"model_params: {model_params}")
    model = Net(**model_params).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"trainable params: {n_params}")

    optimizer_ = optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_, mode="max", factor=0.3, patience=3)

    scaler = torch.amp.GradScaler("cuda") if amp else None
    ema = ModelEMA(model, decay=args.ema_decay, unbias=True, device=device)

    curriculum = None
    if args.enable_curriculum:
        curriculum = IndividualCurriculumTable(
            decay=args.curriculum_decay,
            delta_decay=args.curriculum_delta_decay,
            delta_threshold=args.curriculum_delta_threshold,
            low=args.curriculum_low,
            high=args.curriculum_high,
            cooldown_epochs=args.curriculum_cooldown_epochs,
        )
        log.info(f"★ 课程表已启用：decay={curriculum.decay}  "
                 f"delta_decay={curriculum.delta_decay}  "
                 f"delta_threshold={curriculum.delta_threshold}  "
                 f"low={curriculum.low}  high={curriculum.high}  "
                 f"cooldown_epochs={curriculum.cooldown_epochs}")

    start_epoch = 0
    best_metric = -1e9
    patience_count = 0
    history = []

    if args.resume:
        resume_result = _try_load_resume(
            ckpt_dir, model, optimizer_, scheduler, ema, device)
        if resume_result is not None:
            (start_epoch, best_metric, patience_count, history,
             curriculum_state) = resume_result
            if curriculum is not None and curriculum_state is not None:
                curriculum.load_state_dict(curriculum_state)
                log.info(f"[resume] restored curriculum: "
                         f"{len(curriculum.weight)} individuals")
            log.info(f"[resume] will continue from epoch {start_epoch} "
                     f"to {args.max_epochs}")
        else:
            log.info("[resume] no valid checkpoint, training from scratch")
    else:
        log.info("[resume] disabled (--resume 0), training from scratch")

    epoch = start_epoch

    for epoch in range(start_epoch, args.max_epochs):
        losses = _train_one_epoch(
            model, tr_loader, optimizer_, scaler, ema, device, amp,
            clip_val=args.clip_val, epoch=epoch,
            curriculum=curriculum,
            grad_scale_enable=args.enable_curriculum,
        )

        if curriculum is not None:
            stats = curriculum.stats()
            if stats:
                losses["cur_n"] = float(stats["n_individuals"])
                losses["cur_hi"] = float(stats["frac_high"])
                losses["cur_lo"] = float(stats["frac_low"])
                losses["cur_s"] = float(stats["score_mean"])
                losses["cur_d"] = float(stats["delta_mean"])

        train_str = "  ".join(f"{k}={v:.4f}" for k, v in losses.items())
        log.info(f"epoch {epoch:02d}  {train_str}")

        cur_state = curriculum.state_dict() if curriculum is not None else None

        if (epoch + 1) % args.eval_every != 0 and epoch != args.max_epochs - 1:
            history.append({"epoch": epoch, **losses})
            _save_checkpoint(ckpt_dir, "last", model, {
                "epoch": epoch,
                "best_metric": best_metric,
                "patience_count": patience_count,
                "history": history,
            }, ema_state=ema.state_dict(),
               optimizer_state=optimizer_.state_dict(),
               scheduler_state=scheduler.state_dict(),
               curriculum_state=cur_state)
            continue

        backup = ema.apply_to(model)
        try:
            cv_records = _evaluate(model, cv_loader, device, amp=False)
        finally:
            ema.restore(model, backup)
        cv_summary = _summarize(cv_records)
        if not cv_summary:
            continue
        log.info(f"  cv: {_fmt_summary(cv_summary)}")

        metric_key = args.base_metric
        if metric_key not in cv_summary:
            for k in ["si_sdri", "scale_invariant_signal_noise_ratio",
                      "signal_noise_ratio", "loss"]:
                if k in cv_summary:
                    metric_key = k
                    break
        metric = cv_summary.get(metric_key, -1e9)
        scheduler.step(metric)
        history.append({"epoch": epoch, **losses, "cv": cv_summary})

        if metric > best_metric + 1e-4:
            best_metric = metric
            patience_count = 0
            backup = ema.apply_to(model)
            try:
                _save_checkpoint(ckpt_dir, "best", model, {
                    "epoch": epoch, "cv_summary": cv_summary,
                    "model_params": model_params,
                    "best_metric": best_metric,
                    "patience_count": patience_count,
                    "history": history,
                }, ema_state=ema.state_dict(),
                   curriculum_state=cur_state)
            finally:
                ema.restore(model, backup)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                log.info(f"早停：连续 {args.patience} 次无提升")
                _save_checkpoint(ckpt_dir, "last", model, {
                    "epoch": epoch,
                    "best_metric": best_metric,
                    "patience_count": patience_count,
                    "history": history,
                }, ema_state=ema.state_dict(),
                   optimizer_state=optimizer_.state_dict(),
                   scheduler_state=scheduler.state_dict(),
                   curriculum_state=cur_state)
                break

        _save_checkpoint(ckpt_dir, "last", model, {
            "epoch": epoch,
            "best_metric": best_metric,
            "patience_count": patience_count,
            "history": history,
        }, ema_state=ema.state_dict(),
           optimizer_state=optimizer_.state_dict(),
           scheduler_state=scheduler.state_dict(),
           curriculum_state=cur_state)

        (ckpt_dir / "history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    _save_checkpoint(ckpt_dir, "last", model, {
        "epoch": epoch,
        "best_metric": best_metric,
        "patience_count": patience_count,
        "history": history,
    }, ema_state=ema.state_dict(),
       optimizer_state=optimizer_.state_dict(),
       scheduler_state=scheduler.state_dict(),
       curriculum_state=(curriculum.state_dict()
                         if curriculum is not None else None))

    (ckpt_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(
            {k: v.to(device) for k, v in ckpt["state"].items()}, strict=False)
        log.info(f"\n加载 best: epoch={ckpt['meta']['epoch']}  "
                 f"cv_{args.base_metric}={best_metric:.3f}")

    log.info("\n===== Final Test =====")
    tt_records = _evaluate(model, tt_loader, device, amp=False)
    tt_summary = _summarize(tt_records)
    log.info(f"tt: {_fmt_summary(tt_summary)}")
    (ckpt_dir / "tt_records.json").write_text(
        json.dumps(tt_records, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"saved to {ckpt_dir}")


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--ckpt_dir", type=str, default="./experiments/frnet_e6_v2")
    p.add_argument("--sample_rate", type=int, default=8000)
    p.add_argument("--mix_seconds", type=float, default=4.0)
    p.add_argument("--enroll_seconds", type=float, default=4.0)
    p.add_argument("--norm_ref", type=int, default=1, choices=[0, 1])
    p.add_argument("--vad_enable", type=int, default=0, choices=[0, 1])

    p.add_argument("--label_len", type=int, default=32000)
    p.add_argument("--ch_dim", type=int, default=64)
    p.add_argument("--n_fft", type=int, default=256)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--label_emb_dim", type=int, default=512)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--compress_factor", type=float, default=0.3)
    p.add_argument("--fdrc_factor", type=float, default=0.5)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ema_decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip_val", type=float, default=0.0)
    p.add_argument("--amp", type=int, default=1, choices=[0, 1])
    p.add_argument("--base_metric", type=str, default="si_sdri")

    p.add_argument("--resume", type=int, default=1, choices=[0, 1])

    p.add_argument("--enable_curriculum", type=int, default=1, choices=[0, 1])
    p.add_argument("--curriculum_decay", type=float, default=0.9)
    p.add_argument("--curriculum_delta_decay", type=float, default=0.9)
    p.add_argument("--curriculum_delta_threshold", type=float, default=0.05)
    p.add_argument("--curriculum_low", type=float, default=0.3)
    p.add_argument("--curriculum_high", type=float, default=1.0)
    p.add_argument("--curriculum_cooldown_epochs", type=int, default=5)

    return p


if __name__ == "__main__":
    args = _build_argparser().parse_args()
    args.norm_ref = bool(args.norm_ref)
    args.vad_enable = bool(args.vad_enable)
    args.amp = bool(args.amp)
    if args.compress_factor is not None:
        args.compress_factor = float(args.compress_factor)
    run_standalone(args)