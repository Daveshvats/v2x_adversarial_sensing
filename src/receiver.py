"""receiver.py — differentiable receiver front-end and model definitions.

The ENTIRE receiver chain lives in torch so that gradients flow from the model loss
back to a time-domain (waveform-domain) adversarial perturbation:

    delta(t)  ->  channel convolution  ->  + noise  ->  torch.stft  ->
    log-magnitude + instantaneous-frequency streams  ->  CNN  ->  logits

This is what makes the physically-realizable attack (attacker transmits a waveform,
it propagates through the attacker->victim channel, the receiver senses the sum)
differentiable end-to-end.

Models (reuse of the unpublished ICE2CT-2026 submission's v3 architecture, input-size agnostic):
  DualStreamModel : log-mag stream || IF stream   (86,052 params on 256x15 inputs)
  MagOnlyModel    : log-mag stream only           (ablation / honesty baseline)

Feature naming honesty: the "IF" stream is a per-STFT-bin temporal phase
 difference (a phase-vocoder-style frequency-offset estimate, unambiguous
 range +-78.125 kHz = half the STFT bin) — not a per-sample instantaneous
 frequency. The C5 ablation shows it contributes ~nothing on this task.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from waveforms import FS, N_SAMPLES, NFFT_STFT, HOP_STFT, NUM_CLASSES

EPS_STAB = 1e-10


class FrontEnd(nn.Module):
    """Waveform (B, L) complex -> (mag, ifr) streams, each (B, 1, F, T).

    mag  = log10(|STFT| + eps), z-scored with registered buffers (train-set stats)
    ifr  = wrapped temporal phase difference per STFT bin (a phase-vocoder
           frequency-offset estimate; see the naming note in the module
           docstring), same shape.
    """

    def __init__(self):
        super().__init__()
        window = torch.hann_window(NFFT_STFT)
        self.register_buffer("window", window)
        self.register_buffer("mag_mean", torch.zeros(1))
        self.register_buffer("mag_std", torch.ones(1))

    def set_stats(self, mag_mean, mag_std):
        self.mag_mean.copy_(torch.as_tensor(mag_mean, dtype=torch.float32))
        self.mag_std.copy_(torch.as_tensor(mag_std, dtype=torch.float32))

    def forward(self, wave: torch.Tensor):
        # wave: (B, L) complex64/complex128
        # complex input requires onesided=False; fftshift makes the freq axis
        # monotonically ordered from -10 MHz to +10 MHz
        Z = torch.stft(wave, n_fft=NFFT_STFT, hop_length=HOP_STFT,
                       window=self.window, return_complex=True,
                       center=False, onesided=False)          # (B, F, T)
        Z = torch.fft.fftshift(Z, dim=1)
        mag = torch.log10(torch.abs(Z) + EPS_STAB)
        mag = (mag - self.mag_mean) / self.mag_std
        mag = torch.clamp(mag, -5.0, 5.0)

        phi = torch.angle(Z)
        dphi = phi[:, :, 1:] - phi[:, :, :-1]
        dphi = torch.remainder(dphi + np.pi, 2 * np.pi) - np.pi
        dphi = torch.cat([dphi, dphi[:, :, -1:]], dim=2)     # pad to T frames

        return mag.unsqueeze(1), dphi.unsqueeze(1)


# --------------------------------------------------------------------------
# Models (exact v3 Inception blocks; spatial-size agnostic)
# --------------------------------------------------------------------------
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        c4 = out_ch // 4
        c12 = out_ch // 2
        self.branch1 = nn.Conv2d(in_ch, c4, 1, bias=False)
        self.branch3 = nn.Conv2d(in_ch, c4, 3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, c12, 5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat(
            [self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)))


class SingleStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.inc1 = InceptionBlock(16, 32)
        self.inc2 = InceptionBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.inc1(x)
        x = F.max_pool2d(x, 2)
        x = self.inc2(x)
        return self.pool(x).flatten(1)


class DualStreamModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mag_stream = SingleStream()
        self.if_stream = SingleStream()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, mag, ifr):
        z = torch.cat([self.mag_stream(mag), self.if_stream(ifr)], dim=1)
        z = self.dropout(F.relu(self.fc1(z)))
        return self.fc2(z)

    def forward_wave(self, wave, frontend):
        mag, ifr = frontend(wave)
        return self.forward(mag, ifr)


class MagOnlyModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stream = SingleStream()
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, mag, ifr=None):
        z = self.stream(mag)
        z = self.dropout(F.relu(self.fc1(z)))
        return self.fc2(z)

    def forward_wave(self, wave, frontend):
        mag, _ = frontend(wave)
        return self.forward(mag)


# --------------------------------------------------------------------------
# Training utilities (v3 recipe: label smoothing + Gaussian noise + TF-CutMix)
# --------------------------------------------------------------------------
def tf_cutmix(mag, ifr, y, alpha=1.0, same_mask=True):
    B = mag.size(0)
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    cut_ratio = np.sqrt(1.0 - lam)
    H, W = mag.size(2), mag.size(3)
    ch = int(H * cut_ratio), int(W * cut_ratio)
    if same_mask and (ch[0] < 1 or ch[1] < 1):
        return mag, ifr, y, y, 1.0
    perm = torch.randperm(B, device=mag.device)
    y_a, y_b = y, y[perm]
    if same_mask and (ch[0] >= 1 and ch[1] >= 1):
        y0 = torch.randint(0, H - max(ch[0], 1) + 1, (1,)).item()
        x0 = torch.randint(0, W - max(ch[1], 1) + 1, (1,)).item()
        mag = mag.clone()
        ifr = ifr.clone()
        mag[:, :, y0:y0 + ch[0], x0:x0 + ch[1]] = \
            mag[perm][:, :, y0:y0 + ch[0], x0:x0 + ch[1]]
        ifr[:, :, y0:y0 + ch[0], x0:x0 + ch[1]] = \
            ifr[perm][:, :, y0:y0 + ch[0], x0:x0 + ch[1]]
        lam = 1.0 - (ch[0] * ch[1]) / (H * W)
    return mag, ifr, y_a, y_b, lam


def smoothed_ce(logits, y_a, y_b, lam, n_classes=NUM_CLASSES, eps_ls=0.1):
    ce = F.cross_entropy(logits, y_a, label_smoothing=eps_ls)
    if lam >= 1.0:
        return ce
    return lam * ce + (1 - lam) * F.cross_entropy(logits, y_b,
                                                  label_smoothing=eps_ls)
