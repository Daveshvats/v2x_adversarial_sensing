#!/usr/bin/env python3
"""
V2X Adversarial Spectrum Sensing — Rician Fading & Doppler Edition
===================================================================
Exact same model, data pipeline, training recipe, and attacks as v3.py,
with UPGRADED channel models:
  - Rician fading (configurable K-factor)
  - Doppler shift (vehicular speed 0-120 km/h at 5.9 GHz)
  - Frequency-selective fading (2-4 tap multipath)
  - Multi-channel mode (random mix of all channel types)

Usage:
  python v3_rician_doppler.py                      # multi-channel mode (default)
  python v3_rician_doppler.py --mode rayleigh       # original rayleigh only
  python v3_rician_doppler.py --mode rician --K 10  # fixed Rician K=10
  python v3_rician_doppler.py --compare             # rayleigh vs multi comparison
  python v3_rician_doppler.py --seed 42 --epochs 50 # custom parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import json, os, time, copy, warnings, math, argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION (matches v3.py + channel upgrades)
# =============================================================================
SEEDS          = [42, 123, 456]
NUM_SAMPLES    = 4000
SAMPLE_LENGTH  = 1024
FS             = 10e6
NFFT           = 128
HOP            = 64
NUM_CLASSES    = 4
CLASS_NAMES    = ["LTE", "WiFi", "V2X-PC5", "Noise"]
BATCH_SIZE     = 32
EPOCHS         = 100
LR             = 5e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
WARMUP_EPOCHS  = 5
PATIENCE       = 15

EPSILON        = 0.03
PGD_STEPS      = 20
PGD_STEP_SIZE  = EPSILON / 4
CW_ITERS       = 40
CW_LR          = 0.005
CW_KAPPA       = 0.0

AT_MIX_FRAC    = 0.5
GAUSS_STD      = 0.02
GAUSS_PROB     = 0.3
CUTMIX_ALPHA   = 1.0
CUTMIX_PROB    = 0.3

SNR_RANGE      = [-5, 0, 5, 10, 15, 20, 25, 30]

# Channel upgrades (NEW) — replace simple Rayleigh with more realistic models
# "multi" mode randomly mixes channel types to simulate heterogeneous V2X environments
CHANNEL_MODE   = "multi"        # "rayleigh", "rician", "multi"
# Rician K-factor in dB: ratio of LOS to NLOS power (higher = stronger LOS component)
RICIAN_K       = 6              # Rician K-factor for "rician" mode
DOPPLER_ENABLE = True           # Apply Doppler shift (time-varying phase from vehicle motion)
DOPPLER_V_MIN  = 0              # Min vehicle speed (km/h)
DOPPLER_V_MAX  = 120            # Max vehicle speed (km/h)
V2X_FC         = 5.9e9          # V2X carrier frequency (Hz) — DSRC/C-V2X ITS band
C_LIGHT        = 3e8            # Speed of light (m/s)

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3_channel_data")
FIGURES_DIR  = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# UTILITIES
# =============================================================================

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    return obj

# =============================================================================
# CHANNEL MODELS (NEW — Rician, Doppler, Frequency-Selective)
# =============================================================================

def rayleigh_channel(n, rng):
    """Standard Rayleigh fading: h = (randn + j*randn) / sqrt(2).
    
    Rayleigh fading arises when there is no dominant LOS path and the received
    signal is a sum of many scattered components (central limit theorem).
    The /sqrt(2) normalization ensures E[|h|²] = 1 (unit average gain).
    """
    return (rng.randn(n) + 1j * rng.randn(n)) / np.sqrt(2)


def rician_channel(n, K_dB, rng):
    """Rician fading: h = sqrt(K/(K+1)) * h_LOS + sqrt(1/(K+1)) * h_NLOS.
    
    K_dB: Rician K-factor in dB (linear: K_lin = 10^(K_dB/10))
    h_LOS: deterministic line-of-sight component (all ones = perfect LOS)
    h_NLOS: Rayleigh-distributed scattered component
    
    Power conservation: E[|h|²] = K/(K+1) + 1/(K+1) = 1
    """
    K_lin = 10.0 ** (K_dB / 10.0)
    h_los = np.ones(n, dtype=np.complex128)  # deterministic LOS
    h_nlos = (rng.randn(n) + 1j * rng.randn(n)) / np.sqrt(2)
    return np.sqrt(K_lin / (K_lin + 1)) * h_los + np.sqrt(1 / (K_lin + 1)) * h_nlos


def apply_doppler(h, fs, v_kmh):
    """Apply time-varying Doppler shift to the channel: h_doppler(t) = h(t) * exp(2j*pi*f_d*t/fs).
    
    The Doppler frequency is: f_d = v * f_c / c where v is vehicle speed in m/s.
    This models the phase rotation caused by relative motion between
    transmitter and receiver in V2X scenarios.
    """
    v_ms = v_kmh / 3.6  # Convert km/h to m/s
    f_d = v_ms * V2X_FC / C_LIGHT  # Doppler frequency at 5.9 GHz
    t = np.arange(len(h), dtype=np.float64)
    return h * np.exp(2j * np.pi * f_d * t / fs)


def freq_selective_channel(n, rng, n_taps=None):
    """Frequency-selective fading: 2-4 tap multipath with exponential PDP.
    
    Models a channel with multiple delayed reflections where the power of
    each tap decreases exponentially with delay (exponential Power Delay Profile).
    This causes frequency-selective fading: certain frequency bands are
    attenuated more than others, creating spectral nulls.
    
    The np.roll operation shifts each tap's random process to simulate
    different propagation delays. Power normalization ensures unit gain.
    """
    if n_taps is None:
        n_taps = rng.randint(2, 5)  # 2, 3, or 4 taps
    # Exponential power delay profile: power decreases as exp(-0.5 * delay)
    delays = sorted(rng.randint(0, 10, size=n_taps))  # sample delays in grid units
    powers = np.array([np.exp(-0.5 * d) for d in delays])
    powers /= np.sqrt(np.sum(powers ** 2))  # normalize for unit average power

    h = np.zeros(n, dtype=np.complex128)
    for delay, power in zip(delays, powers):
        tap = (rng.randn(n) + 1j * rng.randn(n)) / np.sqrt(2)  # Rayleigh tap
        shifted = np.roll(tap, delay)  # apply delay via circular shift
        h += power * shifted
    return h


def apply_channel(signal, channel_mode, rng):
    """Apply channel fading + optional Doppler to complex baseband signal.
    
    In "multi" mode, the channel type is randomly sampled to simulate
    a heterogeneous deployment where vehicles experience different
    propagation conditions:
      - 40% Rayleigh: urban NLOS (no dominant LOS path)
      - 30% Rician K=3: urban V2I (weak LOS from buildings)
      - 20% Rician K=6: highway V2I (moderate LOS)
      - 10% Freq-selective: hilly/rural terrain (multipath)
    
    After fading, channel power is normalized to ensure fair SNR control.
    """
    n = len(signal)

    if channel_mode == "rayleigh":
        h = rayleigh_channel(n, rng)
    elif channel_mode == "rician":
        h = rician_channel(n, RICIAN_K, rng)
    elif channel_mode == "multi":
        roll = rng.rand()
        if roll < 0.40:
            h = rayleigh_channel(n, rng)
        elif roll < 0.70:
            h = rician_channel(n, 3.0, rng)   # urban V2I
        elif roll < 0.90:
            h = rician_channel(n, 6.0, rng)   # highway V2I
        else:
            h = freq_selective_channel(n, rng)
    else:
        raise ValueError(f"Unknown channel_mode: {channel_mode}")

    if DOPPLER_ENABLE:
        v_kmh = rng.uniform(DOPPLER_V_MIN, DOPPLER_V_MAX)
        h = apply_doppler(h, FS, v_kmh)

    # Normalize channel power
    pwr = np.mean(np.abs(h) ** 2)
    if pwr > 1e-12:
        h /= np.sqrt(pwr)

    return signal * h

# =============================================================================
# 1. DATASET GENERATION (v3.py signal gen + upgraded channels)
# =============================================================================

def generate_v2x_dataset(seed=42, channel_mode="rayleigh"):
    """Generate synthetic V2X data with configurable channel model."""
    rng = np.random.RandomState(seed)
    n_per_class = NUM_SAMPLES // NUM_CLASSES
    n_frames = (SAMPLE_LENGTH - NFFT) // HOP + 1
    n_freq   = NFFT // 2 + 1
    window   = np.hanning(NFFT).astype(np.float64)

    mag_raw = np.zeros((NUM_SAMPLES, n_freq, n_frames), dtype=np.float64)
    if_raw  = np.zeros((NUM_SAMPLES, n_freq, n_frames), dtype=np.float64)
    y_all   = np.zeros(NUM_SAMPLES, dtype=np.int64)

    idx = 0
    for cls in range(NUM_CLASSES):
        for _ in range(n_per_class):
            t = np.arange(SAMPLE_LENGTH, dtype=np.float64) / FS

            # --- Signal generation (exact v3.py) ---
            if cls == 0:  # LTE
                n_sub = 64
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.35 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub)
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            elif cls == 1:  # WiFi
                n_sub = 52
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.25 * FS
                offset = 0.15 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub) + offset
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            elif cls == 2:  # V2X-PC5
                n_sub = 12
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.08 * FS
                offset = -0.10 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub) + offset
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            else:  # Noise
                sig = (rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)).astype(np.complex128)

            # --- Channel (upgraded) ---
            sig = apply_channel(sig, channel_mode, rng)

            # --- AWGN ---
            snr_db = rng.uniform(5, 25)
            snr_lin = 10.0 ** (snr_db / 10.0)
            noise_pwr = np.var(sig).item() / max(snr_lin, 1e-12)
            noise = np.sqrt(max(noise_pwr, 0)) * (
                rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)) / np.sqrt(2)
            sig = (sig + noise).astype(np.complex128)

            # --- STFT (exact v3.py) ---
            for f in range(n_frames):
                start = f * HOP
                chunk = sig[start: start + NFFT]
                windowed = (chunk * window).astype(np.complex128)
                spectrum = np.fft.fft(windowed, n=NFFT)
                pos = spectrum[:n_freq]
                mag_raw[idx, :, f] = np.log10(np.abs(pos) + 1e-10)
                if_raw[idx, :, f] = np.angle(pos)

            # Instantaneous frequency (exact v3.py)
            phases = if_raw[idx, :, :]
            phases_unwrapped = np.unwrap(phases, axis=1)
            inst_freq = np.diff(phases_unwrapped, axis=1)
            if_raw[idx, :, 1:] = inst_freq
            if_raw[idx, :, 0] = 0.0
            y_all[idx] = cls
            idx += 1

    # Train / Test split
    indices = np.arange(NUM_SAMPLES)
    rng.shuffle(indices)
    n_train = int(0.8 * NUM_SAMPLES)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Z-score (training set only)
    mag_train = mag_raw[train_idx]
    mag_mean, mag_std = mag_train.mean(), mag_train.std() + 1e-8
    mag_norm = np.clip((mag_raw - mag_mean) / mag_std, -5, 5)

    if_train = if_raw[train_idx]
    if_mean, if_std = if_train.mean(), if_train.std() + 1e-8
    if_norm = np.clip((if_raw - if_mean) / if_std, -5, 5)

    X_mag = torch.from_numpy(mag_norm).unsqueeze(1).float()
    X_if  = torch.from_numpy(if_norm).unsqueeze(1).float()
    y     = torch.from_numpy(y_all).long()
    return X_mag, X_if, y, train_idx, val_idx

# =============================================================================
# 2. MODEL (exact v3.py — 86,052 params)
# =============================================================================

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        c4, c12 = out_ch // 4, out_ch // 2
        self.branch1 = nn.Conv2d(in_ch, c4,  1, bias=False)
        self.branch3 = nn.Conv2d(in_ch, c4,  3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, c12, 5, padding=2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat([
            self.branch1(x), self.branch3(x), self.branch5(x)
        ], dim=1)))

class SingleStream(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.inc1  = InceptionBlock(16, 32)
        self.inc2  = InceptionBlock(32, 64)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.inc1(x)
        x = F.max_pool2d(x, 2)
        x = self.inc2(x)
        x = self.pool(x)
        return x.flatten(1)

class DualStreamModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mag_stream = SingleStream()
        self.if_stream  = SingleStream()
        self.fc1     = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        f_mag = self.mag_stream(mag)
        f_if  = self.if_stream(ift)
        x = torch.cat([f_mag, f_if], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

# =============================================================================
# 3. AUGMENTATION + TRAINING (exact v3.py)
# =============================================================================

def tf_cutmix(mag, ift, y, alpha=CUTMIX_ALPHA):
    if np.random.rand() > CUTMIX_PROB:
        return mag, ift, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = mag.size()
    cut_ratio = np.sqrt(1.0 - lam)
    rw, rh = int(W * cut_ratio), int(H * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    rand_idx = torch.randperm(B, device=mag.device)
    mag_out, if_out = mag.clone(), ift.clone()
    mag_out[:, :, y1:y2, x1:x2] = mag[rand_idx, :, y1:y2, x1:x2]
    if_out[:, :, y1:y2, x1:x2]  = ift[rand_idx, :, y1:y2, x1:x2]
    actual_lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)
    return mag_out, if_out, y, y[rand_idx], actual_lam

def get_lr_lambda(epoch, warmup=WARMUP_EPOCHS, total=EPOCHS):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train_standard(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    best_acc, best_state, wait = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for mag, ift, y in train_loader:
            mag, ift, y = mag.to(DEVICE), ift.to(DEVICE), y.to(DEVICE)
            if np.random.rand() < GAUSS_PROB:
                mag = (mag + GAUSS_STD * torch.randn_like(mag)).clamp(-5, 5)
                ift = (ift + GAUSS_STD * torch.randn_like(ift)).clamp(-5, 5)
            mag, ift, ya, yb, lam = tf_cutmix(mag, ift, y)
            optimizer.zero_grad()
            logits = model(mag, ift)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mag, ift, y in val_loader:
                mag, ift, y = mag.to(DEVICE), ift.to(DEVICE), y.to(DEVICE)
                correct += (model(mag, ift).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return best_acc

def train_adversarial(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    best_acc, best_state, wait = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for mag, ift, y in train_loader:
            mag, ift, y = mag.to(DEVICE), ift.to(DEVICE), y.to(DEVICE)
            if np.random.rand() < AT_MIX_FRAC:
                mag_in, if_in = pgd_attack(model, mag, ift, y, steps=7, step_size=PGD_STEP_SIZE)
                mag_in, if_in = mag_in.detach(), if_in.detach()
            else:
                mag_in, if_in = mag, ift
            if np.random.rand() < GAUSS_PROB:
                mag_in = (mag_in + GAUSS_STD * torch.randn_like(mag_in)).clamp(-5, 5)
                if_in  = (if_in  + GAUSS_STD * torch.randn_like(if_in)).clamp(-5, 5)
            mag_in, if_in, ya, yb, lam = tf_cutmix(mag_in, if_in, y)
            optimizer.zero_grad()
            logits = model(mag_in, if_in)
            loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for mag, ift, y in val_loader:
                mag, ift, y = mag.to(DEVICE), ift.to(DEVICE), y.to(DEVICE)
                correct += (model(mag, ift).argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    Early stop epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return best_acc

# =============================================================================
# 4. ATTACKS (exact v3.py)
# =============================================================================

def fgsm_attack(model, mag, ift, y, eps=EPSILON):
    was_training = model.training; model.eval()
    mag_a = mag.clone().detach().requires_grad_(True)
    if_a  = ift.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(mag_a, if_a), y); loss.backward()
    mag_p = (mag + eps * mag_a.grad.sign()).clamp(-5, 5).detach()
    if_p  = (ift  + eps * if_a.grad.sign()).clamp(-5, 5).detach()
    if was_training: model.train()
    return mag_p, if_p

def pgd_attack(model, mag, ift, y, eps=EPSILON, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    was_training = model.training; model.eval()
    mag_delta = torch.zeros_like(mag, requires_grad=True)
    if_delta  = torch.zeros_like(ift,  requires_grad=True)
    for _ in range(steps):
        logits = model((mag + mag_delta).clamp(-5, 5), (ift + if_delta).clamp(-5, 5))
        loss = F.cross_entropy(logits, y); loss.backward()
        mag_delta.data = torch.clamp(mag_delta + step_size * mag_delta.grad.sign(), -eps, eps)
        if_delta.data  = torch.clamp(if_delta  + step_size * if_delta.grad.sign(),  -eps, eps)
        mag_delta.grad.zero_(); if_delta.grad.zero_()
    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p  = (ift  + if_delta).clamp(-5, 5).detach()
    if was_training: model.train()
    return mag_p, if_p

def cw_attack(model, mag, ift, y, eps=EPSILON, iters=CW_ITERS, lr=CW_LR, kappa=CW_KAPPA):
    was_training = model.training; model.eval()
    mag_delta = torch.zeros_like(mag, requires_grad=True)
    if_delta  = torch.zeros_like(ift,  requires_grad=True)
    optimizer = optim.Adam([mag_delta, if_delta], lr=lr)
    for _ in range(iters):
        logits = model((mag + mag_delta).clamp(-5, 5), (ift + if_delta).clamp(-5, 5))
        correct_l = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        wrong_logits = logits.clone(); wrong_logits.scatter_(1, y.unsqueeze(1), -1e9)
        best_wrong = wrong_logits.max(dim=1)[0]
        loss = F.relu(best_wrong - correct_l + kappa).sum()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        mag_delta.data.clamp_(-eps, eps); if_delta.data.clamp_(-eps, eps)
    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p  = (ift  + if_delta).clamp(-5, 5).detach()
    if was_training: model.train()
    return mag_p, if_p

def attack_success_rate(model, mag, ift, y):
    model.eval()
    with torch.no_grad():
        return (model(mag, ift).argmax(1) != y).float().mean().item()

def channel_aware_eval(model, mag, ift, y, attack_fn, snr_range=SNR_RANGE):
    model.eval()
    mag_adv, if_adv = attack_fn(model, mag, ift, y)
    base_asr = attack_success_rate(model, mag_adv, if_adv, y)
    results = {"no_channel": base_asr, "per_snr": {}}
    for snr_db in snr_range:
        snr_lin = 10.0 ** (snr_db / 10.0)
        mag_noise_std = math.sqrt(max((mag_adv ** 2).mean().item() / snr_lin, 0))
        if_noise_std  = math.sqrt(max((if_adv ** 2).mean().item() / snr_lin, 0))
        mag_noisy = (mag_adv + mag_noise_std * torch.randn_like(mag_adv)).clamp(-5, 5)
        if_noisy  = (if_adv  + if_noise_std  * torch.randn_like(if_adv)).clamp(-5, 5)
        results["per_snr"][snr_db] = attack_success_rate(model, mag_noisy, if_noisy, y)
    return results

# =============================================================================
# 5. SINGLE-SEED EXPERIMENT
# =============================================================================

def run_experiment(channel_mode, seed, out_dir):
    print(f"\n{'='*60}")
    print(f"  V3 CHANNEL UPGRADE — {channel_mode.upper()} — SEED {seed}")
    print(f"{'='*60}")
    set_seed(seed); t0 = time.time()

    print(f"\n[1/5] Generating dataset (channel={channel_mode})...")
    X_mag, X_if, y, train_idx, val_idx = generate_v2x_dataset(seed=seed, channel_mode=channel_mode)
    mag_train, if_train, y_train = X_mag[train_idx], X_if[train_idx], y[train_idx]
    mag_val, if_val, y_val = X_mag[val_idx], X_if[val_idx], y[val_idx]
    train_ds = TensorDataset(mag_train, if_train, y_train)
    val_ds   = TensorDataset(mag_val, if_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    mag_dev, if_dev, y_dev = mag_val.to(DEVICE), if_val.to(DEVICE), y_val.to(DEVICE)

    print(f"\n[2/5] Standard training...")
    model_std = DualStreamModel().to(DEVICE)
    n_params = sum(p.numel() for p in model_std.parameters())
    std_acc = train_standard(model_std, train_loader, val_loader)
    print(f"  Params={n_params:,}, Acc={std_acc*100:.2f}%")

    print(f"\n[3/5] Adversarial training...")
    model_at = DualStreamModel().to(DEVICE)
    at_acc = train_adversarial(model_at, train_loader, val_loader)
    print(f"  Acc={at_acc*100:.2f}%")

    eps_list = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    print(f"\n[4/5] Attack evaluation...")
    fgsm_per_eps, pgd_per_eps = {}, {}
    for eps in eps_list:
        m_a, i_a = fgsm_attack(model_std, mag_dev, if_dev, y_dev, eps=eps)
        fgsm_per_eps[eps] = attack_success_rate(model_std, m_a, i_a, y_dev)
        m_a, i_a = pgd_attack(model_std, mag_dev, if_dev, y_dev, eps=eps)
        pgd_per_eps[eps] = attack_success_rate(model_std, m_a, i_a, y_dev)
        print(f"  eps={eps:.3f}: FGSM={fgsm_per_eps[eps]*100:.2f}% PGD={pgd_per_eps[eps]*100:.2f}%")

    m_cw, i_cw = cw_attack(model_std, mag_dev, if_dev, y_dev, eps=EPSILON)
    cw_asr = attack_success_rate(model_std, m_cw, i_cw, y_dev)
    print(f"  C&W eps={EPSILON}: ASR={cw_asr*100:.2f}%")

    print(f"\n[5/5] Channel-aware eval...")
    ch_fgsm = channel_aware_eval(model_std, mag_dev, if_dev, y_dev,
                                 lambda m, mi, ii, yi: fgsm_attack(m, mi, ii, yi))
    ch_pgd = channel_aware_eval(model_std, mag_dev, if_dev, y_dev,
                                lambda m, mi, ii, yi: pgd_attack(m, mi, ii, yi))

    elapsed = time.time() - t0
    print(f"\n  Seed {seed} done in {elapsed:.1f}s")

    return {
        "seed": seed, "channel_mode": channel_mode, "n_params": n_params,
        "std_clean_acc": std_acc, "at_clean_acc": at_acc,
        "fgsm_asr": fgsm_per_eps.get(EPSILON, 0), "pgd_asr": pgd_per_eps.get(EPSILON, 0),
        "cw_asr": cw_asr, "fgsm_per_eps": fgsm_per_eps, "pgd_per_eps": pgd_per_eps,
        "channel_aware_fgsm": ch_fgsm, "channel_aware_pgd": ch_pgd,
        "elapsed_s": elapsed,
    }

# =============================================================================
# 6. COMPARISON MODE
# =============================================================================

def run_comparison(seeds, out_dir):
    """Run both rayleigh and multi-channel, output comparison table."""
    all_results = {"rayleigh": [], "multi": []}
    for ch_mode in ["rayleigh", "multi"]:
        for seed in seeds:
            result = run_experiment(ch_mode, seed, out_dir)
            all_results[ch_mode].append(result)
            path = os.path.join(out_dir, f"v3_{ch_mode}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump(to_serializable(result), f, indent=2)

    # Print comparison
    print("\n" + "=" * 80)
    print("  CHANNEL MODEL COMPARISON")
    print("=" * 80)
    print(f"  {'Metric':>25s}  {'Rayleigh':>15s}  {'Multi-Channel':>15s}  {'Δ':>10s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*15}  {'-'*10}")

    for key, label in [
        ("std_clean_acc", "Clean Accuracy"),
        ("at_clean_acc", "Adv. Training Acc"),
        ("fgsm_asr", "FGSM ASR"),
        ("pgd_asr", "PGD ASR"),
        ("cw_asr", "C&W ASR"),
    ]:
        ray_mean = np.mean([r[key] for r in all_results["rayleigh"]])
        mul_mean = np.mean([r[key] for r in all_results["multi"]])
        delta = mul_mean - ray_mean
        print(f"  {label:>25s}  {ray_mean*100:>14.2f}%  {mul_mean*100:>14.2f}%  {delta*100:>+9.2f}pp")

    agg_path = os.path.join(out_dir, "channel_comparison.json")
    with open(agg_path, "w") as f:
        json.dump(to_serializable(all_results), f, indent=2)
    print(f"\n  Saved: {agg_path}")

# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    global CHANNEL_MODE, RICIAN_K, DOPPLER_ENABLE, DOPPLER_V_MIN, DOPPLER_V_MAX

    parser = argparse.ArgumentParser(description="V3 with Rician/Doppler channels")
    parser.add_argument("--mode", type=str, default=CHANNEL_MODE, choices=["rayleigh", "rician", "multi"])
    parser.add_argument("--K", type=float, default=RICIAN_K, help="Rician K-factor (dB)")
    parser.add_argument("--no-doppler", action="store_true", help="Disable Doppler")
    parser.add_argument("--seed", type=int, default=None, help="Single seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--compare", action="store_true", help="Run rayleigh vs multi comparison")
    args = parser.parse_args()

    CHANNEL_MODE = args.mode
    RICIAN_K = args.K
    DOPPLER_ENABLE = not args.no_doppler
    seeds = args.seeds if args.seeds else ([args.seed] if args.seed else SEEDS)

    print("=" * 60)
    print("  V2X ADVERSARIAL SPECTRUM SENSING")
    print("  RICIAN FADING & DOPPLER EDITION")
    print("=" * 60)
    print(f"  Device:       {DEVICE}")
    print(f"  Channel Mode: {CHANNEL_MODE}")
    print(f"  Rician K:     {RICIAN_K}" if CHANNEL_MODE == "rician" else "")
    print(f"  Doppler:      {'ON' if DOPPLER_ENABLE else 'OFF'} ({DOPPLER_V_MIN}–{DOPPLER_V_MAX} km/h)")
    print(f"  Seeds:        {seeds}")
    print(f"  Output:       {OUTPUT_DIR}")

    if args.compare:
        run_comparison(seeds, OUTPUT_DIR)
    else:
        all_results = []
        for seed in seeds:
            result = run_experiment(CHANNEL_MODE, seed, OUTPUT_DIR)
            all_results.append(result)
            path = os.path.join(OUTPUT_DIR, f"v3_{CHANNEL_MODE}_seed{seed}.json")
            with open(path, "w") as f:
                json.dump(to_serializable(result), f, indent=2)

        agg = {
            "channel_mode": CHANNEL_MODE,
            "std_clean_acc": f"{np.mean([r['std_clean_acc'] for r in all_results])*100:.2f} ± {np.std([r['std_clean_acc'] for r in all_results])*100:.2f}%",
            "fgsm_asr": f"{np.mean([r['fgsm_asr'] for r in all_results])*100:.2f} ± {np.std([r['fgsm_asr'] for r in all_results])*100:.2f}%",
            "pgd_asr": f"{np.mean([r['pgd_asr'] for r in all_results])*100:.2f} ± {np.std([r['pgd_asr'] for r in all_results])*100:.2f}%",
            "cw_asr": f"{np.mean([r['cw_asr'] for r in all_results])*100:.2f} ± {np.std([r['cw_asr'] for r in all_results])*100:.2f}%",
        }
        print(f"\n  SUMMARY: {agg}")
        agg_path = os.path.join(OUTPUT_DIR, f"v3_{CHANNEL_MODE}_aggregated.json")
        with open(agg_path, "w") as f:
            json.dump(to_serializable(agg), f, indent=2)

if __name__ == "__main__":
    main()
