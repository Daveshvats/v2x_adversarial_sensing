#!/usr/bin/env python3
"""
Mobility Scenario-Stratified Adversarial Robustness Experiment
==============================================================
Tests how adversarial robustness changes across highway / urban / rural
mobility scenarios by replacing simple Rayleigh fading with the full
ChannelApplicator (Rician fading, Doppler spread, multipath, pathloss)
from generate_simulated_dataset.py.

Model architecture, training recipe, and signal-generation parameters are
EXACTLY copied from autoattack_eval.py to ensure comparability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import json, os, time, copy, warnings, math, argparse
from dataclasses import dataclass, field
from typing import List, Tuple

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION (matches autoattack_eval.py exactly)
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

AT_MIX_FRAC    = 0.5
GAUSS_STD      = 0.02
GAUSS_PROB     = 0.3
CUTMIX_ALPHA   = 1.0
CUTMIX_PROB    = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# SCENARIO-SPECIFIC CHANNEL PARAMETERS
# =============================================================================
# From generate_simulated_dataset.py ChannelModel:
#   highway: K=3-6 dB, shadowing sigma=4 dB, PDP spread=0.5 us
#   urban:   K=1-3 dB, shadowing sigma=10 dB, PDP spread=2.0 us
#   rural:   K=6-10 dB, shadowing sigma=6 dB, PDP spread=1.0 us
# Also Doppler derived from scenario speed profiles:
#   highway: 80-120 km/h  => high Doppler at 5.9 GHz
#   urban:   0-60 km/h    => variable Doppler
#   rural:   40-80 km/h   => moderate Doppler

V2X_FC = 5.9e9  # DSRC/C-V2X carrier frequency in Hz (ITS band 5.9 GHz)
C_LIGHT = 2.998e8  # Speed of light in m/s

# Scenario-specific channel parameters derived from 3GPP TR 36.885 V2X channel models:
#   highway: High K-factor (strong LOS), low shadowing (open road), compact PDP
#   urban: Low K-factor (NLOS from buildings), high shadowing, spread PDP
#   rural: High K-factor (strong LOS), moderate shadowing, moderate PDP
SCENARIO_PARAMS = {
    "highway": {
        "rician_k_range": (3.0, 6.0),
        "shadowing_sigma": 4.0,
        "pdp_spread": 0.5,
        "speed_range_kmh": (80, 120),
        "n_multipath": 4,
    },
    "urban": {
        "rician_k_range": (1.0, 3.0),
        "shadowing_sigma": 10.0,
        "pdp_spread": 2.0,
        "speed_range_kmh": (0, 60),
        "n_multipath": 4,
    },
    "rural": {
        "rician_k_range": (6.0, 10.0),
        "shadowing_sigma": 6.0,
        "pdp_spread": 1.0,
        "speed_range_kmh": (40, 80),
        "n_multipath": 4,
    },
}

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


def db_to_linear(db):
    return 10.0 ** (db / 10.0)


# =============================================================================
# CHANNEL APPLICATOR (from generate_simulated_dataset.py, Section 4)
# =============================================================================

@dataclass
class ChannelParams:
    """Channel parameters for a single sample."""
    pathloss_db: float = 100.0
    doppler_hz: float = 0.0
    rician_k_db: float = 5.0
    shadowing_db: float = 0.0
    snr_db: float = 15.0
    multipath_delays_us: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.3, 0.7, 1.5]))
    multipath_powers_db: np.ndarray = field(default_factory=lambda: np.array([0, -3, -7, -12]))


class ChannelApplicator:
    """
    Apply realistic V2X channel effects to baseband waveforms.
    Copied from generate_simulated_dataset.py Section 4.
    
    Effects applied (in order):
      1. Multipath (frequency-selective fading)
      2. Rician fading (small-scale, time-varying)
      3. Doppler spread
      4. Pathloss attenuation
      5. AWGN at target SNR
    """

    def apply_multipath(self, signal, delays_us, powers_db, rng):
        """Frequency-selective fading via tapped-delay-line model.
        
        Implements a FIR filter where each tap represents a reflected signal
        path with a specific delay and power. The tapped-delay-line model is:
          y(t) = Σ_k sqrt(p_k) * exp(jφ_k) * x(t - τ_k)
        where τ_k is the delay, p_k is the power, and φ_k is a random phase.
        
        Output is power-normalized to preserve the input signal's average power,
        ensuring that multipath doesn't change the overall signal level.
        """
        n = len(signal)
        output = np.zeros(n, dtype=np.complex128)
        powers_lin = db_to_linear(powers_db)
        for delay_us, power_lin in zip(delays_us, powers_lin):
            delay_samples = int(delay_us * 1e-6 * FS)
            delay_samples = max(0, min(delay_samples, n - 1))
            tap_phase = np.exp(1j * rng.uniform(0, 2 * np.pi))
            gain = math.sqrt(power_lin) * tap_phase
            output[delay_samples:] += gain * signal[:n - delay_samples]
        if np.max(np.abs(output)) > 0:
            output = output / (np.std(output) + 1e-12) * (np.std(signal) + 1e-12)
        return output

    def apply_rician_fading(self, signal, k_db, doppler_hz, rng):
        """Rician fading: h = sqrt(K/(K+1)) * LOS + sqrt(1/(K+1)) * NLOS.
        
        The Rician distribution models a channel with a dominant LOS component
        and scattered NLOS components:
          - LOS: deterministic (or slowly varying with Doppler phase drift)
          - NLOS: Rayleigh-distributed (complex Gaussian, filtered for coherence)
        
        K-factor (K_dB in linear): ratio of LOS to NLOS power.
          K → ∞: pure LOS (no fading, like free space)
          K → 0: pure NLOS (Rayleigh fading, no LOS)
        
        The Doppler filter (moving average) approximates the Jakes spectrum
        by introducing temporal correlation proportional to 1/(10*f_d).
        """
        n = len(signal)
        k_linear = db_to_linear(k_db)
        los = np.ones(n, dtype=np.complex128)
        if doppler_hz > 0:
            t = np.arange(n, dtype=np.float64) / FS
            phase_drift = 2 * np.pi * doppler_hz * 0.3 * t
            los = np.exp(1j * phase_drift)
        nlos_re = rng.randn(n)
        nlos_im = rng.randn(n)
        if doppler_hz > 1.0:
            filter_len = min(n, max(1, int(FS / (10 * doppler_hz))))
            kernel = np.ones(filter_len) / filter_len
            nlos_re = np.convolve(nlos_re, kernel, mode="same")[:n]
            nlos_im = np.convolve(nlos_im, kernel, mode="same")[:n]
        nlos = (nlos_re + 1j * nlos_im) / math.sqrt(2)
        k_factor = k_linear / (k_linear + 1)
        nlos_factor = 1.0 / math.sqrt(k_linear + 1)
        h = math.sqrt(k_factor) * los + nlos_factor * nlos
        return signal * h

    def apply_doppler_spread(self, signal, doppler_hz, rng):
        """Doppler spread as time-varying phase rotation."""
        if doppler_hz < 0.1:
            return signal
        n = len(signal)
        freq_var = doppler_hz * 0.1 * np.cumsum(rng.randn(n) * 0.01)
        freq_var = np.convolve(freq_var, np.ones(50) / 50, mode="same")
        phase = 2 * np.pi * np.cumsum(freq_var) / FS
        return signal * np.exp(1j * phase)

    def apply_pathloss(self, signal, pathloss_db):
        gain = 10.0 ** (-pathloss_db / 20.0)
        return signal * gain

    def apply_awgn(self, signal, snr_db, rng):
        sig_power = np.mean(np.abs(signal) ** 2)
        if sig_power < 1e-15:
            return signal
        noise_power = sig_power / db_to_linear(snr_db)
        noise = (rng.randn(len(signal)) + 1j * rng.randn(len(signal))) * math.sqrt(noise_power / 2)
        return signal + noise

    def apply_channel(self, signal, params: ChannelParams, rng):
        """Apply full channel model."""
        sig = np.asarray(signal, dtype=np.complex128)
        sig = self.apply_multipath(sig, params.multipath_delays_us, params.multipath_powers_db, rng)
        sig = self.apply_rician_fading(sig, params.rician_k_db, params.doppler_hz, rng)
        sig = self.apply_doppler_spread(sig, params.doppler_hz, rng)
        sig = self.apply_pathloss(sig, params.pathloss_db)
        sig = self.apply_awgn(sig, params.snr_db, rng)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        return np.asarray(sig, dtype=np.complex128)


def generate_channel_params(scenario: str, rng: np.random.RandomState) -> ChannelParams:
    """Generate scenario-specific channel parameters."""
    sp = SCENARIO_PARAMS[scenario]
    k_db = rng.uniform(*sp["rician_k_range"])
    shadowing = rng.normal(0, sp["shadowing_sigma"])
    # Distance between 50m and 500m
    distance_m = rng.uniform(50, 500)
    d_km = distance_m / 1000.0
    pathloss = 128.1 + 37.6 * math.log10(max(d_km, 1e-6)) + shadowing
    # Doppler from scenario speed range
    speed_ms = rng.uniform(sp["speed_range_kmh"][0], sp["speed_range_kmh"][1]) / 3.6
    doppler = speed_ms * V2X_FC / C_LIGHT
    # SNR
    snr_db = rng.uniform(5, 25)
    # Multipath PDP
    n_taps = sp["n_multipath"]
    spread = sp["pdp_spread"]
    delays = np.sort(rng.exponential(spread * 0.3, size=n_taps))
    delays[0] = 0.0
    powers = np.array([0.0] + list(-rng.exponential(3.0, size=n_taps - 1)))

    return ChannelParams(
        pathloss_db=pathloss,
        doppler_hz=doppler,
        rician_k_db=k_db,
        shadowing_db=shadowing,
        snr_db=snr_db,
        multipath_delays_us=delays,
        multipath_powers_db=powers,
    )


# =============================================================================
# SIGNAL GENERATION (exact copy from autoattack_eval.py generate_v2x_dataset)
# =============================================================================

def generate_base_signal(cls, rng: np.random.RandomState):
    """Generate baseband complex signal for a given class (same as autoattack_eval.py)."""
    t = np.arange(SAMPLE_LENGTH, dtype=np.float64) / FS

    if cls == 0:  # LTE - OFDM, 64 subcarriers, central 70% BW
        n_sub = 64
        symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
        sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
        bw = 0.35 * FS
        for k in range(n_sub):
            freq_k = (k - n_sub // 2) * (2 * bw / n_sub)
            sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
    elif cls == 1:  # WiFi - OFDM, 52 subcarriers, upper band
        n_sub = 52
        symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
        sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
        bw = 0.25 * FS
        offset = 0.15 * FS
        for k in range(n_sub):
            freq_k = (k - n_sub // 2) * (2 * bw / n_sub) + offset
            sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
    elif cls == 2:  # V2X-PC5 - SC-FDMA, 12 subcarriers, lower band
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

    return sig


def compute_spectrograms(sig: np.ndarray):
    """Compute log-magnitude and instantaneous frequency spectrograms (same as autoattack_eval.py)."""
    n_frames = (SAMPLE_LENGTH - NFFT) // HOP + 1
    n_freq = NFFT // 2 + 1
    window = np.hanning(NFFT).astype(np.float64)

    mag_raw = np.zeros((n_freq, n_frames), dtype=np.float64)
    if_raw = np.zeros((n_freq, n_frames), dtype=np.float64)

    for f in range(n_frames):
        start = f * HOP
        chunk = sig[start: start + NFFT]
        windowed = (chunk * window).astype(np.complex128)
        spectrum = np.fft.fft(windowed, n=NFFT)
        pos = spectrum[:n_freq]
        mag_raw[:, f] = np.log10(np.abs(pos) + 1e-10)
        if_raw[:, f] = np.angle(pos)

    # Instantaneous frequency via unwrapped phase differences
    phases_unwrapped = np.unwrap(if_raw, axis=1)
    inst_freq = np.diff(phases_unwrapped, axis=1)
    if_raw[:, 1:] = inst_freq
    if_raw[:, 0] = 0.0

    return mag_raw, if_raw


def generate_scenario_dataset(seed, scenario):
    """Generate full dataset for a specific mobility scenario with ChannelApplicator."""
    rng = np.random.RandomState(seed)
    ch = ChannelApplicator()
    n_per_class = NUM_SAMPLES // NUM_CLASSES

    n_frames = (SAMPLE_LENGTH - NFFT) // HOP + 1
    n_freq = NFFT // 2 + 1

    mag_raw = np.zeros((NUM_SAMPLES, n_freq, n_frames), dtype=np.float64)
    if_raw = np.zeros((NUM_SAMPLES, n_freq, n_frames), dtype=np.float64)
    y_all = np.zeros(NUM_SAMPLES, dtype=np.int64)

    idx = 0
    for cls in range(NUM_CLASSES):
        for i in range(n_per_class):
            # Generate base signal
            sig = generate_base_signal(cls, rng)

            # Apply mobility-aware channel (replaces simple Rayleigh + AWGN)
            cp = generate_channel_params(scenario, rng)
            sample_rng = np.random.RandomState(seed + idx + 1)
            sig = ch.apply_channel(sig, cp, sample_rng)

            # Compute spectrograms
            m, ip = compute_spectrograms(sig)
            mag_raw[idx] = m
            if_raw[idx] = ip
            y_all[idx] = cls
            idx += 1
            if idx % 1000 == 0:
                print(f"    Generated {idx}/{NUM_SAMPLES} samples...")

    # Train / Test split (80/20) - same as autoattack_eval.py
    indices = np.arange(NUM_SAMPLES)
    rng.shuffle(indices)
    n_train = int(0.8 * NUM_SAMPLES)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Z-score normalise (training set only)
    mag_train = mag_raw[train_idx]
    mag_mean = mag_train.mean()
    mag_std = mag_train.std() + 1e-8
    mag_norm = np.clip((mag_raw - mag_mean) / mag_std, -5, 5)

    if_train = if_raw[train_idx]
    if_mean = if_train.mean()
    if_std = if_train.std() + 1e-8
    if_norm = np.clip((if_raw - if_mean) / if_std, -5, 5)

    X_mag = torch.from_numpy(mag_norm).unsqueeze(1).float()
    X_if = torch.from_numpy(if_norm).unsqueeze(1).float()
    y = torch.from_numpy(y_all).long()

    return X_mag, X_if, y, train_idx, val_idx


# =============================================================================
# MODEL (exact copy from autoattack_eval.py - 86,052 params)
# =============================================================================

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        c4 = out_ch // 4
        c12 = out_ch // 2
        self.branch1 = nn.Conv2d(in_ch, c4, kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_ch, c4, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, c12, kernel_size=5, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat([
            self.branch1(x), self.branch3(x), self.branch5(x)
        ], dim=1)))


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
        x = self.pool(x)
        return x.flatten(1)


class DualStreamModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mag_stream = SingleStream()
        self.if_stream = SingleStream()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        f_mag = self.mag_stream(mag)
        f_if = self.if_stream(ift)
        x = torch.cat([f_mag, f_if], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# TF-CUTMIX + TRAINING (exact copy from autoattack_eval.py)
# =============================================================================

def tf_cutmix(mag, ift, y, alpha=CUTMIX_ALPHA):
    if np.random.rand() > CUTMIX_PROB:
        return mag, ift, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = mag.size()
    cut_ratio = np.sqrt(1.0 - lam)
    rw = int(W * cut_ratio)
    rh = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    rand_idx = torch.randperm(B, device=mag.device)
    mag_out = mag.clone()
    if_out = ift.clone()
    mag_out[:, :, y1:y2, x1:x2] = mag[rand_idx, :, y1:y2, x1:x2]
    if_out[:, :, y1:y2, x1:x2] = ift[rand_idx, :, y1:y2, x1:x2]
    actual_lam = 1.0 - float((x2 - x1) * (y2 - y1)) / float(H * W)
    return mag_out, if_out, y, y[rand_idx], actual_lam


def get_lr_lambda(epoch, warmup=WARMUP_EPOCHS, total=EPOCHS):
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = (epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_standard(model, train_loader, val_loader, epochs=EPOCHS, patience=PATIENCE):
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    best_acc, best_state = 0.0, None
    wait = 0
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
        if epoch % 5 == 0 or val_acc > best_acc:
            print(f"      epoch {epoch:3d}  val_acc={val_acc*100:.1f}%  lr={scheduler.get_last_lr()[0]:.6f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"      Early stop epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return best_acc


# =============================================================================
# ATTACKS (FGSM + PGD from autoattack_eval.py)
# =============================================================================

def fgsm_attack(model, mag, ift, y, eps=EPSILON):
    was_training = model.training
    model.eval()
    mag_a = mag.clone().detach().requires_grad_(True)
    if_a = ift.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(mag_a, if_a), y)
    loss.backward()
    mag_p = (mag + eps * mag_a.grad.sign()).clamp(-5, 5).detach()
    if_p = (ift + eps * if_a.grad.sign()).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def pgd_attack(model, mag, ift, y, eps=EPSILON, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    was_training = model.training
    model.eval()
    mag_delta = torch.zeros_like(mag, requires_grad=True)
    if_delta = torch.zeros_like(ift, requires_grad=True)
    for _ in range(steps):
        mag_adv = (mag + mag_delta).clamp(-5, 5)
        if_adv = (ift + if_delta).clamp(-5, 5)
        logits = model(mag_adv, if_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        mag_delta.data = mag_delta + step_size * mag_delta.grad.sign()
        if_delta.data = if_delta + step_size * if_delta.grad.sign()
        mag_delta.data = torch.clamp(mag_delta, -eps, eps)
        if_delta.data = torch.clamp(if_delta, -eps, eps)
        mag_delta.grad.zero_()
        if_delta.grad.zero_()
    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p = (ift + if_delta).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def attack_success_rate(model, mag, ift, y):
    model.eval()
    with torch.no_grad():
        preds = model(mag, ift).argmax(1)
    return (preds != y).float().mean().item()


# =============================================================================
# SINGLE SCENARIO EXPERIMENT
# =============================================================================

def run_scenario_experiment(scenario, seed, eps=EPSILON, epochs=EPOCHS):
    """Train and evaluate model on a single scenario."""
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario.upper()}  |  SEED: {seed}")
    print(f"{'='*60}")
    set_seed(seed)
    t0 = time.time()

    # --- Dataset with mobility-aware channel ---
    print(f"\n[1/3] Generating {scenario} dataset with ChannelApplicator...")
    t1 = time.time()
    X_mag, X_if, y, train_idx, val_idx = generate_scenario_dataset(seed, scenario)
    print(f"  Data generation: {time.time()-t1:.1f}s")

    mag_train, if_train, y_train = X_mag[train_idx], X_if[train_idx], y[train_idx]
    mag_val, if_val, y_val = X_mag[val_idx], X_if[val_idx], y[val_idx]

    train_ds = TensorDataset(mag_train, if_train, y_train)
    val_ds = TensorDataset(mag_val, if_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    mag_dev = mag_val.to(DEVICE)
    if_dev = if_val.to(DEVICE)
    y_dev = y_val.to(DEVICE)
    print(f"  Train: {len(train_idx)} samples, Test: {len(val_idx)} samples")

    # --- Train ---
    patience = min(PATIENCE, max(5, epochs // 3))
    print(f"\n[2/3] Training (max {epochs} epochs, patience={patience})...")
    model = DualStreamModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    best_acc = train_standard(model, train_loader, val_loader, epochs=epochs, patience=patience)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")

    # --- Attacks ---
    print(f"\n[3/3] FGSM + PGD attacks at eps={eps}...")
    m_fgsm, i_fgsm = fgsm_attack(model, mag_dev, if_dev, y_dev, eps=eps)
    fgsm_asr = attack_success_rate(model, m_fgsm, i_fgsm, y_dev)
    print(f"  FGSM ASR: {fgsm_asr*100:.2f}%")

    m_pgd, i_pgd = pgd_attack(model, mag_dev, if_dev, y_dev, eps=eps)
    pgd_asr = attack_success_rate(model, m_pgd, i_pgd, y_dev)
    print(f"  PGD ASR:  {pgd_asr*100:.2f}%")

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    return {
        "scenario": scenario,
        "seed": seed,
        "test_acc": best_acc,
        "fgsm_asr": fgsm_asr,
        "pgd_asr": pgd_asr,
        "elapsed_s": elapsed,
        "n_params": n_params,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Mobility Scenario-Stratified Adversarial Robustness Experiment"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--scenarios", nargs="+", type=str,
                        default=["highway", "urban", "rural"])
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 1 seed only")
    parser.add_argument("--eps", type=float, default=EPSILON)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick-epochs", type=int, default=20,
                        help="Epochs for quick mode (default: 20)")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epochs = args.quick_epochs

    print("=" * 60)
    print("  MOBILITY SCENARIO-STRATIFIED ROBUSTNESS EXPERIMENT")
    print("  Dual-Stream Inception-Time CNN")
    print("=" * 60)
    print(f"  Device:    {DEVICE}")
    print(f"  Scenarios: {args.scenarios}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Epsilon:   {args.eps}")
    print(f"  Epochs:    {args.epochs}")

    all_results = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            result = run_scenario_experiment(scenario, seed, eps=args.eps, epochs=args.epochs)
            all_results.append(result)

    # --- Aggregate by scenario ---
    agg = {
        "config": {
            "model": "DualStreamModel (exact autoattack_eval.py)",
            "channel_model": "ChannelApplicator (Rician + Doppler + multipath + pathloss)",
            "scenarios": args.scenarios,
            "seeds": args.seeds,
            "epsilon": args.eps,
            "num_samples": NUM_SAMPLES,
            "signal_params": {
                "sample_length": SAMPLE_LENGTH,
                "fs_hz": FS,
                "nfft": NFFT,
                "hop": HOP,
            },
            "scenario_channel_params": {
                s: {
                    "rician_k_range": list(SCENARIO_PARAMS[s]["rician_k_range"]),
                    "shadowing_sigma_db": SCENARIO_PARAMS[s]["shadowing_sigma"],
                    "pdp_spread_us": SCENARIO_PARAMS[s]["pdp_spread"],
                    "speed_range_kmh": list(SCENARIO_PARAMS[s]["speed_range_kmh"]),
                }
                for s in args.scenarios
            },
            "training": {
                "tf_cutmix_prob": CUTMIX_PROB,
                "cutmix_alpha": CUTMIX_ALPHA,
                "label_smoothing": LABEL_SMOOTH,
                "gauss_noise_prob": GAUSS_PROB,
                "gauss_noise_std": GAUSS_STD,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "warmup_epochs": WARMUP_EPOCHS,
                "patience": PATIENCE,
            },
            "attacks": {
                "fgsm": {"eps": args.eps},
                "pgd": {"eps": args.eps, "steps": PGD_STEPS, "step_size": PGD_STEP_SIZE},
            },
        },
        "per_scenario_results": {},
    }

    for scenario in args.scenarios:
        scenario_results = [r for r in all_results if r["scenario"] == scenario]
        if not scenario_results:
            continue
        mean_acc = np.mean([r["test_acc"] for r in scenario_results])
        std_acc = np.std([r["test_acc"] for r in scenario_results]) if len(scenario_results) > 1 else 0.0
        mean_fgsm = np.mean([r["fgsm_asr"] for r in scenario_results])
        std_fgsm = np.std([r["fgsm_asr"] for r in scenario_results]) if len(scenario_results) > 1 else 0.0
        mean_pgd = np.mean([r["pgd_asr"] for r in scenario_results])
        std_pgd = np.std([r["pgd_asr"] for r in scenario_results]) if len(scenario_results) > 1 else 0.0

        agg["per_scenario_results"][scenario] = {
            "test_acc": f"{mean_acc*100:.2f} +/- {std_acc*100:.2f}%",
            "fgsm_asr": f"{mean_fgsm*100:.2f} +/- {std_fgsm*100:.2f}%",
            "pgd_asr": f"{mean_pgd*100:.2f} +/- {std_pgd*100:.2f}%",
            "per_seed": to_serializable(scenario_results),
        }

    agg["results_table"] = []
    for scenario in args.scenarios:
        sr = agg["per_scenario_results"][scenario]
        agg["results_table"].append({
            "scenario": scenario,
            "test_acc_pct": float(np.mean([r["test_acc"] for r in
                [rr for rr in all_results if rr["scenario"] == scenario]])) * 100,
            "fgsm_asr_pct": float(np.mean([r["fgsm_asr"] for r in
                [rr for rr in all_results if rr["scenario"] == scenario]])) * 100,
            "pgd_asr_pct": float(np.mean([r["pgd_asr"] for r in
                [rr for rr in all_results if rr["scenario"] == scenario]])) * 100,
        })

    # --- Print Summary ---
    print("\n" + "=" * 80)
    print("  MOBILITY SCENARIO-STRATIFIED ROBUSTNESS SUMMARY")
    print("=" * 80)
    print(f"  {'Scenario':>10s}  {'Test Acc':>12s}  {'FGSM ASR':>12s}  {'PGD ASR':>12s}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*12}  {'-'*12}")
    for row in agg["results_table"]:
        print(f"  {row['scenario']:>10s}  {row['test_acc_pct']:>11.2f}%  "
              f"{row['fgsm_asr_pct']:>11.2f}%  {row['pgd_asr_pct']:>11.2f}%")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "mobility_scenario_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(to_serializable(agg), f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    return agg


if __name__ == "__main__":
    main()
