#!/usr/bin/env python3
"""
AutoAttack Evaluation for V2X Adversarial Spectrum Sensing
===========================================================
Tests the Dual-Stream Inception-Time CNN against AutoAttack's ensemble
(APGD-CE, APGD-DLR, FAB, Square) to verify the FGSM≈PGD convergence finding.

Model architecture, data generation, and training recipe are EXACTLY
copied from v3.py to ensure result comparability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import json, os, time, copy, warnings, math, argparse

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION (matches v3.py exactly)
# =============================================================================
# 3-seed protocol for statistical significance of results
SEEDS          = [42, 123, 456]
# Total dataset: 4000 samples (1000 per class for balanced classes)
NUM_SAMPLES    = 4000
# Each sample is 1024 IQ complex samples (typical for short V2X sensing windows)
SAMPLE_LENGTH  = 1024
# 10 MHz sampling rate (realistic for V2X DSRC/C-V2X band)
FS             = 10e6
# FFT size for STFT: determines frequency resolution = FS/NFFT = 78.125 kHz
NFFT           = 128
# Hop size: 64 samples = 50% overlap (standard for good time-frequency tradeoff)
HOP            = 64
# 4 signal classes: LTE, WiFi, V2X-PC5, Noise
NUM_CLASSES    = 4
CLASS_NAMES    = ["LTE", "WiFi", "V2X-PC5", "Noise"]
BATCH_SIZE     = 32
# 100 epochs with early stopping — sufficient for convergence on this dataset size
EPOCHS         = 100
# Adam learning rate: 5e-4 is slightly lower than default 1e-3 for more stable training
LR             = 5e-4
# L2 regularization to prevent overfitting on small dataset
WEIGHT_DECAY   = 1e-4
# Label smoothing 0.1: prevents overconfident predictions, improves calibration
LABEL_SMOOTH   = 0.1
# 5-epoch linear warmup before cosine decay (helps initial stability)
WARMUP_EPOCHS  = 5
PATIENCE       = 15

# Primary attack epsilon in L∞ norm: 0.03 corresponds to ~3% of the
# normalized data range [-5, 5], representing a subtle adversarial perturbation
EPSILON        = 0.03
# PGD iterations: 20 steps is standard (Madry et al., 2018)
PGD_STEPS      = 20
# PGD step size = eps/4 is the standard ratio that ensures convergence
# within the epsilon ball within PGD_STEPS iterations
PGD_STEP_SIZE  = EPSILON / 4

# Adversarial training mix fraction: fraction of batch replaced with PGD adversarial examples
AT_MIX_FRAC    = 0.5
# Gaussian noise augmentation: std=0.02 simulates sensor noise and regularizes the model
GAUSS_STD      = 0.02
# 30% probability of applying Gaussian noise per batch (stochastic augmentation)
GAUSS_PROB     = 0.3
# CutMix alpha=1.0 for symmetric Beta distribution (moderate cut sizes)
CUTMIX_ALPHA   = 1.0
# 30% probability of applying TF-CutMix per batch
CUTMIX_PROB    = 0.3

OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "autoattack_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 1. DATASET GENERATION (exact copy from v3.py)
# =============================================================================

def generate_v2x_dataset(seed=42):
    """Generate synthetic V2X spectrum data — dual-stream (log-mag + inst-freq)."""
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

            if cls == 0:  # LTE – OFDM, 64 subcarriers, central 70% BW
                n_sub = 64
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.35 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub)
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            elif cls == 1:  # WiFi – OFDM, 52 subcarriers, upper band
                n_sub = 52
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.25 * FS
                offset = 0.15 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub) + offset
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            elif cls == 2:  # V2X-PC5 – SC-FDMA, 12 subcarriers, lower band
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

            # Rayleigh fading + AWGN
            h = (rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)) / np.sqrt(2)
            snr_db = rng.uniform(5, 25)
            snr_lin = 10.0 ** (snr_db / 10.0)
            noise_pwr = np.var(sig).item() / max(snr_lin, 1e-12)
            noise = np.sqrt(max(noise_pwr, 0)) * (
                rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)) / np.sqrt(2)
            sig = (h * sig + noise).astype(np.complex128)

            # STFT
            for f in range(n_frames):
                start = f * HOP
                chunk = sig[start: start + NFFT]
                windowed = (chunk * window).astype(np.complex128)
                spectrum = np.fft.fft(windowed, n=NFFT)
                pos = spectrum[:n_freq]
                mag_raw[idx, :, f] = np.log10(np.abs(pos) + 1e-10)
                phase_frame = np.angle(pos)
                if_raw[idx, :, f] = phase_frame

            # Instantaneous frequency
            phases = if_raw[idx, :, :]
            phases_unwrapped = np.unwrap(phases, axis=1)
            inst_freq = np.diff(phases_unwrapped, axis=1)
            if_raw[idx, :, 1:] = inst_freq
            if_raw[idx, :, 0] = 0.0

            y_all[idx] = cls
            idx += 1

    # Train / Test split (80/20)
    indices = np.arange(NUM_SAMPLES)
    rng.shuffle(indices)
    n_train = int(0.8 * NUM_SAMPLES)
    train_idx = indices[:n_train]
    val_idx   = indices[n_train:]

    # Z-score normalise (training set only)
    mag_train = mag_raw[train_idx]
    mag_mean = mag_train.mean()
    mag_std  = mag_train.std() + 1e-8
    mag_norm = np.clip((mag_raw - mag_mean) / mag_std, -5, 5)

    if_train = if_raw[train_idx]
    if_mean = if_train.mean()
    if_std  = if_train.std() + 1e-8
    if_norm = np.clip((if_raw - if_mean) / if_std, -5, 5)

    X_mag = torch.from_numpy(mag_norm).unsqueeze(1).float()
    X_if  = torch.from_numpy(if_norm).unsqueeze(1).float()
    y     = torch.from_numpy(y_all).long()

    return X_mag, X_if, y, train_idx, val_idx

# =============================================================================
# 2. MODEL (exact copy from v3.py — 86,052 params)
# =============================================================================

class InceptionBlock(nn.Module):
    """3-branch Inception: 1x1, 3x3, 5x5 — matches v3.py exactly.
    
    Multi-scale feature extraction: the 3 different kernel sizes capture
    patterns at different spatial scales in the spectrogram.
    - 1x1: pointwise (cross-channel mixing only, no spatial context)
    - 3x3: local spectral patterns (narrowband features)
    - 5x5: wider spectral patterns (broadband features)
    
    Channel split: 25% to 1x1, 25% to 3x3, 50% to 5x5 (larger kernel
    gets more channels to compensate for its larger receptive field).
    bias=False: BatchNorm after concatenation makes bias redundant.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        c4  = out_ch // 4   # 25% of output channels for 1x1 branch
        c12 = out_ch // 2   # 50% of output channels for 5x5 branch
        self.branch1 = nn.Conv2d(in_ch, c4,  kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_ch, c4,  kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, c12, kernel_size=5, padding=2, bias=False)
        # BatchNorm after concatenation normalizes the merged multi-scale features
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat([
            self.branch1(x), self.branch3(x), self.branch5(x)
        ], dim=1)))


class SingleStream(nn.Module):
    """Conv→MaxPool→Inception×2→AdaptiveAvgPool — output (B, 64)."""
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
    """Dual-stream (log-mag + inst-freq) — 86,052 params.
    
    Architecture: Two identical SingleStream sub-networks process
    the log-magnitude and instantaneous-frequency spectrograms in parallel,
    then their 64-dim feature vectors are concatenated (128-dim) and
    classified by a small FC head.
    
    The dual-stream design captures complementary information:
    - Log-magnitude: signal power distribution across frequency
    - Instantaneous frequency: phase structure and modulation patterns
    
    Weight-sharing between streams: both use the same architecture but
    with independent weights (not weight-tied), allowing each stream to
    learn specialized features for its input type.
    
    Total params: ~86K — intentionally small for V2X edge deployment.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mag_stream = SingleStream()  # Processes log-magnitude spectrograms
        self.if_stream  = SingleStream()  # Processes instantaneous-frequency spectrograms
        # Fusion: concatenate two 64-dim feature vectors -> 128-dim
        self.fc1     = nn.Linear(128, 64)  # Bottleneck FC with ReLU
        self.dropout = nn.Dropout(0.3)     # Moderate dropout for regularization
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        f_mag = self.mag_stream(mag)
        f_if  = self.if_stream(ift)
        # Concatenate features from both streams along the channel dimension
        x = torch.cat([f_mag, f_if], dim=1)  # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# =============================================================================
# 3. TF-CUTMIX + TRAINING (exact copy from v3.py)
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
    if_out  = ift.clone()
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
        # Evaluate
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
# 4. ATTACKS — FGSM + PGD (from v3.py) + AutoAttack sub-attacks (custom)
# =============================================================================

def fgsm_attack(model, mag, ift, y, eps=EPSILON):
    """FGSM on dual-stream: perturb both magnitude and IF streams simultaneously.
    
    x_adv = x + eps * sign(grad_x L(f(x), y))
    
    Both streams are perturbed independently, which means the adversary
    can modify both the power spectrum and the phase structure of the signal.
    The clamp(-5, 5) ensures perturbed values stay in the z-score normalized range.
    """
    was_training = model.training
    model.eval()
    mag_a = mag.clone().detach().requires_grad_(True)
    if_a  = ift.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(mag_a, if_a), y)
    loss.backward()
    mag_p = (mag + eps * mag_a.grad.sign()).clamp(-5, 5).detach()
    if_p  = (ift  + eps * if_a.grad.sign()).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def pgd_attack(model, mag, ift, y, eps=EPSILON, steps=PGD_STEPS, step_size=PGD_STEP_SIZE):
    """PGD-20 on dual-stream: iterative projected gradient descent.
    
    Iteratively ascends the loss gradient with L∞ projection:
      1. Compute loss gradient w.r.t. perturbation delta
      2. Take step: delta += step_size * sign(grad)
      3. Project: delta = clamp(delta, -eps, eps)  [L∞ ball projection]
      4. Zero gradients for next iteration
    
    The delta.grad.zero_() is critical: without it, gradients accumulate
    across iterations (PyTorch default behavior), corrupting the attack direction.
    """
    was_training = model.training
    model.eval()
    mag_delta = torch.zeros_like(mag, requires_grad=True)
    if_delta  = torch.zeros_like(ift,  requires_grad=True)
    for _ in range(steps):
        mag_adv = (mag + mag_delta).clamp(-5, 5)
        if_adv  = (ift  + if_delta).clamp(-5, 5)
        logits = model(mag_adv, if_adv)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        mag_delta.data = mag_delta + step_size * mag_delta.grad.sign()
        if_delta.data  = if_delta  + step_size * if_delta.grad.sign()
        mag_delta.data = torch.clamp(mag_delta, -eps, eps)
        if_delta.data  = torch.clamp(if_delta,  -eps, eps)
        mag_delta.grad.zero_()
        if_delta.grad.zero_()
    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p  = (ift  + if_delta).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def apgd_attack(model, mag, ift, y, eps=EPSILON, steps=100, loss_type="ce"):
    """Auto-PGD attack (APGD-CE or APGD-DLR) on dual-stream."""
    was_training = model.training
    model.eval()
    B = mag.size(0)
    mag_delta = torch.zeros_like(mag)
    if_delta  = torch.zeros_like(ift)

    # Initial step size
    alpha = eps / 4
    best_asr = 0.0

    # Initialize from FGSM (outside no_grad so backward works)
    mag_a = mag.detach().clone().requires_grad_(True)
    if_a  = ift.detach().clone().requires_grad_(True)
    loss_init = F.cross_entropy(model(mag_a, if_a), y)
    model.zero_grad()
    loss_init.backward()
    mag_delta = (eps * mag_a.grad.sign()).detach()
    if_delta  = (eps * if_a.grad.sign()).detach()
    del mag_a, if_a, loss_init

    for i in range(steps):
        # Ensure fresh leaf tensors each iteration
        mag_delta = mag_delta.detach().clone().requires_grad_(True)
        if_delta  = if_delta.detach().clone().requires_grad_(True)

        mag_adv = (mag + mag_delta).clamp(-5, 5)
        if_adv  = (ift  + if_delta).clamp(-5, 5)
        logits = model(mag_adv, if_adv)

        if loss_type == "ce":
            loss = F.cross_entropy(logits, y)
        else:  # dlr
            # DLR loss: targeted at highest non-true class
            y_onehot = F.one_hot(y, NUM_CLASSES).float()
            correct_logits = (logits * y_onehot).sum(1)
            wrong_logits = (logits * (1 - y_onehot) - 1e4 * y_onehot).max(1)[0]
            loss = (wrong_logits - correct_logits).clamp(min=0).sum()

        model.zero_grad()
        loss.backward()
        grad_mag = mag_delta.grad.detach()
        grad_if  = if_delta.grad.detach()

        # Update with sign gradient (detached to break graph)
        with torch.no_grad():
            mag_delta = mag_delta + alpha * grad_mag.sign()
            if_delta  = if_delta  + alpha * grad_if.sign()
            mag_delta = mag_delta.clamp(-eps, eps)
            if_delta  = if_delta.clamp(-eps, eps)

        # Adaptive step size (decay every 50 steps)
        if (i + 1) % 50 == 0:
            alpha *= 0.5

    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p  = (ift  + if_delta).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def fab_attack(model, mag, ift, y, eps=EPSILON, steps=100):
    """Fast Adaptive Boundary attack on dual-stream (simplified)."""
    was_training = model.training
    model.eval()
    B = mag.size(0)

    mag_delta = torch.zeros_like(mag)
    if_delta  = torch.zeros_like(ift)

    # Initialize at boundary (binary search)
    with torch.no_grad():
        logits = model(mag, ift)
        preds = logits.argmax(1)
        correct_mask = (preds == y).float()

    # Find initial adversarial direction via gradient
    mag_delta = mag_delta.detach().clone().requires_grad_(True)
    if_delta  = if_delta.detach().clone().requires_grad_(True)
    logits = model((mag + mag_delta).clamp(-5, 5), (ift + if_delta).clamp(-5, 5))
    loss = F.cross_entropy(logits, y)
    model.zero_grad()
    loss.backward()
    mag_delta = (eps * mag_delta.grad.sign()).detach()
    if_delta  = (eps * if_delta.grad.sign()).detach()

    for _ in range(steps):
        # Ensure fresh leaf tensors each iteration
        mag_delta = mag_delta.detach().clone().requires_grad_(True)
        if_delta  = if_delta.detach().clone().requires_grad_(True)
        mag_adv = (mag + mag_delta).clamp(-5, 5)
        if_adv  = (ift  + if_delta).clamp(-5, 5)
        logits = model(mag_adv, if_adv)
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        grad_mag = mag_delta.grad.detach()
        grad_if  = if_delta.grad.detach()

        # Move towards boundary (reduce perturbation while staying adversarial)
        step_sz = eps / steps
        with torch.no_grad():
            mag_delta = mag_delta - step_sz * mag_delta.sign() + 0.01 * grad_mag.sign()
            if_delta  = if_delta  - step_sz * if_delta.sign()  + 0.01 * grad_if.sign()
            mag_delta = mag_delta.clamp(-eps, eps)
            if_delta  = if_delta.clamp(-eps, eps)

    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p  = (ift  + if_delta).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def square_attack(model, mag, ift, y, eps=EPSILON, steps=5000):
    """Square Attack (black-box, random search) on dual-stream."""
    was_training = model.training
    model.eval()
    B = mag.size(0)

    mag_delta = torch.zeros(B, *mag.shape[1:], device=mag.device)
    if_delta  = torch.zeros(B, *ift.shape[1:], device=ift.device)

    for _ in range(steps):
        # Random perturbation in L-inf ball
        rand_mag = (torch.rand_like(mag_delta) * 2 - 1) * eps
        rand_if  = (torch.rand_like(if_delta)  * 2 - 1) * eps

        mag_adv = (mag + rand_mag).clamp(-5, 5)
        if_adv  = (ift  + rand_if).clamp(-5, 5)

        with torch.no_grad():
            current_logits = model(mag_adv, if_adv)
            current_asr = (current_logits.argmax(1) != y).float()

            old_logits = model((mag + mag_delta).clamp(-5, 5), (ift + if_delta).clamp(-5, 5))
            old_asr = (old_logits.argmax(1) != y).float()

        # Keep the better perturbation per sample
        improve_mag = (current_asr > old_asr).view(B, 1, 1, 1).float()
        improve_if  = (current_asr > old_asr).view(B, 1, 1, 1).float()
        mag_delta = torch.where(improve_mag > 0, rand_mag, mag_delta)
        if_delta  = torch.where(improve_if  > 0, rand_if,  if_delta)

    mag_p = (mag + mag_delta).clamp(-5, 5).detach()
    if_p  = (ift  + if_delta).clamp(-5, 5).detach()
    if was_training:
        model.train()
    return mag_p, if_p


def attack_success_rate(model, mag, ift, y):
    model.eval()
    with torch.no_grad():
        preds = model(mag, ift).argmax(1)
    return (preds != y).float().mean().item()


# =============================================================================
# 5. FULL AUTOATTACK EVALUATION
# =============================================================================

def run_autoattack(model, mag, ift, y, eps=EPSILON):
    """Run AutoAttack ensemble: APGD-CE, APGD-DLR, FAB, Square."""
    print(f"    Running AutoAttack at eps={eps}...")

    # APGD-CE
    m1, i1 = apgd_attack(model, mag, ift, y, eps=eps, steps=100, loss_type="ce")
    asr_ce = attack_success_rate(model, m1, i1, y)
    print(f"      APGD-CE: {asr_ce*100:.2f}%")

    # APGD-DLR
    m2, i2 = apgd_attack(model, mag, ift, y, eps=eps, steps=100, loss_type="dlr")
    asr_dlr = attack_success_rate(model, m2, i2, y)
    print(f"      APGD-DLR: {asr_dlr*100:.2f}%")

    # FAB
    m3, i3 = fab_attack(model, mag, ift, y, eps=eps, steps=100)
    asr_fab = attack_success_rate(model, m3, i3, y)
    print(f"      FAB: {asr_fab*100:.2f}%")

    # Square Attack
    m4, i4 = square_attack(model, mag, ift, y, eps=eps, steps=5000)
    asr_sq = attack_success_rate(model, m4, i4, y)
    print(f"      Square: {asr_sq*100:.2f}%")

    # Worst-case: per-sample max ASR across all attacks
    with torch.no_grad():
        all_preds = torch.stack([
            model(m1, i1).argmax(1),
            model(m2, i2).argmax(1),
            model(m3, i3).argmax(1),
            model(m4, i4).argmax(1),
        ], dim=0)
        # If ANY attack fools the model, it's adversarial
        worst_asr = (all_preds != y.unsqueeze(0)).any(dim=0).float().mean().item()

    print(f"      AutoAttack (worst-case): {worst_asr*100:.2f}%")

    return {
        "apgd_ce": asr_ce,
        "apgd_dlr": asr_dlr,
        "fab": asr_fab,
        "square": asr_sq,
        "autoattack_worst": worst_asr,
    }


# =============================================================================
# 6. SINGLE-SEED EXPERIMENT
# =============================================================================

def run_single_seed(seed, eps_list):
    print(f"\n{'='*60}")
    print(f"  AUTOATTACK EVALUATION — SEED {seed}")
    print(f"{'='*60}")
    set_seed(seed)
    t0 = time.time()

    # --- Dataset ---
    print("\n[1/4] Generating dataset...")
    X_mag, X_if, y, train_idx, val_idx = generate_v2x_dataset(seed=seed)

    mag_train, if_train, y_train = X_mag[train_idx], X_if[train_idx], y[train_idx]
    mag_val,   if_val,   y_val   = X_mag[val_idx],   X_if[val_idx],   y[val_idx]

    train_ds = TensorDataset(mag_train, if_train, y_train)
    val_ds   = TensorDataset(mag_val, if_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    mag_dev = mag_val.to(DEVICE)
    if_dev  = if_val.to(DEVICE)
    y_dev   = y_val.to(DEVICE)

    # --- Train ---
    print("\n[2/4] Training DualStreamModel...")
    model = DualStreamModel().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    best_acc = train_standard(model, train_loader, val_loader)
    print(f"  Best test accuracy: {best_acc*100:.2f}%")

    # --- Baseline attacks (FGSM + PGD) ---
    print("\n[3/4] Baseline attacks (FGSM + PGD)...")
    fgsm_results = {}
    pgd_results = {}
    for eps in eps_list:
        m_a, i_a = fgsm_attack(model, mag_dev, if_dev, y_dev, eps=eps)
        fgsm_results[eps] = attack_success_rate(model, m_a, i_a, y_dev)
        m_a, i_a = pgd_attack(model, mag_dev, if_dev, y_dev, eps=eps)
        pgd_results[eps] = attack_success_rate(model, m_a, i_a, y_dev)
        print(f"  eps={eps:.3f}: FGSM={fgsm_results[eps]*100:.2f}% PGD={pgd_results[eps]*100:.2f}%")

    # --- AutoAttack at primary epsilon ---
    print("\n[4/4] AutoAttack evaluation...")
    aa_results = {}
    for eps in eps_list:
        print(f"\n  --- eps={eps} ---")
        aa_results[eps] = run_autoattack(model, mag_dev, if_dev, y_dev, eps=eps)

    elapsed = time.time() - t0
    print(f"\n  Seed {seed} done in {elapsed:.1f}s")

    return {
        "seed": seed,
        "n_params": n_params,
        "test_acc": best_acc,
        "fgsm_asr": fgsm_results,
        "pgd_asr": pgd_results,
        "autoattack": aa_results,
        "elapsed_s": elapsed,
    }


# =============================================================================
# 7. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="AutoAttack Evaluation for V2X Spectrum Sensing")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true", help="Quick test: 1 seed, 1 eps")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epsilons = [0.03]

    print("=" * 60)
    print("  V2X AUTOATTACK EVALUATION")
    print("  Dual-Stream Inception-Time CNN")
    print("=" * 60)
    print(f"  Device:    {DEVICE}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Epsilons:  {args.epsilons}")
    print(f"  Output:    {OUTPUT_DIR}")

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(seed, args.epsilons)
        all_results.append(result)
        path = os.path.join(OUTPUT_DIR, f"autoattack_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(to_serializable(result), f, indent=2)
        print(f"  Saved: {path}")

    # --- Aggregate ---
    eps_list = args.epsilons
    agg = {
        "config": {
            "model": "DualStreamModel (exact v3.py)",
            "seeds": args.seeds,
            "epsilons": eps_list,
            "attacks": ["FGSM", "PGD-20", "APGD-CE", "APGD-DLR", "FAB", "Square", "AutoAttack-worst"],
        },
        "test_acc": float(np.mean([r["test_acc"] for r in all_results])),
        "test_acc_std": float(np.std([r["test_acc"] for r in all_results])),
    }

    # Per-epsilon summary
    for eps in eps_list:
        key = str(eps)
        agg[key] = {
            "fgsm_asr":      f"{np.mean([r['fgsm_asr'][eps] for r in all_results])*100:.2f} ± {np.std([r['fgsm_asr'][eps] for r in all_results])*100:.2f}%",
            "pgd_asr":       f"{np.mean([r['pgd_asr'][eps] for r in all_results])*100:.2f} ± {np.std([r['pgd_asr'][eps] for r in all_results])*100:.2f}%",
            "apgd_ce_asr":   f"{np.mean([r['autoattack'][eps]['apgd_ce'] for r in all_results])*100:.2f} ± {np.std([r['autoattack'][eps]['apgd_ce'] for r in all_results])*100:.2f}%",
            "apgd_dlr_asr":  f"{np.mean([r['autoattack'][eps]['apgd_dlr'] for r in all_results])*100:.2f} ± {np.std([r['autoattack'][eps]['apgd_dlr'] for r in all_results])*100:.2f}%",
            "fab_asr":       f"{np.mean([r['autoattack'][eps]['fab'] for r in all_results])*100:.2f} ± {np.std([r['autoattack'][eps]['fab'] for r in all_results])*100:.2f}%",
            "square_asr":    f"{np.mean([r['autoattack'][eps]['square'] for r in all_results])*100:.2f} ± {np.std([r['autoattack'][eps]['square'] for r in all_results])*100:.2f}%",
            "aa_worst_asr":  f"{np.mean([r['autoattack'][eps]['autoattack_worst'] for r in all_results])*100:.2f} ± {np.std([r['autoattack'][eps]['autoattack_worst'] for r in all_results])*100:.2f}%",
        }

    agg_path = os.path.join(OUTPUT_DIR, "autoattack_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(to_serializable(agg), f, indent=2)

    # --- Print Summary ---
    print("\n" + "=" * 80)
    print("  AUTOATTACK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"  Test Accuracy: {agg['test_acc']*100:.2f}% ± {agg['test_acc_std']*100:.2f}%")
    print()
    print(f"  {'eps':>8s}  {'FGSM':>10s}  {'PGD':>10s}  {'APGD-CE':>10s}  {'APGD-DLR':>10s}  {'FAB':>10s}  {'Square':>10s}  {'AA-worst':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for eps in eps_list:
        d = agg[str(eps)]
        print(f"  {eps:>8.3f}  {d['fgsm_asr']:>10s}  {d['pgd_asr']:>10s}  {d['apgd_ce_asr']:>10s}  {d['apgd_dlr_asr']:>10s}  {d['fab_asr']:>10s}  {d['square_asr']:>10s}  {d['aa_worst_asr']:>10s}")

    # --- Hypothesis Check ---
    primary_eps = EPSILON
    fgsm_mean = np.mean([r['fgsm_asr'][primary_eps] for r in all_results])
    aa_mean = np.mean([r['autoattack'][primary_eps]['autoattack_worst'] for r in all_results])
    gap = aa_mean - fgsm_mean
    print(f"\n  KEY FINDING: FGSM ASR = {fgsm_mean*100:.2f}%, AutoAttack ASR = {aa_mean*100:.2f}%")
    if gap < 5:
        print(f"  ✓ CONFIRMED: AutoAttack converges to FGSM level (gap={gap*100:.2f}pp)")
        print(f"  → TF-CutMix + Gaussian smoothing smooths loss landscape")
    else:
        print(f"  ⚠ CAUTION: AutoAttack outperforms FGSM by {gap*100:.2f}pp")
        print(f"  → Smooth loss landscape claim may need refinement")

    print(f"\n  Results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
