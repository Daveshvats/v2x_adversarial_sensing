#!/usr/bin/env python3
"""
Official AutoAttack Evaluation for V2X Adversarial Spectrum Sensing
====================================================================
Uses the **official** `autoattack` library (fra31/auto-attack) to re-evaluate
the Dual-Stream Inception-Time CNN.

The official library expects a single-input model f(x) -> logits where x is in
[0, 1].  Since our data lives in [-5, 5] after z-score normalisation we:

  1. Concatenate (log-mag, inst-freq) along the channel dim -> (B, 2, 65, 15).
  2. Linearly map to [0, 1]: x_norm = (x + 5) / 10.
  3. Scale epsilon:          eps_norm = eps / 10.

Inside DualStreamWrapper the inverse mapping is applied before splitting and
feeding the underlying DualStreamModel.

Model architecture, data generation, and training recipe are EXACTLY copied
from v3.py (via autoattack_eval.py) to ensure result comparability.
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

# Data lives in [-5, 5] after z-score clipping; AutoAttack expects [0, 1]
DATA_MIN, DATA_MAX = -5.0, 5.0
DATA_RANGE       = DATA_MAX - DATA_MIN           # 10.0
# Scaling factor to convert epsilon from [-5,5] space to [0,1] space
# This is necessary because AutoAttack's internal algorithms assume [0,1] input
AA_EPS_SCALE     = 1.0 / DATA_RANGE              # eps_aa = eps / 10

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "official_autoattack_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")

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


def to_aa_range(x):
    """Map data from [-5, 5] to [0, 1] for AutoAttack compatibility."""
    return (x - DATA_MIN) / DATA_RANGE


def from_aa_range(x):
    """Map data from [0, 1] back to [-5, 5]."""
    return x * DATA_RANGE + DATA_MIN


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
    """3-branch Inception: 1x1, 3x3, 5x5 — matches v3.py exactly."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        assert out_ch % 4 == 0
        c4  = out_ch // 4
        c12 = out_ch // 2
        self.branch1 = nn.Conv2d(in_ch, c4,  kernel_size=1, bias=False)
        self.branch3 = nn.Conv2d(in_ch, c4,  kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_ch, c12, kernel_size=5, padding=2, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(torch.cat([
            self.branch1(x), self.branch3(x), self.branch5(x)
        ], dim=1)))


class SingleStream(nn.Module):
    """Conv->MaxPool->Inception x2->AdaptiveAvgPool — output (B, 64)."""
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
    """Dual-stream (log-mag + inst-freq) — 86,052 params."""
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
# 2b. WRAPPER — makes DualStreamModel compatible with AutoAttack
# =============================================================================

class DualStreamWrapper(nn.Module):
    """
    Wraps DualStreamModel so the official autoattack library can use it.

    * Accepts  x of shape (B, 2, H, W) in [0, 1] (AutoAttack convention).
    * Internally maps to [-5, 5], splits channels, calls underlying model.
    * Returns logits.
    
    Design: AutoAttack operates on a single tensor, so we concatenate
    the two spectrogram streams along the channel dimension:
      x_aa[:, 0:1] -> log-magnitude spectrogram
      x_aa[:, 1:2] -> instantaneous-frequency spectrogram
    The wrapper handles the inverse scaling and channel splitting internally.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x is in [0, 1] — map to [-5, 5]
        x_real = from_aa_range(x)
        mag = x_real[:, 0:1]   # (B, 1, H, W)
        ift = x_real[:, 1:2]   # (B, 1, H, W)
        return self.model(mag, ift)


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
    """Cosine annealing with linear warmup.
    
    During warmup (epoch < warmup): linearly increase LR from ~0 to LR.
    After warmup: LR follows cosine decay: LR * 0.5 * (1 + cos(π * progress))
    This schedule combines the benefits of warmup (stable initial training)
    with cosine decay (smooth convergence to near-zero LR).
    """
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
# 4. BASELINE ATTACKS — FGSM + PGD (same formulation as v3.py, but via wrapper)
# =============================================================================

def fgsm_attack(model, mag, ift, y, eps=EPSILON):
    """FGSM on the raw (mag, ift) tensors — operates in [-5, 5] space."""
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


def pgd_attack(model, mag, ift, y, eps=EPSILON, steps=PGD_STEPS, step_size=None):
    """PGD-20 on the raw (mag, ift) tensors — operates in [-5, 5] space."""
    if step_size is None:
        step_size = eps / 4
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


def attack_success_rate(model, mag, ift, y):
    model.eval()
    with torch.no_grad():
        preds = model(mag, ift).argmax(1)
    return (preds != y).float().mean().item()


# =============================================================================
# 5. OFFICIAL AUTOATTACK EVALUATION
# =============================================================================

def run_official_autoattack(wrapper, x_aa, y_aa, eps=EPSILON, bs=BATCH_SIZE):
    """
    Run the official autoattack library (version='standard').

    Parameters
    ----------
    wrapper : DualStreamWrapper
    x_aa    : Tensor (B, 2, H, W) in [0, 1]
    y_aa    : Tensor (B,)   long
    eps     : float  — epsilon in the ORIGINAL [-5, 5] data space
    bs      : int

    Returns
    -------
    dict with per-attack ASRs and worst-case ASR
    """
    from autoattack import AutoAttack

    eps_aa = eps * AA_EPS_SCALE  # scale to [0, 1] space

    # ---- run individual attacks so we can record each component's ASR ----
    print(f"    Running official AutoAttack at eps={eps:.4f}  "
          f"(eps_aa={eps_aa:.6f}) ...")

    aa = AutoAttack(wrapper, norm='Linf', eps=eps_aa, version='custom',
                    attacks_to_run=['apgd-ce', 'apgd-dlr', 'fab', 'square'],
                    device=str(DEVICE), verbose=True)

    indiv = aa.run_standard_evaluation_individual(x_aa, y_aa, bs=bs)

    # compute per-component ASR
    results = {}
    for attack_name, x_adv in indiv.items():
        wrapper.eval()
        with torch.no_grad():
            preds = wrapper(x_adv.to(DEVICE)).argmax(1)
            asr = (preds != y_aa.to(DEVICE)).float().mean().item()
        results[attack_name] = asr
        print(f"      {attack_name:>12s}: {asr*100:.2f}%")

    # worst-case across all attacks (per-sample OR)
    wrapper.eval()
    with torch.no_grad():
        all_preds = torch.stack([
            wrapper(x_adv.to(DEVICE)).argmax(1)
            for x_adv in indiv.values()
        ], dim=0)  # (n_attacks, B)
        worst_asr = (all_preds != y_aa.unsqueeze(0).to(DEVICE)).any(dim=0).float().mean().item()

    results['aa_worst'] = worst_asr
    print(f"      {'AA worst':>12s}: {worst_asr*100:.2f}%")

    return results


# =============================================================================
# 6. SINGLE-SEED EXPERIMENT
# =============================================================================

def run_single_seed(seed, eps_list):
    print(f"\n{'='*70}")
    print(f"  OFFICIAL AUTOATTACK EVALUATION — SEED {seed}")
    print(f"{'='*70}")
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

    # --- Wrap model for AutoAttack ---
    wrapper = DualStreamWrapper(model).to(DEVICE)
    wrapper.eval()

    # Prepare concatenated test data in [0, 1] for AutoAttack
    x_aa = to_aa_range(torch.cat([mag_val, if_val], dim=1)).to(DEVICE)  # (B, 2, H, W) in [0,1]
    y_aa = y_val.to(DEVICE)

    # --- Baseline attacks (FGSM + PGD) on raw tensors ---
    print("\n[3/4] Baseline attacks (FGSM + PGD)...")
    fgsm_results = {}
    pgd_results = {}
    for eps in eps_list:
        m_a, i_a = fgsm_attack(model, mag_dev, if_dev, y_dev, eps=eps)
        fgsm_results[eps] = attack_success_rate(model, m_a, i_a, y_dev)
        m_a, i_a = pgd_attack(model, mag_dev, if_dev, y_dev, eps=eps)
        pgd_results[eps] = attack_success_rate(model, m_a, i_a, y_dev)
        print(f"  eps={eps:.3f}: FGSM={fgsm_results[eps]*100:.2f}%  "
              f"PGD={pgd_results[eps]*100:.2f}%")

    # --- Official AutoAttack ---
    print("\n[4/4] Official AutoAttack evaluation...")
    aa_results = {}
    for eps in eps_list:
        print(f"\n  --- eps={eps} ---")
        aa_results[eps] = run_official_autoattack(wrapper, x_aa, y_aa,
                                                   eps=eps, bs=BATCH_SIZE)

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
# 7. AGGREGATION & COMPARISON
# =============================================================================

def aggregate_results(all_results, eps_list):
    """Build the aggregated dict matching the existing JSON schema."""
    agg = {
        "config": {
            "library": "official autoattack (fra31/auto-attack)",
            "model": "DualStreamModel (exact v3.py)",
            "seeds": [r["seed"] for r in all_results],
            "epsilons": eps_list,
            "attacks": ["FGSM", "PGD-20", "APGD-CE", "APGD-DLR",
                        "FAB", "Square", "AA-worst"],
            "eps_scale": f"1/{DATA_RANGE:.0f}  (data [-5,5] -> [0,1])",
        },
        "test_acc": float(np.mean([r["test_acc"] for r in all_results])),
        "test_acc_std": float(np.std([r["test_acc"] for r in all_results])),
        "per_epsilon": {},
    }

    for eps in eps_list:
        key = str(eps)
        agg["per_epsilon"][key] = {
            "fgsm_asr":     _fmt(all_results, "fgsm_asr", eps),
            "pgd_asr":      _fmt(all_results, "pgd_asr", eps),
            "apgd_ce_asr":  _fmt_aa(all_results, "apgd-ce", eps),
            "apgd_dlr_asr": _fmt_aa(all_results, "apgd-dlr", eps),
            "fab_asr":      _fmt_aa(all_results, "fab", eps),
            "square_asr":   _fmt_aa(all_results, "square", eps),
            "aa_worst_asr": _fmt_aa(all_results, "aa_worst", eps),
        }

    return agg


def _fmt(all_results, key, eps):
    vals = [r[key][eps] for r in all_results]
    return f"{np.mean(vals)*100:.2f} +/- {np.std(vals)*100:.2f}%"


def _fmt_aa(all_results, aa_key, eps):
    vals = [r["autoattack"][eps][aa_key] for r in all_results]
    return f"{np.mean(vals)*100:.2f} +/- {np.std(vals)*100:.2f}%"


def print_summary(agg, eps_list):
    print("\n" + "=" * 100)
    print("  OFFICIAL AUTOATTACK EVALUATION SUMMARY")
    print("=" * 100)
    print(f"  Library:  {agg['config']['library']}")
    print(f"  Model:    {agg['config']['model']}")
    print(f"  Seeds:    {agg['config']['seeds']}")
    print(f"  Test Acc: {agg['test_acc']*100:.2f}% +/- {agg['test_acc_std']*100:.2f}%")
    print()
    hdr = (f"  {'eps':>8s}  {'FGSM':>16s}  {'PGD-20':>16s}  "
           f"{'APGD-CE':>16s}  {'APGD-DLR':>16s}  "
           f"{'FAB':>16s}  {'Square':>16s}  {'AA-worst':>16s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for eps in eps_list:
        d = agg["per_epsilon"][str(eps)]
        print(f"  {eps:>8.3f}  {d['fgsm_asr']:>16s}  {d['pgd_asr']:>16s}  "
              f"{d['apgd_ce_asr']:>16s}  {d['apgd_dlr_asr']:>16s}  "
              f"{d['fab_asr']:>16s}  {d['square_asr']:>16s}  "
              f"{d['aa_worst_asr']:>16s}")


def compare_with_old(agg, old_path):
    """Load the old autoattack_aggregated.json and print side-by-side."""
    if not os.path.isfile(old_path):
        print(f"\n  [Comparison] Old results not found at {old_path}")
        return

    with open(old_path) as f:
        old = json.load(f)

    print("\n" + "=" * 100)
    print("  SIDE-BY-SIDE COMPARISON: official AA vs custom AA (from autoattack_eval.py)")
    print("=" * 100)

    eps_list = agg["config"]["epsilons"]
    # use the intersection of epsilons present in both
    old_eps_keys = [k for k in old.keys() if _is_float(k)]
    common_eps = [e for e in eps_list if str(e) in old_eps_keys]
    if not common_eps:
        print("  No common epsilon values found for comparison.")
        return

    print(f"\n  {'eps':>8s}  |  {'Metric':>12s}  |  "
          f"{'Old (custom)':>20s}  |  {'New (official)':>20s}  |  {'Delta':>10s}")
    print(f"  {'-'*8}--+--{'-'*12}--+--{'-'*20}--+--{'-'*20}--+--{'-'*10}")

    for eps in common_eps:
        new_d = agg["per_epsilon"][str(eps)]
        old_d = old[str(eps)]

        metrics = [
            ("fgsm_asr",     "FGSM"),
            ("pgd_asr",      "PGD"),
            ("apgd_ce_asr",  "APGD-CE"),
            ("apgd_dlr_asr", "APGD-DLR"),
            ("fab_asr",      "FAB"),
            ("square_asr",   "Square"),
            ("aa_worst_asr", "AA-worst"),
        ]
        for mkey, mname in metrics:
            old_str = old_d.get(mkey, "N/A")
            new_str = new_d.get(mkey, "N/A")
            # parse means
            old_mean = _parse_mean(old_str)
            new_mean = _parse_mean(new_str)
            if old_mean is not None and new_mean is not None:
                delta = new_mean - old_mean
                delta_str = f"{delta:+.2f}pp"
            else:
                delta_str = "N/A"
            if mkey == "fgsm_asr":
                print(f"  {eps:>8.3f}  |  {mname:>12s}  |  {old_str:>20s}  |  "
                      f"{new_str:>20s}  |  {delta_str:>10s}")
            else:
                print(f"  {'':>8s}  |  {mname:>12s}  |  {old_str:>20s}  |  "
                      f"{new_str:>20s}  |  {delta_str:>10s}")
        print()


def _is_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _parse_mean(s):
    """Extract mean percentage from 'XX.XX +/- Y.YY%' string."""
    if not isinstance(s, str):
        return None
    try:
        return float(s.split("+/-")[0].strip().replace("%", ""))
    except (ValueError, IndexError):
        return None


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Official AutoAttack Evaluation for V2X Spectrum Sensing")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.08])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 seed, 1 epsilon (0.03)")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epsilons = [0.03]

    print("=" * 70)
    print("  V2X OFFICIAL AUTOATTACK EVALUATION")
    print("  Dual-Stream Inception-Time CNN")
    print("  Library: fra31/auto-attack")
    print("=" * 70)
    print(f"  Device:    {DEVICE}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Epsilons:  {args.epsilons}")
    print(f"  Output:    {OUTPUT_DIR}")

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(seed, args.epsilons)
        all_results.append(result)
        path = os.path.join(OUTPUT_DIR, f"official_autoattack_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(to_serializable(result), f, indent=2)
        print(f"  Saved: {path}")

    # --- Aggregate ---
    agg = aggregate_results(all_results, args.epsilons)
    agg_path = os.path.join(OUTPUT_DIR, "official_autoattack_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(to_serializable(agg), f, indent=2)
    print(f"\n  Aggregated results saved to: {agg_path}")

    # --- Print summary ---
    print_summary(agg, args.epsilons)

    # --- Comparison with old results ---
    old_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "autoattack_data", "autoattack_aggregated.json")
    compare_with_old(agg, old_path)

    # --- Hypothesis check ---
    primary_eps = EPSILON
    if primary_eps in args.epsilons:
        fgsm_mean = np.mean([r["fgsm_asr"][primary_eps] for r in all_results])
        aa_mean = np.mean([r["autoattack"][primary_eps]["aa_worst"]
                           for r in all_results])
        gap = aa_mean - fgsm_mean
        print(f"\n  KEY FINDING (eps={primary_eps}): "
              f"FGSM ASR = {fgsm_mean*100:.2f}%, "
              f"Official AA ASR = {aa_mean*100:.2f}%")
        if gap < 5:
            print(f"  CONFIRMED: Official AutoAttack converges to FGSM level "
                  f"(gap = {gap*100:.2f}pp)")
            print(f"  -> TF-CutMix + Gaussian smoothing smooths loss landscape")
        else:
            print(f"  CAUTION: Official AutoAttack outperforms FGSM by "
                  f"{gap*100:.2f}pp")
            print(f"  -> Smooth loss landscape claim may need refinement")

    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print("  Done.")


if __name__ == "__main__":
    main()
