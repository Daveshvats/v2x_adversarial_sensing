#!/usr/bin/env python3
"""
Adversarial Attack Transferability Study
=========================================
Tests whether adversarial examples crafted on one model variant transfer
to other variants. Demonstrates architectural diversity as implicit defense.

Model variants: Mag-Only, IF-Only, Dual-Stream (all exact v3.py architecture).
Attacks: FGSM, PGD-20 at multiple epsilon values.
Seeds: 3-seed protocol.

Transfer matrix (9 pairs × 2 attacks × N epsilons):
  Mag-Only → Mag-Only, IF-Only, Dual-Stream
  IF-Only  → Mag-Only, IF-Only, Dual-Stream
  Dual     → Mag-Only, IF-Only, Dual-Stream
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
# CONFIGURATION (matches v3.py)
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

GAUSS_STD      = 0.02
GAUSS_PROB     = 0.3
CUTMIX_ALPHA   = 1.0
CUTMIX_PROB    = 0.3

OUTPUT_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transfer_data")
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
            if cls == 0:
                n_sub = 64
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.35 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub)
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            elif cls == 1:
                n_sub = 52
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.25 * FS
                offset = 0.15 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub) + offset
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            elif cls == 2:
                n_sub = 12
                symbols = rng.randn(n_sub) + 1j * rng.randn(n_sub)
                sig = np.zeros(SAMPLE_LENGTH, dtype=np.complex128)
                bw = 0.08 * FS
                offset = -0.10 * FS
                for k in range(n_sub):
                    freq_k = (k - n_sub // 2) * (2 * bw / n_sub) + offset
                    sig += symbols[k] * np.exp(2j * np.pi * freq_k * t)
            else:
                sig = (rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)).astype(np.complex128)

            h = (rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)) / np.sqrt(2)
            snr_db = rng.uniform(5, 25)
            snr_lin = 10.0 ** (snr_db / 10.0)
            noise_pwr = np.var(sig).item() / max(snr_lin, 1e-12)
            noise = np.sqrt(max(noise_pwr, 0)) * (
                rng.randn(SAMPLE_LENGTH) + 1j * rng.randn(SAMPLE_LENGTH)) / np.sqrt(2)
            sig = (h * sig + noise).astype(np.complex128)

            for f in range(n_frames):
                start = f * HOP
                chunk = sig[start: start + NFFT]
                windowed = (chunk * window).astype(np.complex128)
                spectrum = np.fft.fft(windowed, n=NFFT)
                pos = spectrum[:n_freq]
                mag_raw[idx, :, f] = np.log10(np.abs(pos) + 1e-10)
                if_raw[idx, :, f] = np.angle(pos)

            phases = if_raw[idx, :, :]
            phases_unwrapped = np.unwrap(phases, axis=1)
            inst_freq = np.diff(phases_unwrapped, axis=1)
            if_raw[idx, :, 1:] = inst_freq
            if_raw[idx, :, 0] = 0.0
            y_all[idx] = cls
            idx += 1

    indices = np.arange(NUM_SAMPLES)
    rng.shuffle(indices)
    n_train = int(0.8 * NUM_SAMPLES)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

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
# 2. MODELS (exact v3.py components + single-stream wrappers)
# =============================================================================

class InceptionBlock(nn.Module):
    """3-branch Inception: 1×1, 3×3, 5×5 — exact v3.py."""
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
    """86,052 params — exact v3.py."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.mag_stream = SingleStream()
        self.if_stream  = SingleStream()
        self.fc1     = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        return self.fc2(self.dropout(F.relu(self.fc1(
            torch.cat([self.mag_stream(mag), self.if_stream(ift)], dim=1)))))


class MagOnlyModel(nn.Module):
    """Single-stream magnitude-only model — 43,188 params.
    
    Uses only the log-magnitude spectrogram; ignores the IF stream entirely.
    This model tests whether phase information (instantaneous frequency)
    contributes significantly to adversarial robustness.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stream = SingleStream()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        return self.fc(self.stream(mag))  # ignore ift


class IFOnlyModel(nn.Module):
    """Single-stream instantaneous-frequency-only model — 43,188 params.
    
    Uses only the instantaneous-frequency spectrogram; ignores magnitude.
    Tests whether power spectral information is the primary vulnerability.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.stream = SingleStream()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, mag, ift):
        return self.fc(self.stream(ift))  # ignore mag


# =============================================================================
# 3. TRAINING (exact v3.py recipe)
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


def train_model(model, train_loader, val_loader, epochs=EPOCHS):
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
                break
    model.load_state_dict(best_state)
    return best_acc


# =============================================================================
# 4. ATTACKS (stream-aware)
# =============================================================================

def attack_stream(model, mag, ift, y, attack="fgsm", eps=EPSILON, stream="both"):
    """Craft adversarial perturbation on SOURCE model for specified stream(s).
    
    stream: "mag", "if", or "both"
    
    Key insight: when crafting adversarial examples for transferability,
    we only compute gradients w.r.t. the streams that the SOURCE model uses.
    This ensures the perturbation is optimized for the source model's architecture.
    
    For example: if attacking a Mag-Only model, only the magnitude stream
    gets perturbed (the IF stream is zero). If this transfer to the IF-Only
    model, the IF-Only model won't see any perturbation at all — demonstrating
    that cross-stream attacks are ineffective.
    """
    was_training = model.training
    model.eval()

    if attack == "fgsm":
        # Only enable gradients for the streams the source model uses
        mag_delta = torch.zeros_like(mag, requires_grad=(stream in ("mag", "both")))
        if_delta  = torch.zeros_like(ift,  requires_grad=(stream in ("if", "both")))
        logits = model(
            (mag + mag_delta).clamp(-5, 5) if stream in ("mag", "both") else mag,
            (ift  + if_delta).clamp(-5, 5) if stream in ("if", "both") else ift,
        )
        loss = F.cross_entropy(logits, y)
        model.zero_grad()
        loss.backward()
        if stream in ("mag", "both"):
            mag_delta = (eps * mag_delta.grad.sign()).detach()
        if stream in ("if", "both"):
            if_delta  = (eps * if_delta.grad.sign()).detach()

    elif attack == "pgd":
        mag_delta = torch.zeros_like(mag).detach()
        if_delta  = torch.zeros_like(ift).detach()
        step = eps / 4
        for _ in range(PGD_STEPS):
            # Ensure fresh leaf tensors each iteration
            mag_delta = mag_delta.detach().clone()
            if_delta  = if_delta.detach().clone()
            mag_delta.requires_grad_(stream in ("mag", "both"))
            if_delta.requires_grad_(stream in ("if", "both"))
            m_in = (mag + mag_delta).clamp(-5, 5) if stream in ("mag", "both") else mag
            i_in = (ift  + if_delta).clamp(-5, 5) if stream in ("if", "both") else ift
            logits = model(m_in, i_in)
            loss = F.cross_entropy(logits, y)
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                if stream in ("mag", "both") and mag_delta.grad is not None:
                    mag_delta = mag_delta + step * mag_delta.grad.sign()
                    mag_delta = mag_delta.clamp(-eps, eps)
                if stream in ("if", "both") and if_delta.grad is not None:
                    if_delta  = if_delta  + step * if_delta.grad.sign()
                    if_delta  = if_delta.clamp(-eps, eps)
    else:
        raise ValueError(f"Unknown attack: {attack}")

    mag_adv = (mag + mag_delta).clamp(-5, 5).detach() if stream in ("mag", "both") else mag
    if_adv  = (ift  + if_delta).clamp(-5, 5).detach() if stream in ("if", "both") else ift

    if was_training:
        model.train()
    return mag_adv, if_adv


def eval_asr(model, mag, ift, y):
    model.eval()
    with torch.no_grad():
        return (model(mag, ift).argmax(1) != y).float().mean().item()


# =============================================================================
# 5. TRANSFER MATRIX EVALUATION
# =============================================================================

def get_active_stream(model_name):
    """Which stream does the source model use for gradients?"""
    if model_name == "Mag-Only":
        return "mag"
    elif model_name == "IF-Only":
        return "if"
    else:
        return "both"


def run_transfer_eval(models, mag_dev, if_dev, y_dev, eps_list):
    """Evaluate full 9×9 transfer matrix."""
    model_names = ["Mag-Only", "IF-Only", "Dual-Stream"]
    attacks = ["fgsm", "pgd"]
    results = {}

    for src_name in model_names:
        src_model = models[src_name]
        src_stream = get_active_stream(src_name)
        for tgt_name in model_names:
            tgt_model = models[tgt_name]
            for attack in attacks:
                for eps in eps_list:
                    # Craft perturbation on source
                    mag_adv, if_adv = attack_stream(
                        src_model, mag_dev, if_dev, y_dev,
                        attack=attack, eps=eps, stream=src_stream
                    )
                    # Test on target
                    asr = eval_asr(tgt_model, mag_adv, if_adv, y_dev)
                    key = f"{src_name}->{tgt_name}_{attack}_eps{eps}"
                    results[key] = asr

    return results


# =============================================================================
# 6. SINGLE-SEED EXPERIMENT
# =============================================================================

def run_single_seed(seed, eps_list):
    print(f"\n{'='*60}")
    print(f"  TRANSFERABILITY STUDY — SEED {seed}")
    print(f"{'='*60}")
    set_seed(seed)
    t0 = time.time()

    # --- Dataset ---
    print("\n[1/3] Generating dataset...")
    X_mag, X_if, y, train_idx, val_idx = generate_v2x_dataset(seed=seed)
    mag_train, if_train, y_train = X_mag[train_idx], X_if[train_idx], y[train_idx]
    mag_val, if_val, y_val = X_mag[val_idx], X_if[val_idx], y[val_idx]
    train_ds = TensorDataset(mag_train, if_train, y_train)
    val_ds   = TensorDataset(mag_val, if_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    mag_dev, if_dev, y_dev = mag_val.to(DEVICE), if_val.to(DEVICE), y_val.to(DEVICE)

    # --- Train 3 model variants ---
    print("\n[2/3] Training model variants...")
    models = {}
    model_classes = {
        "Mag-Only": MagOnlyModel,
        "IF-Only": IFOnlyModel,
        "Dual-Stream": DualStreamModel,
    }
    for name, cls in model_classes.items():
        model = cls().to(DEVICE)
        n_p = sum(p.numel() for p in model.parameters())
        acc = train_model(model, train_loader, val_loader)
        models[name] = model
        print(f"  {name}: {n_p:,} params, acc={acc*100:.2f}%")

    # --- Transfer evaluation ---
    print("\n[3/3] Running transfer matrix...")
    transfer_results = run_transfer_eval(models, mag_dev, if_dev, y_dev, eps_list)

    elapsed = time.time() - t0
    print(f"  Seed {seed} done in {elapsed:.1f}s")
    return {"seed": seed, "transfer": transfer_results, "elapsed_s": elapsed}


# =============================================================================
# 7. VISUALIZATION
# =============================================================================

def generate_heatmaps(all_results, eps_list, out_dir):
    """Generate transfer matrix heatmap for each attack × epsilon."""
    model_names = ["Mag-Only", "IF-Only", "Dual-Stream"]
    attacks = ["fgsm", "pgd"]
    seeds = [r["seed"] for r in all_results]

    for attack in attacks:
        for eps in eps_list:
            # Build 3×3 matrix (source × target)
            matrix = np.zeros((3, 3))
            for i, src in enumerate(model_names):
                for j, tgt in enumerate(model_names):
                    key = f"{src}->{tgt}_{attack}_eps{eps}"
                    vals = [r["transfer"][key] for r in all_results]
                    matrix[i, j] = np.mean(vals) * 100

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
            ax.set_xticks(range(3))
            ax.set_yticks(range(3))
            ax.set_xticklabels(model_names, fontsize=11)
            ax.set_yticklabels(model_names, fontsize=11)
            ax.set_xlabel("Target Model", fontsize=12)
            ax.set_ylabel("Source Model", fontsize=12)
            ax.set_title(f"Transfer ASR (%) — {attack.upper()} ε={eps}", fontsize=13)

            # Annotate cells
            for i in range(3):
                for j in range(3):
                    # Compute std
                    key = f"{model_names[i]}->{model_names[j]}_{attack}_eps{eps}"
                    std = np.std([r["transfer"][key] for r in all_results]) * 100
                    val = matrix[i, j]
                    color = "white" if val > 50 else "black"
                    ax.text(j, i, f"{val:.1f}±{std:.1f}", ha="center", va="center",
                            fontsize=11, color=color, fontweight="bold")
                    # Diagonal = baseline
                    if i == j:
                        rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False,
                                             edgecolor="blue", linewidth=3)
                        ax.add_patch(rect)

            plt.colorbar(im, ax=ax, label="ASR (%)")
            plt.tight_layout()
            path = os.path.join(out_dir, f"transfer_{attack}_eps{eps}.png")
            fig.savefig(path, dpi=200)
            plt.close(fig)
            print(f"  Saved: {path}")


def generate_transfer_ratio_figure(all_results, eps_list, out_dir):
    """Bar chart: transfer ratio = transferred ASR / baseline ASR."""
    model_names = ["Mag-Only", "IF-Only", "Dual-Stream"]
    attacks = ["fgsm", "pgd"]
    primary_eps = EPSILON

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, attack in enumerate(attacks):
        ax = axes[ax_idx]
        # For each source, compute transfer ratio to each target
        x = np.arange(3)
        width = 0.25
        for j, tgt in enumerate(model_names):
            ratios = []
            for i, src in enumerate(model_names):
                key = f"{src}->{tgt}_{attack}_eps{primary_eps}"
                transferred = np.mean([r["transfer"][key] for r in all_results])
                baseline_key = f"{src}->{src}_{attack}_eps{primary_eps}"
                baseline = np.mean([r["transfer"][baseline_key] for r in all_results])
                ratio = transferred / max(baseline, 1e-6) * 100
                ratios.append(ratio)
            ax.bar(x + j * width, ratios, width, label=f"→ {tgt}")
        ax.set_xticks(x + width)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Transfer Ratio (%)", fontsize=11)
        ax.set_title(f"{attack.upper()} Transfer Ratio (ε={primary_eps})", fontsize=12)
        ax.legend(loc="best")
        ax.axhline(y=100, color="gray", ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "transfer_ratio.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# 8. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Adversarial Transferability Study")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epsilons", nargs="+", type=float,
                        default=[0.01, 0.03, 0.05])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [42]
        args.epsilons = [0.03]

    print("=" * 60)
    print("  ADVERSARIAL TRANSFERABILITY STUDY")
    print("  V2X Dual-Stream Inception-Time CNN")
    print("=" * 60)
    print(f"  Device:   {DEVICE}")
    print(f"  Seeds:    {args.seeds}")
    print(f"  Epsilons: {args.epsilons}")
    print(f"  Output:   {OUTPUT_DIR}")

    all_results = []
    for seed in args.seeds:
        result = run_single_seed(seed, args.epsilons)
        all_results.append(result)
        path = os.path.join(OUTPUT_DIR, f"transfer_seed{seed}.json")
        with open(path, "w") as f:
            json.dump(to_serializable(result), f, indent=2)
        print(f"  Saved: {path}")

    # --- Aggregate ---
    model_names = ["Mag-Only", "IF-Only", "Dual-Stream"]
    agg = {
        "config": {"seeds": args.seeds, "epsilons": args.epsilons},
        "transfer_matrix": {},
    }
    for src in model_names:
        for tgt in model_names:
            for attack in ["fgsm", "pgd"]:
                for eps in args.epsilons:
                    key = f"{src}->{tgt}_{attack}_eps{eps}"
                    vals = [r["transfer"][key] * 100 for r in all_results]
                    agg["transfer_matrix"][key] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}%"

    agg_path = os.path.join(OUTPUT_DIR, "transfer_aggregated.json")
    with open(agg_path, "w") as f:
        json.dump(to_serializable(agg), f, indent=2)

    # --- Print Transfer Matrix ---
    print("\n" + "=" * 80)
    print("  TRANSFER MATRIX (ASR % — mean ± std)")
    print("=" * 80)
    primary_eps = EPSILON
    for attack in ["fgsm", "pgd"]:
        print(f"\n  {attack.upper()} at ε={primary_eps}")
        print(f"  {'Source':>12s}  {'→ Mag-Only':>14s}  {'→ IF-Only':>14s}  {'→ Dual':>14s}")
        print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}")
        for src in model_names:
            row = []
            for tgt in model_names:
                key = f"{src}->{tgt}_{attack}_eps{primary_eps}"
                row.append(agg["transfer_matrix"][key])
            print(f"  {src:>12s}  {row[0]:>14s}  {row[1]:>14s}  {row[2]:>14s}")

    # --- Figures ---
    print("\n  Generating figures...")
    generate_heatmaps(all_results, args.epsilons, OUTPUT_DIR)
    generate_transfer_ratio_figure(all_results, args.epsilons, OUTPUT_DIR)

    # --- Key Findings ---
    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"\n  KEY FINDINGS:")
    # Check within-architecture vs cross-architecture
    for attack in ["fgsm", "pgd"]:
        # Baseline (diagonal)
        diag_keys = [f"{m}->{m}_{attack}_eps{primary_eps}" for m in model_names]
        diag_mean = np.mean([np.mean([r["transfer"][k] for r in all_results]) * 100 for k in diag_keys])
        # Cross-architecture (off-diagonal)
        cross_vals = []
        for src in model_names:
            for tgt in model_names:
                if src != tgt:
                    key = f"{src}->{tgt}_{attack}_eps{primary_eps}"
                    cross_vals.append(np.mean([r["transfer"][key] for r in all_results]) * 100)
        cross_mean = np.mean(cross_vals)
        print(f"    {attack.upper()}: Baseline (diag) = {diag_mean:.1f}%, Cross-arch = {cross_mean:.1f}%, "
              f"Transfer ratio = {cross_mean/max(diag_mean,1e-6)*100:.1f}%")


if __name__ == "__main__":
    main()
