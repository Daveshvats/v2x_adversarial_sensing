"""
defenses.py
Adversarial Training and Input Denoising defenses.
"""

import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

random.seed(42); np.random.seed(42); torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fgsm_batch(model, X, y, eps):
    """Generate FGSM adversarial examples on a batch (utility for adversarial training).
    
    Same algorithm as adversarial_attacks.fgsm_attack but designed for use
    inside the training loop where the model is already on the device.
    """
    Xa = X.clone().detach().to(DEVICE); Xa.requires_grad_(True)
    nn.CrossEntropyLoss()(model(Xa), y.to(DEVICE)).backward()
    return torch.clamp(Xa.detach() + eps * Xa.grad.sign(), X.min().item(), X.max().item())


class EarlyStopping:
    """Stops training when validation loss stops improving."""
    def __init__(self, patience=5):
        self.patience, self.counter, self.best, self.stop = patience, 0, None, False
    def step(self, val_loss):
        if self.best is None or val_loss < self.best - 1e-4:
            self.best = val_loss; self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.stop = True


def adv_train(train_loader, val_loader, num_classes=5, num_epochs=5,
              lr=0.001, fgsm_eps=0.03, mix_ratio=0.3, ckpt="adv_model.pt"):
    """Adversarial Training defense: train on a mix of clean and adversarial examples.
    
    Adversarial training is the most effective defense against gradient-based attacks
    (Madry et al., 2018). The key idea is to include adversarial examples in the
    training data so the model learns to be robust against them.
    
    Implementation:
      For each batch:
        1. Split the batch: first (1-ratio) samples are clean,
           last (ratio) samples are replaced with FGSM adversarial examples
        2. Train the model on the mixed batch
    
    The mix_ratio parameter controls the fraction of adversarial examples:
      - Higher ratio → stronger robustness but potentially lower clean accuracy
      - Lower ratio → weaker robustness but better clean accuracy
      - This trade-off is known as the "accuracy vs. robustness" dilemma
    
    Hyperparameters:
      - num_epochs=5: fewer than standard training because adversarial training
        converges faster (the harder adversarial examples accelerate learning)
      - fgsm_eps=0.03: the perturbation budget used for generating training
        adversarial examples (should match the expected attack budget)
      - lr=0.001: standard Adam learning rate with cosine annealing
    """
    from model import V2XSpectrumCNN
    model = V2XSpectrumCNN(num_classes=num_classes).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
    crit = nn.CrossEntropyLoss()
    early = EarlyStopping(patience=5)
    best_a, best_s = 0.0, None

    for ep in range(1, num_epochs + 1):
        model.train()
        for Xb, yb, _ in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            # Number of adversarial samples in this batch
            na = int(len(Xb) * mix_ratio)
            if na > 0:
                # Clean portion: first (batch_size - na) samples
                Xc, yc = Xb[:-na], yb[:-na]
                # Generate FGSM adversarial examples for the last na samples
                Xa = fgsm_batch(model, Xb[-na:], yb[-na:], eps=fgsm_eps)
                # Concatenate clean + adversarial into mixed batch
                Xb = torch.cat([Xc, Xa]); yb = torch.cat([yc, yb[-na:]])
            opt.zero_grad(); crit(model(Xb), yb).backward(); opt.step()
        sch.step()

        model.eval(); vo, vn = 0, 0
        with torch.no_grad():
            for Xv, yv, _ in val_loader:
                p = model(Xv.to(DEVICE)).argmax(1)
                vo += (p == yv.to(DEVICE)).sum().item(); vn += yv.size(0)
        va = vo / max(vn, 1)
        if va > best_a: best_a = va; best_s = copy.deepcopy(model.state_dict())
        early.step(1.0 - va)
        if early.stop: break

    if best_s: model.load_state_dict(best_s)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, ckpt))
    print(f"    AdvTrain r={mix_ratio} best val: {best_a:.4f}")
    return model


def run_adv_train_sweep(train_loader, val_loader, test_loader, snr_levels,
                        mix_ratios=[0.1, 0.5], fgsm_eps=0.03, num_classes=5,
                        num_epochs=5):
    """Sweep adversarial training mix ratios and evaluate robustness.
    
    Trains a separate model for each mix_ratio, then evaluates both
    clean accuracy and FGSM robustness at the primary epsilon.
    
    This sweep demonstrates the robustness-accuracy trade-off:
    higher mix ratios improve robustness but may degrade clean accuracy.
    """
    n = len(snr_levels)
    res = {"mix_ratios": mix_ratios, "snr_levels": snr_levels, "clean_acc": {}, "fgsm_robust_acc": {}}

    for r in mix_ratios:
        model = adv_train(train_loader, val_loader, num_classes, num_epochs,
                          fgsm_eps=fgsm_eps, mix_ratio=r, ckpt=f"adv_r{r}.pt")
        model.eval()
        # Per-SNR counters for clean and FGSM-attacked accuracy
        cc, ct = [0]*n, [0]*n
        fc, ft = [0]*n, [0]*n
        for X, y, snr in test_loader:
            # Clean accuracy
            with torch.no_grad():
                pc = model(X.to(DEVICE)).argmax(1).cpu()
            # FGSM robustness: generate attacks on the adversarially-trained model
            # (the model must see its own gradients to craft the attack)
            Xa = fgsm_batch(model, X, y, eps=fgsm_eps)
            with torch.no_grad():
                pf = model(Xa).argmax(1).cpu()
            for p_, pf_, l, s in zip(pc, pf, y, snr):
                cc[s.item()] += (p_ == l).item(); ct[s.item()] += 1
                fc[s.item()] += (pf_ == l).item(); ft[s.item()] += 1
        res["clean_acc"][str(r)] = [cc[i]/max(ct[i],1) for i in range(n)]
        res["fgsm_robust_acc"][str(r)] = [fc[i]/max(ft[i],1) for i in range(n)]

    return res


class Denoiser(nn.Module):
    """Lightweight convolutional autoencoder for input denoising defense.
    
    Architecture:
      Encoder: Conv(1,32)->ReLU->MaxPool -> Conv(32,64)->ReLU->MaxPool
      Decoder: ConvTranspose2d(64,32)->ReLU -> ConvTranspose2d(32,1)->Sigmoid
    
    Design choices:
      - Encoder halves spatial dims twice: 64->32->16
      - Decoder uses transposed convolutions (stride=2) to upsample back: 16->32->64
      - Sigmoid output activation ensures output is in [0, 1] range
      - No bottleneck FC layers — preserves spatial structure important for spectrograms
      - MSE loss for pixel-level reconstruction quality
    
    The denoiser is trained to remove additive Gaussian noise from spectrograms.
    The hypothesis is that adversarial perturbations (which are often small and
    high-frequency) will also be partially removed by the denoiser, reducing
    their effectiveness.
    
    Limitation: this defense can be circumvented by attacks that account for
    the denoiser (i.e., "expectation over transformation" attacks).
    """
    def __init__(self):
        super().__init__()
        # Encoder: extract features while reducing spatial resolution
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        # Decoder: reconstruct the clean spectrogram from encoded features
        # ConvTranspose2d with stride=2 doubles the spatial dimensions
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.dec(self.enc(x))


def train_denoiser(train_loader, val_loader, epochs=5, lr=0.001):
    """Train the denoiser autoencoder on artificially noised spectrograms.
    
    Training procedure:
      For each batch:
        1. Add Gaussian noise (σ=0.1) to the clean spectrograms
        2. Train the denoiser to reconstruct the original clean spectrograms
        3. Loss = MSE(denoised, clean)
    
    The noise level σ=0.1 is chosen to be similar in magnitude to typical
    adversarial perturbations (ε=0.03-0.1), so the denoiser learns to remove
    perturbations of that scale.
    
    The clamp(-3, 3) ensures the noisy input stays within a reasonable range
    for the Sigmoid output of the denoiser.
    """
    model = Denoiser().to(DEVICE)
    crit = nn.MSELoss(); opt = optim.Adam(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    early = EarlyStopping(patience=5)
    best_l, best_s = float("inf"), None

    for ep in range(1, epochs+1):
        model.train()
        for Xb, _, _ in train_loader:
            Xb = Xb.to(DEVICE)
            # Add synthetic Gaussian noise to simulate adversarial perturbations
            # std=0.1 chosen to match typical adversarial perturbation magnitudes
            Xn = torch.clamp(Xb + torch.randn_like(Xb) * 0.1, -3, 3)
            opt.zero_grad(); crit(model(Xn), Xb).backward(); opt.step()
        sch.step()

        model.eval(); vl = 0
        with torch.no_grad():
            for Xb, _, _ in val_loader:
                Xb = Xb.to(DEVICE)
                Xn = torch.clamp(Xb + torch.randn_like(Xb) * 0.1, -3, 3)
                vl += crit(model(Xn), Xb).item()
        vl /= max(len(val_loader), 1)
        if vl < best_l: best_l = vl; best_s = copy.deepcopy(model.state_dict())
        early.step(vl)
        if early.stop: break

    if best_s: model.load_state_dict(best_s)
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "denoiser.pt"))
    print(f"    Denoiser best val loss: {best_l:.6f}")
    return model


def run_denoiser_exp(model, train_loader, val_loader, test_loader,
                     snr_levels, fgsm_eps=0.03):
    """Evaluate the denoiser defense: denoise then classify.
    
    Two evaluation modes:
      1. Clean accuracy after denoising: model(denoiser(x)) vs. y
         This measures whether denoising hurts clean performance.
      2. FGSM robustness with denoising: model(denoiser(fgsm(denoiser(x)))) vs. y
         This measures whether denoising mitigates adversarial attacks.
    
    The FGSM attack is generated ON the denoised input (not the raw input),
    which represents a realistic threat model where the adversary may not
    know about the denoiser defense.
    """
    den = train_denoiser(train_loader, val_loader, epochs=5)
    n = len(snr_levels)
    model.eval(); den.eval()
    cc, fc, st = [0]*n, [0]*n, [0]*n

    for X, y, snr in tqdm(test_loader, desc="  Denoiser eval", leave=False):
        with torch.no_grad():
            # Clean accuracy: denoise first, then classify
            Xd = den(X.to(DEVICE))
            pc = model(Xd).argmax(1).cpu()
        # FGSM attack on denoised input, then denoise the adversarial example
        # This tests whether the denoiser can "clean up" adversarial perturbations
        Xa = fgsm_batch(model, Xd.detach(), y, eps=fgsm_eps)
        with torch.no_grad():
            Xad = den(Xa)
            pf = model(Xad).argmax(1).cpu()
        for p_, f_, l, s in zip(pc, pf, y, snr):
            cc[s.item()] += (p_ == l).item()
            fc[s.item()] += (f_ == l).item()
            st[s.item()] += 1

    ca = [cc[i]/max(st[i],1) for i in range(n)]
    fa = [fc[i]/max(st[i],1) for i in range(n)]
    return {"snr_levels": snr_levels, "clean_acc": ca, "fgsm_robust_acc": fa,
            "avg_clean": float(np.mean(ca)), "avg_fgsm_robust": float(np.mean(fa))}
