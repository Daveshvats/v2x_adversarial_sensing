"""
adversarial_attacks.py
FGSM, PGD, and C&W (Auto-PGD L2) attacks on V2X spectrum sensing model.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

random.seed(42); np.random.seed(42); torch.manual_seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "..", "checkpoints")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_test_data(batch_size=256):
    """Load the test split for attack evaluation."""
    d = np.load(os.path.join(DATA_DIR, "test.npz"))
    X = torch.from_numpy(d["X"].astype(np.float32))
    y = torch.from_numpy(d["y"]); snr = torch.from_numpy(d["snr"])
    return DataLoader(TensorDataset(X, y, snr), batch_size=batch_size, shuffle=False)


def fgsm_attack(model, X, y, eps):
    """Fast Gradient Sign Method (FGSM) — single-step attack.
    
    FGSM computes the gradient of the loss w.r.t. the input and takes
    a single step in the direction that maximizes the loss:
    
        x_adv = x + ε * sign(∇_x L(f(x), y))
    
    where:
      - ε controls the perturbation magnitude (L∞ norm bound)
      - sign() takes the sign of each gradient element
      - L is the cross-entropy loss
    
    FGSM is a fast (single-step) but often suboptimal attack; it provides
    a lower bound on the model's vulnerability. Despite its simplicity,
    it's surprisingly effective on poorly regularized models.
    
    The clamp to [X.min(), X.max()] ensures the adversarial example stays
    in the valid data range (important for z-scored spectrograms).
    """
    # Detach from graph, move to device, enable gradient computation on input
    Xa = X.clone().detach().to(DEVICE); Xa.requires_grad_(True)
    # Forward pass and compute classification loss
    loss = nn.CrossEntropyLoss()(model(Xa), y.to(DEVICE))
    # Backward pass to compute input gradients
    loss.backward()
    # Take one step in the gradient direction, scaled by epsilon
    Xa = Xa.detach() + eps * Xa.grad.sign()
    # Clamp to valid data range to maintain spectrogram realism
    return torch.clamp(Xa, X.min().item(), X.max().item()).detach()


def pgd_attack(model, X, y, eps, steps, step_size):
    """Projected Gradient Descent (PGD) — multi-step iterative attack.
    
    PGD is the strongest first-order attack. It iteratively:
      1. Computes the loss gradient w.r.t. the perturbed input
      2. Takes a small step (α = step_size) in the gradient direction
      3. Projects the perturbation back onto the L∞ ball of radius ε
      4. Clamps to the valid data range
    
    This is equivalent to:
        δ_{t+1} = Π_{B(x,ε)}[ δ_t + α * sign(∇_x L(f(x + δ_t), y)) ]
    
    where Π denotes projection onto the ε-ball centered at the clean input.
    
    Key design choices:
      - step_size = eps/4 by default (standard ratio from Madry et al., 2018)
      - The projection ensures the final perturbation satisfies ||x_adv - x||_∞ ≤ ε
      - Typically 10-20 steps are sufficient for convergence
    
    PGD is much stronger than FGSM but also much slower due to the
    iterative nature (requires `steps` forward+backward passes).
    """
    Xa = X.clone().detach().to(DEVICE); orig = Xa.clone()
    for _ in range(steps):
        Xa.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(Xa), y.to(DEVICE))
        loss.backward()
        # Ascend the loss gradient with step_size
        Xa = Xa.detach() + step_size * Xa.grad.sign()
        # Project perturbation onto L∞ ball: clip δ to [-eps, +eps]
        delta = torch.clamp(Xa - orig, -eps, eps)
        # Also clamp to valid data range (spectrogram values must be realistic)
        Xa = torch.clamp(orig + delta, orig.min().item(), orig.max().item())
    return Xa.detach()


def auto_pgd_l2(model, X, y, eps, steps=5):
    """Auto-PGD L2 (C&W-style) — L2-norm constrained optimization attack.
    
    Unlike FGSM/PGD which use the L∞ norm, this attack constrains the
    perturbation magnitude in the L2 (Euclidean) norm:
        ||x_adv - x||_2 ≤ ε
    
    The attack works by:
      1. Computing the L2-normalized gradient direction
      2. Stepping along that direction with a decaying step size
      3. Scaling the perturbation to satisfy the L2 constraint
    
    Key algorithm details:
      - Step size decays linearly: α = eps * 0.8 * ((1 - i/steps) + 0.1)
        This encourages exploration early and refinement later.
      - L2 projection: if ||δ||_2 > ε, scale δ to have norm exactly ε
        Implemented via min(1, eps/||δ||_2) scaling factor.
      - Best result tracking: keeps the adversarial example with highest
        loss across all iterations, not just the final one.
    
    The "C&W" name comes from Carlini & Wagner (2017) who pioneered
    optimization-based attacks that are much stronger than gradient-based
    methods like FGSM/PGD.
    """
    Xa = X.clone().detach().to(DEVICE); y = y.to(DEVICE)
    bs = X.shape[0]; best = Xa.clone(); best_loss = torch.full((bs,), -1e9, device=DEVICE)
    for i in range(steps):
        Xa.requires_grad_(True)
        # Per-sample loss (not mean-reduced) to track individual sample quality
        loss = nn.CrossEntropyLoss(reduction="none")(model(Xa), y)
        loss.sum().backward()
        g = Xa.grad.detach()
        # L2-normalize gradient: g_unit = g / ||g||_2
        # This ensures the step direction is a unit vector in L2 space
        gn = g.view(bs, -1).norm(dim=1, keepdim=True).clamp(min=1e-12)
        # Decaying step size: larger steps early, smaller later
        alpha = eps * 0.8 * ((1 - i/steps) + 0.1)
        g_unit = g / gn.view(bs, 1, 1, 1)
        # Step along the normalized gradient direction
        Xa = Xa.detach() + alpha * g_unit
        # L2 projection: scale perturbation to satisfy ||δ||_2 ≤ ε
        delta = Xa - X
        dn = delta.view(bs, -1).norm(dim=1, keepdim=True)
        # If perturbation norm exceeds ε, scale it down proportionally
        scale = torch.min(torch.ones_like(dn), eps / (dn + 1e-12))
        delta = delta * scale.view(bs, 1, 1, 1)
        # Clamp to valid data range
        Xa = torch.clamp(X + delta, X.min().item(), X.max().item())
        # Track the best adversarial example per sample (highest loss)
        with torch.no_grad():
            cl = nn.CrossEntropyLoss(reduction="none")(model(Xa), y)
            # Keep new example only if it produces higher loss (more adversarial)
            imp = cl > best_loss
            best = torch.where(imp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1), Xa, best)
            best_loss = torch.where(imp, cl, best_loss)
    return best.detach()


def _acc(model, X, y):
    """Compute per-sample accuracy (returns float tensor, not scalar)."""
    with torch.no_grad():
        return (model(X.to(DEVICE)).argmax(1) == y.to(DEVICE)).float()


def run_fgsm(model, loader, snr_levels, eps_list):
    """Run FGSM attack at multiple epsilon values and report per-SNR accuracy.
    
    This sweep over epsilon values demonstrates the trade-off between
    perturbation magnitude and attack success rate. In the V2X context,
    small ε values correspond to subtle spectral modifications that
    may be physically realizable by an adversary.
    """
    print("\n" + "=" * 50)
    print("FGSM ATTACK")
    print("=" * 50)
    n = len(snr_levels)
    res = {"eps_list": eps_list, "snr_levels": snr_levels, "per_snr": {}}
    for eps in eps_list:
        sc, st = [0]*n, [0]*n
        for X, y, snr in tqdm(loader, desc=f"  FGSM e={eps}", leave=False):
            ok = _acc(model, fgsm_attack(model, X, y, eps), y)
            for o, s in zip(ok, snr): sc[s.item()] += o.item(); st[s.item()] += 1
        a = [sc[i]/max(st[i],1) for i in range(n)]
        res["per_snr"][str(eps)] = a
        print(f"    eps={eps}: avg={np.mean(a):.4f}")
    return res


def run_pgd(model, loader, snr_levels, eps_list, steps_list):
    """Run PGD attack sweeping over epsilon and step count combinations.
    
    More PGD steps generally find stronger adversarial examples but at
    increased computational cost. This sweep helps determine the minimum
    number of steps needed for convergence.
    
    step_size is always eps/4 (standard ratio from Madry et al., 2018).
    """
    print("\n" + "=" * 50)
    print("PGD ATTACK")
    print("=" * 50)
    n = len(snr_levels)
    res = {"configurations": [], "snr_levels": snr_levels, "per_snr": {}}
    for eps in eps_list:
        for steps in steps_list:
            cfg = f"eps={eps}_steps={steps}"
            res["configurations"].append(cfg); sc, st = [0]*n, [0]*n
            for X, y, snr in tqdm(loader, desc=f"  PGD {cfg}", leave=False):
                # Standard PGD step size ratio: step_size = eps / 4
                ok = _acc(model, pgd_attack(model, X, y, eps, steps, eps/4), y)
                for o, s in zip(ok, snr): sc[s.item()] += o.item(); st[s.item()] += 1
            a = [sc[i]/max(st[i],1) for i in range(n)]
            res["per_snr"][cfg] = a
            print(f"    {cfg}: avg={np.mean(a):.4f}")
    return res


def run_cw(model, loader, snr_levels, eps_list):
    """Run C&W (Auto-PGD L2) attack at multiple epsilon values.
    
    Note: epsilon values for C&W are in L2 norm space (not L∞ like FGSM/PGD),
    so they are on a different scale. Typical L2 ε values of 1-3 correspond
    to moderate perturbation magnitudes for 64×64 spectrograms.
    """
    print("\n" + "=" * 50)
    print("C&W (Auto-PGD L2) ATTACK")
    print("=" * 50)
    n = len(snr_levels)
    res = {"eps_list": eps_list, "snr_levels": snr_levels, "per_snr": {}}
    for eps in eps_list:
        sc, st = [0]*n, [0]*n
        for X, y, snr in tqdm(loader, desc=f"  C&W e={eps}", leave=False):
            ok = _acc(model, auto_pgd_l2(model, X, y, eps, steps=5), y)
            for o, s in zip(ok, snr): sc[s.item()] += o.item(); st[s.item()] += 1
        a = [sc[i]/max(st[i],1) for i in range(n)]
        res["per_snr"][str(eps)] = a
        print(f"    L2 eps={eps}: avg={np.mean(a):.4f}")
    return res


def gen_samples(model, loader, n=5):
    """Generate clean, FGSM, and PGD adversarial spectrogram samples for visualization.
    
    These samples are used to produce side-by-side comparison figures showing
    how adversarial perturbations alter the visual appearance of spectrograms.
    
    Returns:
        Tuple of (clean_samples, fgsm_samples, pgd_samples), each with n samples
    """
    model.eval()
    cs, fs, ps = [], [], []
    for X, y, snr in loader:
        cs.append(X.cpu())
        # FGSM at ε=0.03: moderate perturbation for visual clarity
        fs.append(fgsm_attack(model, X, y, eps=0.03).cpu())
        # PGD at ε=0.05, 5 steps: stronger perturbation
        ps.append(pgd_attack(model, X, y, eps=0.05, steps=5, step_size=0.05/4).cpu())
        if sum(len(s) for s in cs) >= n: break
    return torch.cat(cs,0)[:n], torch.cat(fs,0)[:n], torch.cat(ps,0)[:n]
