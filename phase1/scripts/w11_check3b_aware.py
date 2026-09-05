#!/usr/bin/env python3
"""w11_check3b_aware.py — W11 auditor, supplementary to CHECK 5.

The paper (sec:papr) and CLAIMS.md C30 claim "post-PA PAPR 4.9 dB" for the
PA-aware re-optimization arm at IBO 0 (pa_aware_p3_ibo0). That number is NOT
present anywhere in results/papr_pa_results.json — this script tests whether
it is at least REPRODUCIBLE, by re-implementing the PA-aware PGD chain from
scratch on a small fresh run:

    delta -> project (src/attack_mask.project, allowed import)
          -> OWN Rapp (p=3, IBO 0, asat = sqrt(p_budget/N * 10^(IBO/10)))
          -> OWN renormalization to the budget (power control)
          -> channel (src/attack_mask.cconv) -> model -> loss

PGD-10, untargeted, urban eval set n_eval=10 seed 7 (30 windows), PSR -36 dB,
psd_margin 2.0, checkpoint_dual.pt. Post-PA PAPR of the transmit waveform
computed with this file's own code. 30 windows vs the paper's 300, so this is
an order-of-magnitude check of the untraced 4.9 dB figure.
"""
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(2)

from waveforms import ATTACK_BANDS, FS, N_SAMPLES
from receiver import FrontEnd, DualStreamModel
from attack_mask import waveform_pgd, project, cconv   # src imports (allowed)
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")


# --- own PA implementation (torch, fresh) -----------------------------------

def rapp_own_torch(x, asat, p):
    """x: (B, N) complex torch; asat: (B,) saturation amplitude."""
    r = x.abs()
    g = (1.0 + (r / asat.unsqueeze(1)) ** (2.0 * p)) ** (-1.0 / (2.0 * p))
    return x * g


def pa_aware_pgd_own(model, frontend, x_rx, h_a, y, pb, band, steps,
                     ibo_db, rapp_p, alpha_frac=0.25, psd_margin=2.0):
    """PA-aware PGD with OWN Rapp + renorm; project/cconv/model from src."""
    h = torch.as_tensor(h_a, dtype=x_rx.dtype)
    delta = torch.zeros_like(x_rx, requires_grad=True)

    def chain(d):
        d = project(d, band, pb, psd_margin=psd_margin)
        # asat sized to the INTENDED per-sample mean power (budget / N)
        asat = torch.sqrt(pb / N_SAMPLES * (10.0 ** (ibo_db / 10.0)) + 1e-20)
        yv = rapp_own_torch(d, asat, rapp_p)
        e_out = (yv.abs() ** 2).sum(dim=1)
        eps = 1e-12 * pb
        yv = yv * torch.sqrt(pb / (e_out + eps)).unsqueeze(1)
        return yv

    for _ in range(steps):
        yv = chain(delta)
        r = x_rx + cconv(yv, h)
        logits = model.forward_wave(r, frontend)
        loss = -F.cross_entropy(logits, y)          # untargeted
        g, = torch.autograd.grad(loss, delta)
        gn = g / (g.abs().norm(dim=1, keepdim=True) + 1e-12)
        step = alpha_frac * pb.sqrt().unsqueeze(1) * gn
        with torch.no_grad():
            delta = delta - step
            delta = project(delta, band, pb, psd_margin=psd_margin)
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        yv = chain(delta)
    return yv.detach(), delta.detach()


def papr_db_own_np(x):
    p = np.abs(x).astype(np.float64) ** 2
    return 10.0 * np.log10(p.max(axis=1) / p.mean(axis=1))


def mask_excess_dbr_own(y, lo=0.0, hi=10.0):
    Y = np.fft.fft(y, axis=1)
    f = np.fft.fftfreq(y.shape[1], d=1.0 / FS) / 1e6
    inb = (f >= lo) & (f <= hi)
    pbin = np.abs(Y) ** 2
    pin = np.max(np.where(inb[None, :], pbin, 0.0), axis=1)
    pout = np.max(np.where(inb[None, :], 0.0, pbin), axis=1)
    return 10.0 * np.log10(pout / pin)


def main():
    t0 = time.time()
    band = ATTACK_BANDS["cv2x_attacker"]
    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    x_rx, y, h_a, p_clean = build_eval_set("urban", 10, 7)
    pb = p_clean * (10.0 ** (-36.0 / 10.0))

    y_tx, delta = pa_aware_pgd_own(model, frontend, x_rx, h_a, y, pb, band,
                                   steps=10, ibo_db=0.0, rapp_p=3.0)
    y_np = y_tx.numpy()
    pr = papr_db_own_np(y_np)
    me = mask_excess_dbr_own(y_np)
    print(f"PA-aware arm (own chain), IBO 0, p=3, PSR -36, 30 windows:")
    print(f"  post-PA PAPR  mean {pr.mean():.2f} dB  (paper claim: 4.9 dB, "
          f"NOT stored in JSON)")
    print(f"  post-PA mask excess p95 {np.percentile(me, 95):.2f} dBr "
          f"(JSON 300-window: -12.79, i.e. non-compliant)")
    # effective? flip fraction on this small set
    with torch.no_grad():
        r = x_rx + cconv(y_tx, torch.as_tensor(h_a, dtype=x_rx.dtype))
        lg = model.forward_wave(r, frontend)
        flips = float((lg.argmax(1) != y).float().mean())
    print(f"  untargeted flip rate on 30 windows: {flips*100:.1f}%")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
