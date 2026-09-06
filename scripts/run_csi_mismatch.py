#!/usr/bin/env python3
"""run_csi_mismatch.py — C13: attack robustness to attacker CSI error.

The prototype attack assumes the attacker knows its attacker->receiver
channel h_a perfectly (white-box with CSI). This script evaluates two
realistic degradations, evaluated against the TRUE channel:

  no-CSI      : the attacker optimizes delta against an INDEPENDENT channel
                draw h_est (knows the channel DISTRIBUTION, not the
                realization); the resulting waveform is transmitted through
                the true h_a.
  csi-err-0.3 : the attacker knows h_a up to an additive error of relative
                magnitude 0.3: h_est = h_a + 0.3*||h_a||*CN(0,I)/sqrt(L).

Settings: urban, untargeted, seed 7, PGD-10, psd_margin 2.0, mask + genie,
PSR points {-30, -25, -20, -15, -10}. Writes results/csi_mismatch.json
(incrementally after each condition; safe under process kills).
"""
import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

from run_attack import build_eval_set
from waveforms import ATTACK_BANDS, NOISE_CLASS
import channels as CH
from receiver import FrontEnd, DualStreamModel
from attack_mask import cconv, project, cond_asr

OUT = os.path.join(ROOT, "results")
RES = os.path.join(OUT, "csi_mismatch.json")
PSR_LIST = [-30.0, -25.0, -20.0, -15.0, -10.0]
STEPS = 10

torch.manual_seed(7)
np.random.seed(7)

ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                  map_location="cpu", weights_only=False)
model = DualStreamModel(); model.load_state_dict(ckpt["model"]); model.eval()
frontend = FrontEnd(); frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

x_rx, y, h_a, p_clean = build_eval_set("urban", 100, seed=7)
B = x_rx.size(0)

with torch.no_grad():
    clean_preds = []
    for i in range(0, B, 256):
        clean_preds.append(model.forward_wave(x_rx[i:i + 256], frontend)
                           .argmax(1))
    clean_preds = torch.cat(clean_preds)
print(f"clean acc: {(clean_preds == y).float().mean()*100:.1f}%")

# attacker's estimated channels (seeded separately for reproducibility)
rng_est = np.random.default_rng(701)
h_nocs = np.stack([CH.draw_channel("urban", rng_est) for _ in range(B)])
h_nocs = torch.from_numpy(h_nocs)
# additive CSI error, relative magnitude 0.3
h_true = h_a
err = (torch.randn_like(h_true) + 1j * torch.randn_like(h_true)) \
    / np.sqrt(2)
h_err = h_true + 0.3 * h_true.abs().norm(dim=1, keepdim=True) \
    / np.sqrt(h_true.size(1)) * err

VARIANTS = {"perfect_csi": h_true, "csi_err_0.3": h_err, "no_csi": h_nocs}


def attack_with_estimated_channel(h_est, band, p_budget):
    """PGD optimized against h_est; evaluated through the TRUE channel."""
    delta = torch.zeros_like(x_rx, requires_grad=True)
    for _ in range(STEPS):
        r = x_rx + cconv(delta, h_est)
        logits = model.forward_wave(r, frontend)
        loss = -F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, delta)
        gn = g / (g.abs().norm(dim=1, keepdim=True) + 1e-12)
        with torch.no_grad():
            step = 0.25 * p_budget.sqrt().unsqueeze(1) * gn
            delta = delta - step
            delta = project(delta, band, p_budget)
        delta = delta.detach().requires_grad_(True)
    with torch.no_grad():
        r_adv = x_rx + cconv(delta, h_true)     # TRUE channel
        preds = model.forward_wave(r_adv, frontend).argmax(1)
    return preds


results = {"experiment": "attacker CSI-error robustness (C13)",
           "config": {"scenario": "urban", "mode": "untargeted", "seed": 7,
                      "steps": STEPS, "psd_margin": 2.0, "n_eval": B,
                      "csi_error": "0.3 relative additive",
                      "no_csi": "independent draw from same scenario"},
           "runs": {}}
if os.path.exists(RES):
    results = json.load(open(RES))
    results["config"] = {**results["config"], **results["config"]}

t0 = time.time()
for vname, h_est in VARIANTS.items():
    for setting, band in [("genie", None),
                          ("cv2x_mask", ATTACK_BANDS["cv2x_attacker"])]:
        for psr in PSR_LIST:
            key = f"{vname}/{setting}/psr={psr:+.0f}dB"
            if key in results["runs"]:
                continue
            if time.time() - t0 > 200:
                print("[guard] time budget reached; re-run to continue",
                      flush=True)
                with open(RES, "w") as f:
                    json.dump(results, f, indent=2)
                sys.exit(0)
            p_budget = p_clean * (10 ** (psr / 10.0))
            preds = attack_with_estimated_channel(h_est, band, p_budget)
            c_asr, n_elig = cond_asr(clean_preds, preds, y)
            results["runs"][key] = {"cond_asr": round(100 * c_asr, 2),
                                    "n_eligible": n_elig}
            print(f"  {key:34s}: cond-ASR {100*c_asr:5.1f}%", flush=True)
            with open(RES, "w") as f:
                json.dump(results, f, indent=2)

with open(RES, "w") as f:
    json.dump(results, f, indent=2)
print(f"wrote {RES}")
print("DONE")
