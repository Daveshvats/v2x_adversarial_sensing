#!/usr/bin/env python3
"""Smoke test for waveform_pgd_restarts (continuity + restart diversity)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(2)

from waveforms import ATTACK_BANDS
from receiver import FrontEnd, DualStreamModel
from attack_mask import waveform_pgd, waveform_pgd_restarts
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")
torch.manual_seed(7)

ck = torch.load(os.path.join(OUT, "checkpoint_dual_at.pt"),
                map_location="cpu", weights_only=False)
model = DualStreamModel(); model.load_state_dict(ck["model"]); model.eval()
fe = FrontEnd(); fe.set_stats(ck["mag_mean"], ck["mag_std"])

x_rx, y, h_a, p_clean = build_eval_set("urban", 10, 7)   # 30 samples
print("eval set:", x_rx.shape, "clean acc:",
      (model.forward_wave(x_rx, fe).argmax(1) == y).float().mean().item())

psr = -20.0
p_budget = p_clean * (10 ** (psr / 10.0))
band = ATTACK_BANDS["cv2x_attacker"]
steps = 4

# 1) continuity: R=1 must equal classic waveform_pgd exactly
a = waveform_pgd(model, fe, x_rx, h_a, y, p_budget, steps=steps,
                 band=band, targeted=False)
b = waveform_pgd_restarts(model, fe, x_rx, h_a, y, p_budget, steps=steps,
                          restarts=1, base_seed=7, band=band)
same = torch.equal(a["preds"], b["preds"])
print("continuity R=1 == waveform_pgd:", same,
      "| preds match:", (a["preds"] == b["preds"]).float().mean().item())

# 2) restarts improve the attack (ASR should not decrease)
c = waveform_pgd_restarts(model, fe, x_rx, h_a, y, p_budget, steps=steps,
                          restarts=4, base_seed=7, band=band)
clean = model.forward_wave(x_rx, fe).argmax(1)
def asr(preds):
    elig = clean == y
    return ((preds != y) & elig).sum().item() / max(elig.sum().item(), 1)
print(f"ASR R=1: {asr(b['preds']):.3f}   ASR R=4: {asr(c['preds']):.3f}")
print("win histogram:", torch.bincount(c["win_restart"], minlength=4).tolist())
print("SMOKE OK" if same else "CONTINUITY FAIL")
