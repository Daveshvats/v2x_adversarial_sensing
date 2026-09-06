#!/usr/bin/env python3
"""run_mag_baseline.py — finish the Phase-1 honesty ablations in a short process:
(1) Mag-Only model (pre-registered stream ablation), (2) energy-feature LR baseline.
Kept separate from run_train.py so each step fits within sandbox process limits.
"""
import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch

from run_train import build_dataset, train_model, energy_baseline
from receiver import FrontEnd, MagOnlyModel

OUT = os.path.join(ROOT, "results")
np.random.seed(42); torch.manual_seed(42)

t0 = time.time()
waves, labels, scen = build_dataset(42, 1000)
n_val = int(0.2 * waves.size(0))
val_w, val_y = waves[:n_val], labels[:n_val]
tr_w, tr_y = waves[n_val:], labels[n_val:]

frontend = FrontEnd()
with torch.no_grad():
    mags = []
    for i in range(0, tr_w.size(0), 512):
        mag, _ = frontend(tr_w[i:i + 512])
        mags.append(mag)
    allmag = torch.cat(mags)
    frontend.set_stats(allmag.mean(), allmag.std())

print("training Mag-Only (pre-registered ablation, same budget as dual)...",
      flush=True)
magm = MagOnlyModel()
acc_mag = train_model(magm, frontend, tr_w, tr_y, val_w, val_y, epochs=25)
torch.save({"model": magm.state_dict(),
            "mag_mean": frontend.mag_mean, "mag_std": frontend.mag_std},
           os.path.join(OUT, "checkpoint_mag.pt"))
print(f"best val acc (mag-only): {acc_mag*100:.2f}%", flush=True)

print("energy-feature honesty baseline...", flush=True)
ebase, epair = energy_baseline(waves.numpy(), labels.numpy(), n_val)
print(f"LR on energy features (4-class): {ebase*100:.2f}%  "
      f"(PC5-vs-11p pair: {epair*100:.2f}%)", flush=True)

# merge into train_report.json
rep_path = os.path.join(OUT, "train_report.json")
rep = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
rep.update({"clean_acc_mag_only": round(acc_mag, 4),
            "energy_baseline_acc": round(ebase, 4),
            "energy_baseline_pc5_11p_acc": round(epair, 4),
            "mag_epochs": 25,
            "params_mag": sum(p.numel() for p in magm.parameters())})
json.dump(rep, open(rep_path, "w"), indent=2)
print("DONE", json.dumps(rep), flush=True)
