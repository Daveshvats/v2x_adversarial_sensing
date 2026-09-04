#!/usr/bin/env python3
"""run_snr_sweep.py — C4: clean-task accuracy vs SNR (CNN vs energy-LR).

Evaluates the trained dual CNN and the energy-feature LR baseline on
urban-channel eval sets whose SNR is drawn from low ranges. Uses the SAME
waveform/channel/noise generation as training, only the SNR range varies.
Writes results/snr_sweep.json incrementally.
"""
import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

from run_train import energy_baseline
from waveforms import gen_dataset, gen_signal
import channels as CH
from receiver import FrontEnd, DualStreamModel

OUT = os.path.join(ROOT, "results")
RES = os.path.join(OUT, "snr_sweep.json")
SNR_RANGES = [(-10.0, 0.0), (0.0, 10.0), (10.0, 20.0), (5.0, 25.0)]
N_PER_CLASS = 250
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                  map_location="cpu", weights_only=False)
model = DualStreamModel(); model.load_state_dict(ckpt["model"]); model.eval()
frontend = FrontEnd(); frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])


def build_set(snr_lo, snr_hi, n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            s = gen_signal(cls, rng)
            h = CH.draw_channel("urban", rng)
            r = np.convolve(s, h)[(len(h) - 1) // 2:(len(h) - 1) // 2 + 2048]
            p = np.mean(np.abs(r) ** 2)
            snr = rng.uniform(snr_lo, snr_hi)
            ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
            r = r + ns * (rng.standard_normal(2048) +
                          1j * rng.standard_normal(2048))
            X.append(r.astype(np.complex64)); y.append(cls)
    X = np.array(X); y = np.array(y)
    idx = np.random.default_rng(seed + 1).permutation(len(y))
    return X[idx], y[idx]


results = {"experiment": "clean-task SNR sweep (C4): CNN vs energy-LR",
           "config": {"n_per_class": N_PER_CLASS, "seed": SEED,
                      "scenario": "urban", "snr_ranges": SNR_RANGES},
           "per_range": {}}
if os.path.exists(RES):
    results = json.load(open(RES))

for lo, hi in SNR_RANGES:
    key = f"snr=[{lo},{hi}]"
    if key in results["per_range"]:
        continue
    t0 = time.time()
    X, y = build_set(lo, hi, N_PER_CLASS, SEED)
    xt = torch.from_numpy(X)
    with torch.no_grad():
        preds = []
        for i in range(0, xt.size(0), 256):
            preds.append(model.forward_wave(xt[i:i + 256], frontend)
                         .argmax(1))
        preds = torch.cat(preds).numpy()
    cnn_acc = float((preds == y).mean())
    # CNN PC5-vs-11p pair
    pm = np.isin(y, [0, 1])
    cnn_pair = float((preds[pm] == y[pm]).mean())
    # energy-LR trained on the standard train split, evaluated here
    # (reuse energy_baseline's feature construction via a small wrapper)
    from run_train import build_dataset
    f = np.fft.fftfreq(X.shape[1], d=1.0 / 20e6)
    l_ = (f >= -10e6) & (f < -1e6); h_ = (f > 1e6) & (f <= 10e6)
    m_ = (f >= -1e6) & (f <= 1e6)
    S = np.abs(np.fft.fft(X, axis=1)) ** 2
    P = np.abs(X) ** 2
    frames = P.reshape(P.shape[0], -1, 128).mean(axis=2)
    feats = np.stack([S[:, l_].mean(1), S[:, h_].mean(1), S[:, m_].mean(1),
                      P.mean(1),
                      frames.std(1) / (frames.mean(1) + 1e-9)], axis=1)
    feats = np.log10(feats + 1e-12)
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = build_dataset(42, 1000)[0:2]
    # LR features for the TRAIN set (same recipe)
    Str = np.abs(np.fft.fft(Xtr.numpy(), axis=1)) ** 2
    Ptr = np.abs(Xtr.numpy()) ** 2
    ftr = Ptr.reshape(Ptr.shape[0], -1, 128).mean(axis=2)
    ftr_feats = np.stack([Str[:, l_].mean(1), Str[:, h_].mean(1),
                          Str[:, m_].mean(1), Ptr.mean(1),
                          ftr.std(1) / (ftr.mean(1) + 1e-9)], axis=1)
    ftr_feats = np.log10(ftr_feats + 1e-12)
    clf = LogisticRegression(max_iter=2000).fit(ftr_feats, ytr.numpy())
    lr_acc = float(clf.score(feats, y))
    results["per_range"][key] = {
        "cnn_acc": round(100 * cnn_acc, 2),
        "cnn_pc5_11p_pair_acc": round(100 * cnn_pair, 2),
        "energy_lr_acc": round(100 * lr_acc, 2),
    }
    print(f"{key}: CNN {100*cnn_acc:5.2f}%  pair {100*cnn_pair:5.2f}%  "
          f"LR {100*lr_acc:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    with open(RES, "w") as fjson:
        json.dump(results, fjson, indent=2)

print(f"wrote {RES}")
print("DONE")
