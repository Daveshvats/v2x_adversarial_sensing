#!/usr/bin/env python3
"""run_snr_sweep_clean.py — C4 leakage fix (council P1-b, reviewer 21-d).

THE LEAK: run_snr_sweep.py builds its eval sets with rng seed 42 — the SAME
stream used to generate the seed-42 training corpus — so the *baseband*
waveforms (gen_signal draws) in the sweep are bit-identical to training
corpus members. Channel and noise draws are independent, so the leak is
mild, but a clean-task evaluation should not share baseband symbols with
the training set at all.

THE FIX: re-run the identical sweep with an eval-only rng (seed 777) that
never seeded training, and quantify the original leak with a direct
diagnostic: hash the raw baseband waveforms (pre-channel, pre-noise) of
each eval set and count exact collisions against the seed-42 training
corpus, for BOTH the old seed (expected: ~full overlap by construction)
and the new seed (expected: 0).

Outputs:
  results/snr_sweep_clean.json   — the seed-777 sweep + overlap diagnostics
    + A/B deltas vs the stored seed-42 numbers.

The old results/snr_sweep.json is kept untouched as the leaky A-arm.
"""
import os
import sys
import time
import hashlib
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from waveforms import gen_signal  # noqa: E402
import channels as CH  # noqa: E402
from run_train import build_dataset  # noqa: E402
from receiver import FrontEnd, DualStreamModel  # noqa: E402

OUT = os.path.join(ROOT, "results")
RES = os.path.join(OUT, "snr_sweep_clean.json")
SNR_RANGES = [(-10.0, 0.0), (0.0, 10.0), (10.0, 20.0), (5.0, 25.0)]
N_PER_CLASS = 250
EVAL_SEED = 777
TRAIN_SEED = 42

torch.manual_seed(EVAL_SEED)
np.random.seed(EVAL_SEED)


def raw_baseband_hashes(seed, n_per_class=N_PER_CLASS):
    """Hashes of the raw baseband waveforms an eval set with this rng seed
    would use (same draw order as run_snr_sweep.build_set)."""
    rng = np.random.default_rng(seed)
    hs = []
    for cls in range(4):
        for _ in range(n_per_class):
            s = gen_signal(cls, rng)
            hs.append(hashlib.sha256(
                np.ascontiguousarray(s).tobytes()).hexdigest())
    return set(hs)


def train_baseband_hashes():
    """Hashes of the seed-42 training-corpus baseband waveforms.

    build_dataset(seed, n) generates n_per_class = n waveforms per class
    with default_rng(seed) (see run_train.py); only the first 80% are
    training windows, but for the overlap diagnostic the full corpus is
    the honest reference (a collision anywhere is a leak).
    """
    rng = np.random.default_rng(TRAIN_SEED)
    hs = []
    for cls in range(4):
        for _ in range(1000):
            s = gen_signal(cls, rng)
            hs.append(hashlib.sha256(
                np.ascontiguousarray(s).tobytes()).hexdigest())
    return set(hs)


def build_set(snr_lo, snr_hi, n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            s = gen_signal(cls, rng)
            h = CH.draw_channel("urban", rng)
            r = np.convolve(s, h)[(len(h) - 1) // 2:
                                   (len(h) - 1) // 2 + 2048]
            p = np.mean(np.abs(r) ** 2)
            snr = rng.uniform(snr_lo, snr_hi)
            ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
            r = r + ns * (rng.standard_normal(2048) +
                          1j * rng.standard_normal(2048))
            X.append(r.astype(np.complex64))
            y.append(cls)
    X = np.array(X)
    y = np.array(y)
    idx = np.random.default_rng(seed + 1).permutation(len(y))
    return X[idx], y[idx]


def lr_features(X):
    f = np.fft.fftfreq(X.shape[1], d=1.0 / 20e6)
    l_ = (f >= -10e6) & (f < -1e6)
    h_ = (f > 1e6) & (f <= 10e6)
    m_ = (f >= -1e6) & (f <= 1e6)
    S = np.abs(np.fft.fft(X, axis=1)) ** 2
    P = np.abs(X) ** 2
    frames = P.reshape(P.shape[0], -1, 128).mean(axis=2)
    feats = np.stack([S[:, l_].mean(1), S[:, h_].mean(1), S[:, m_].mean(1),
                      P.mean(1),
                      frames.std(1) / (frames.mean(1) + 1e-9)], axis=1)
    return np.log10(feats + 1e-12)


def main():
    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    results = {"experiment": "C4 SNR sweep, CLEAN eval seed (leakage fix)",
               "config": {"n_per_class": N_PER_CLASS, "eval_seed": EVAL_SEED,
                          "train_seed": TRAIN_SEED, "scenario": "urban",
                          "snr_ranges": SNR_RANGES},
               "leakage_diagnostic": {},
               "per_range": {}}
    if os.path.exists(RES):
        results = json.load(open(RES))

    # ---- leakage diagnostic (once) ----
    if not results["leakage_diagnostic"]:
        t0 = time.time()
        tr = train_baseband_hashes()
        old = raw_baseband_hashes(TRAIN_SEED)
        new = raw_baseband_hashes(EVAL_SEED)
        results["leakage_diagnostic"] = {
            "train_corpus_waveforms": len(tr),
            "old_eval_seed": TRAIN_SEED,
            "old_eval_overlap_count": len(old & tr),
            "old_eval_overlap_fraction": round(len(old & tr) / len(old), 4),
            "new_eval_seed": EVAL_SEED,
            "new_eval_overlap_count": len(new & tr),
            "new_eval_overlap_fraction": round(len(new & tr) / len(new), 4),
            "hash": "sha256 of raw baseband (pre-channel, pre-noise) bytes",
        }
        print(f"[leak] old seed overlap: {len(old & tr)}/{len(old)}  "
              f"new seed overlap: {len(new & tr)}/{len(new)}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        with open(RES, "w") as f:
            json.dump(results, f, indent=2)

    # ---- LR baseline (trained once on the standard train split) ----
    Xtr, ytr = build_dataset(TRAIN_SEED, 1000)[0:2]
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    clf = LogisticRegression(max_iter=2000).fit(
        lr_features(Xtr.numpy()), ytr.numpy())

    for lo, hi in SNR_RANGES:
        key = f"snr=[{lo},{hi}]"
        if key in results["per_range"]:
            continue
        t0 = time.time()
        X, y = build_set(lo, hi, N_PER_CLASS, EVAL_SEED)
        xt = torch.from_numpy(X)
        with torch.no_grad():
            preds = []
            for i in range(0, xt.size(0), 256):
                preds.append(model.forward_wave(
                    xt[i:i + 256], frontend).argmax(1))
            preds = torch.cat(preds).numpy()
        cnn_acc = float((preds == y).mean())
        pm = np.isin(y, [0, 1])
        cnn_pair = float((preds[pm] == y[pm]).mean())
        lr_acc = float(clf.score(lr_features(X), y))
        results["per_range"][key] = {
            "cnn_acc": round(100 * cnn_acc, 2),
            "cnn_pc5_11p_pair_acc": round(100 * cnn_pair, 2),
            "energy_lr_acc": round(100 * lr_acc, 2),
        }
        print(f"{key}: CNN {100*cnn_acc:5.2f}%  pair {100*cnn_pair:5.2f}%  "
              f"LR {100*lr_acc:5.2f}%  ({time.time()-t0:.0f}s)", flush=True)
        with open(RES, "w") as f:
            json.dump(results, f, indent=2)

    # ---- A/B deltas vs the stored leaky seed-42 sweep ----
    old = json.load(open(os.path.join(OUT, "snr_sweep.json")))
    ab = {}
    for key, cells in results["per_range"].items():
        o = old["per_range"].get(key)
        if o:
            ab[key] = {m: round(cells[m] - o[m], 2)
                       for m in ("cnn_acc", "cnn_pc5_11p_pair_acc",
                                 "energy_lr_acc")}
    results["ab_vs_leaky_seed42"] = ab
    with open(RES, "w") as f:
        json.dump(results, f, indent=2)
    print("A/B deltas (clean - leaky):", json.dumps(ab))
    print(f"wrote {RES}")
    print("DONE")


if __name__ == "__main__":
    main()
