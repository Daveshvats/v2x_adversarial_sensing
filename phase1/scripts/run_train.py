#!/usr/bin/env python3
"""run_train.py — train Dual-Stream and Mag-Only models on the post-FCC coexistence
dataset, save checkpoints, and report the energy-feature honesty baseline.

Dataset: 4,000 waveforms (1,000/class), mixed TR 37.885-style scenarios (highway /
urban / rural drawn per sample), SNR uniform [5, 25] dB, 80/20 split.
Recipe: Adam 5e-4, cosine schedule, label smoothing 0.1, Gaussian aug p=0.3,
TF-CutMix p=0.3, early stopping patience 15 (v3 recipe, for comparability).

Saves: results/checkpoint_dual.pt, results/checkpoint_mag.pt,
       results/train_report.json
Runtime: ~2-3 min on CPU (25 epochs max here; use --epochs for more).
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from waveforms import gen_dataset, CLASS_NAMES
import channels as CH
from receiver import (FrontEnd, DualStreamModel, MagOnlyModel,
                      tf_cutmix, smoothed_ce)

OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)


def build_dataset(seed, n_per_class, frontend_stats_seed=42):
    """Generate waveforms, apply per-sample mixed-scenario channels, return tensors."""
    X, y = gen_dataset(n_per_class=n_per_class, seed=seed)
    rng = np.random.default_rng(seed + 1)
    B = X.shape[0]
    X_rx = np.empty_like(X)
    scen_labels = []
    for i in range(B):
        scen = CH.SCENARIO_NAMES[i % 3]      # deterministic interleaving
        h = CH.draw_channel(scen, rng)
        L = X.shape[1]
        yv = np.convolve(X[i], h)[(len(h) - 1) // 2:(len(h) - 1) // 2 + L]
        p = np.mean(np.abs(yv) ** 2)
        snr = rng.uniform(5, 25)
        ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
        yv = yv + ns * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        X_rx[i] = yv.astype(np.complex64)
        scen_labels.append(scen)
    idx = np.random.default_rng(seed + 2).permutation(B)
    return (torch.from_numpy(X_rx[idx]),
            torch.from_numpy(y[idx]).long(),
            np.array(scen_labels)[idx])


def energy_baseline(X_np, y_np, n_val):
    """Honesty baseline: logistic regression on region-energy + envelope features.
    This is what 'you don't need DL' looks like; the paper must beat it.

    Evaluated on the SAME validation split as the CNN (first n_val samples of
    the permuted dataset) so the comparison is apples-to-apples. Also reports
    the PC5-vs-11p-only accuracy: those two classes are co-channel in the ITS
    band, so this pair is where region energy genuinely cannot separate and
    the DL-necessity claim must stand or fall.
    """
    from sklearn.linear_model import LogisticRegression
    f = np.fft.fftfreq(X_np.shape[1], d=1.0 / 20e6)
    lo = (f >= -10e6) & (f < -1e6)
    hi = (f > 1e6) & (f <= 10e6)
    mid = (f >= -1e6) & (f <= 1e6)
    S = np.abs(np.fft.fft(X_np, axis=1)) ** 2
    e_lo = S[:, lo].mean(axis=1)
    e_hi = S[:, hi].mean(axis=1)
    e_mid = S[:, mid].mean(axis=1)
    total = (np.abs(X_np) ** 2).mean(axis=1)
    # short-time power variance (PAPR proxy, 128-sample frames)
    P = np.abs(X_np) ** 2
    frames = P.reshape(P.shape[0], -1, 128).mean(axis=2)
    pv = frames.std(axis=1) / (frames.mean(axis=1) + 1e-9)
    feats = np.stack([e_lo, e_hi, e_mid, total, pv], axis=1)
    feats = np.log10(feats + 1e-12)
    Xtr, Xte = feats[n_val:], feats[:n_val]      # SAME split as the CNN
    ytr, yte = y_np[n_val:], y_np[:n_val]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    acc4 = float(clf.score(Xte, yte))
    # PC5-vs-11p pair (co-channel in ITS band): the honest hard problem.
    # The permutation order is inherited, so the first 20% of the pair subset
    # is the pair members of the same validation split.
    pair = np.isin(y_np, [0, 1])
    fp, yp = feats[pair], y_np[pair]
    n_val_pair = int(0.2 * pair.sum())
    clf2 = LogisticRegression(max_iter=2000)
    clf2.fit(fp[n_val_pair:], yp[n_val_pair:])
    acc_pair = float(clf2.score(fp[:n_val_pair], yp[:n_val_pair]))
    return acc4, acc_pair


def train_model(model, frontend, train_waves, train_y, val_waves, val_y,
                epochs, batch_size=64, lr=5e-4, cutmix_prob=0.3,
                gauss_prob=0.3, gauss_std=0.02):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = train_waves.size(0)
    best_acc, best_state, patience, no_improve = 0.0, None, 15, 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(B)
        tot_loss = 0.0
        for i in range(0, B, batch_size):
            idx = perm[i:i + batch_size]
            wb, yb = train_waves[idx], train_y[idx]
            # gaussian augmentation on waveform (small complex noise)
            if np.random.rand() < gauss_prob:
                wb = wb + gauss_std * torch.complex(
                    torch.randn_like(wb.real), torch.randn_like(wb.imag))
            mag, ifr = frontend(wb)
            if np.random.rand() < cutmix_prob:
                mag, ifr, y_a, y_b, lam = tf_cutmix(mag, ifr, yb)
            else:
                y_a, y_b, lam = yb, yb, 1.0
            logits = model(mag, ifr)
            loss = smoothed_ce(logits, y_a, y_b, lam)
            opt.zero_grad(); loss.backward(); opt.step()
            tot_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, val_waves.size(0), 256):
                lg = model.forward_wave(val_waves[i:i + 256], frontend)
                preds.append(lg.argmax(1))
            acc = (torch.cat(preds) == val_y).float().mean().item()
        if acc > best_acc + 1e-4:
            best_acc, no_improve = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        print(f"    epoch {ep+1:3d}: loss {tot_loss/B:.4f}  val acc {acc*100:5.2f}%"
              f"  best {best_acc*100:5.2f}%", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-mag", action="store_true",
                    help="train only the dual-stream model (faster, resumable)")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(args.seed)
    print("[1/4] generating channelized dataset...", flush=True)
    waves, labels, scen = build_dataset(args.seed, args.n_per_class)
    n_val = int(0.2 * waves.size(0))
    val_w, val_y = waves[:n_val], labels[:n_val]
    tr_w, tr_y = waves[n_val:], labels[n_val:]
    print(f"    train {tr_w.size(0)}  val {val_w.size(0)}", flush=True)

    print("[2/4] fitting front-end normalization stats...", flush=True)
    frontend = FrontEnd()
    with torch.no_grad():
        mags = []
        for i in range(0, tr_w.size(0), 512):
            mag, _ = frontend(tr_w[i:i + 512])
            mags.append(mag)
        allmag = torch.cat(mags)
        frontend.set_stats(allmag.mean(), allmag.std())

    report = {"seed": args.seed, "epochs_max": args.epochs,
              "n_train": tr_w.size(0), "n_val": val_w.size(0),
              "classes": CLASS_NAMES, "scenarios": "mixed"}

    print("[3/4] training Dual-Stream model...", flush=True)
    dual = DualStreamModel()
    acc_dual = train_model(dual, frontend, tr_w, tr_y, val_w, val_y, args.epochs)
    print(f"    best val acc (dual): {acc_dual*100:.2f}%", flush=True)

    # save IMMEDIATELY (sandbox-safe: this artifact survives process kills)
    torch.save({"model": dual.state_dict(),
                "mag_mean": frontend.mag_mean, "mag_std": frontend.mag_std,
                "config": {"seed": args.seed, "n_train": tr_w.size(0),
                           "n_val": val_w.size(0), "epochs": args.epochs}},
               os.path.join(OUT, "checkpoint_dual.pt"))
    print("    checkpoint_dual.pt saved", flush=True)

    acc_mag, ebase = None, None
    if not args.skip_mag:
        print("[3b/4] training Mag-Only model (pre-registered ablation)...",
              flush=True)
        magm = MagOnlyModel()
        acc_mag = train_model(magm, frontend, tr_w, tr_y, val_w, val_y,
                              args.epochs)
        print(f"    best val acc (mag-only): {acc_mag*100:.2f}%", flush=True)
        torch.save({"model": magm.state_dict(),
                    "mag_mean": frontend.mag_mean,
                    "mag_std": frontend.mag_std},
                   os.path.join(OUT, "checkpoint_mag.pt"))
        print("    checkpoint_mag.pt saved", flush=True)

    print("[4/4] energy-feature honesty baseline...", flush=True)
    ebase, epair = energy_baseline(waves.numpy(), labels.numpy(), n_val)
    print(f"    LR on energy features (4-class): {ebase*100:.2f}%  "
          f"(PC5-vs-11p pair only: {epair*100:.2f}%)", flush=True)

    # dual-model per-class + PC5-vs-11p pair accuracy on the SAME val split
    with torch.no_grad():
        preds = []
        for i in range(0, val_w.size(0), 256):
            lg = dual.forward_wave(val_w[i:i + 256], frontend)
            preds.append(lg.argmax(1))
        preds = torch.cat(preds)
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        m = val_y == c
        per_class[name] = round(100 * (preds[m] == c).float().mean().item(), 2)
    pm = torch.isin(val_y, torch.tensor([0, 1]))
    pair_acc = (preds[pm] == val_y[pm]).float().mean().item()
    print(f"    CNN PC5-vs-11p pair acc: {pair_acc*100:.2f}%", flush=True)

    report.update({
        "clean_acc_dual": round(acc_dual, 4),
        "clean_acc_mag_only": (round(acc_mag, 4) if acc_mag is not None else None),
        "energy_baseline_acc": round(ebase, 4),
        "energy_baseline_pc5_11p_acc": round(epair, 4),
        "cnn_pc5_11p_pair_acc": round(pair_acc, 4),
        "val_acc_per_class": per_class,
        "params_dual": sum(p.numel() for p in dual.parameters()),
        "elapsed_s": round(time.time() - t0, 1),
    })
    if acc_mag is not None:
        report["params_mag"] = sum(p.numel() for p in magm.parameters())
    with open(os.path.join(OUT, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
