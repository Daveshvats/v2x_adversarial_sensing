#!/usr/bin/env python3
"""run_reproduction.py — reproducible harness for the ICE2CT-2026 paper's v3 pipeline.

Trains the Dual-Stream model exactly as the paper does (via src/v3_pipeline.py), runs
attacks, and reports BOTH the paper's raw-ASR metric and the corrected metrics
(robust accuracy + conditional ASR) from src/metrics.py.

Modes:
  --smoke            3 epochs, FGSM+PGD only  (pipeline integrity check, ~3 min CPU)
  (default)          full training with early stopping, FGSM+PGD+APGD-CE+APGD-DLR
  --fab              add FAB (adds ~2-4 min)
  --square-steps N   add Square with N queries (paper: 5000; adds ~5-10 min)
  --seeds 42 123 456 paper's 3-seed protocol

Every result JSON embeds the config, seed, torch version, and device so the number is
traceable (see AUDIT.md §2 for why that matters).
"""
import sys, os, json, time, argparse, platform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import v3_pipeline as V          # the paper's authentic pipeline
import metrics as M              # corrected metrics


def predict(model, mag, ift):
    with torch.no_grad():
        return model(mag, ift).argmax(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--epsilons", nargs="+", type=float, default=[0.03])
    ap.add_argument("--epochs", type=int, default=V.EPOCHS)
    ap.add_argument("--smoke", action="store_true",
                    help="3 epochs, FGSM+PGD only (fast integrity check)")
    ap.add_argument("--fab", action="store_true", help="also run FAB")
    ap.add_argument("--square-steps", type=int, default=0,
                    help="also run Square with this many queries (0=skip)")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 3

    out_dir = os.path.join(ROOT, "results", "reproduction")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    t_start = time.time()

    for seed in args.seeds:
        print(f"\n{'='*62}\n  REPRODUCTION — seed {seed}"
              f"{' (SMOKE)' if args.smoke else ''}\n{'='*62}", flush=True)
        t0 = time.time()
        V.set_seed(seed)

        # Dataset + loaders exactly as in v3_pipeline
        X_mag, X_if, y, tr_idx, va_idx = V.generate_v2x_dataset(seed=seed)
        train_loader = DataLoader(
            TensorDataset(X_mag[tr_idx], X_if[tr_idx], y[tr_idx]),
            batch_size=V.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(X_mag[va_idx], X_if[va_idx], y[va_idx]),
            batch_size=V.BATCH_SIZE, shuffle=False)

        mag_dev, if_dev, y_dev = X_mag[va_idx], X_if[va_idx], y[va_idx]

        # Train (paper recipe)
        model = V.DualStreamModel()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  parameters: {n_params:,}", flush=True)
        best_acc = V.train_standard(model, train_loader, val_loader,
                                    epochs=args.epochs)
        print(f"  best clean test acc: {best_acc*100:.2f}%", flush=True)

        clean_preds = predict(model, mag_dev, if_dev)

        seed_result = {
            "seed": seed,
            "n_params": n_params,
            "clean_acc": best_acc,
            "attacks": {},
        }

        attack_fns = {
            "fgsm": lambda e: V.fgsm_attack(model, mag_dev, if_dev, y_dev, eps=e),
            "pgd20": lambda e: V.pgd_attack(model, mag_dev, if_dev, y_dev, eps=e),
        }
        if not args.smoke:
            attack_fns["apgd_ce"] = lambda e: V.apgd_attack(
                model, mag_dev, if_dev, y_dev, eps=e, steps=100, loss_type="ce")
            attack_fns["apgd_dlr"] = lambda e: V.apgd_attack(
                model, mag_dev, if_dev, y_dev, eps=e, steps=100, loss_type="dlr")
        if args.fab and not args.smoke:
            attack_fns["fab"] = lambda e: V.fab_attack(
                model, mag_dev, if_dev, y_dev, eps=e, steps=100)
        if args.square_steps > 0 and not args.smoke:
            attack_fns["square"] = lambda e: V.square_attack(
                model, mag_dev, if_dev, y_dev, eps=e, steps=args.square_steps)

        for name, fn in attack_fns.items():
            seed_result["attacks"][name] = {}
            for eps in args.epsilons:
                m_a, i_a = fn(eps)
                adv_preds = predict(model, m_a, i_a)
                mm = M.evaluate_attack(
                    lambda inp: predict(model, *inp),
                    (mag_dev, if_dev), y_dev.numpy(),
                    (m_a, i_a))
                seed_result["attacks"][name][f"eps={eps}"] = {
                    "raw_asr": round(mm["raw_asr"] * 100, 2),
                    "robust_acc": round(mm["robust_acc"] * 100, 2),
                    "cond_asr": round(mm["cond_asr"] * 100, 2),
                    "n_eligible": mm["n_eligible"],
                }
                print(f"    {name:9s} eps={eps:.3f}: "
                      f"raw {mm['raw_asr']*100:5.2f}%  "
                      f"cond {mm['cond_asr']*100:5.2f}%  "
                      f"robust {mm['robust_acc']*100:5.2f}%", flush=True)

        seed_result["elapsed_s"] = round(time.time() - t0, 1)
        all_results.append(seed_result)

        # per-seed JSON (traceable: config + env embedded)
        payload = {
            "experiment": "v3_pipeline reproduction (honest release)",
            "config": {
                "epochs": args.epochs,
                "epsilons": args.epsilons,
                "smoke": args.smoke,
                "attacks": list(attack_fns.keys()),
                "batch_size": V.BATCH_SIZE,
            },
            "env": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "device": "cpu",
            },
            "seeds": all_results,
        }
        tag = "smoke" if args.smoke else "repro"
        path = os.path.join(out_dir, f"{tag}_seed{'_'.join(map(str, args.seeds))}.json")
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {path}", flush=True)

    print(f"\nTotal elapsed: {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
