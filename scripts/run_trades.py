#!/usr/bin/env python3
"""run_trades.py — TRADES-style ablation (CLAIMS C17, Wave 7).

Same protocol as run_at_defense.py (v3 recipe, mask-matched generator,
at_prob 0.5, steps 5, PSR U[-25,-5], seed 42, 16 epochs) with the TRADES
loss replacing standard AT:

    L = CE(logits_clean, y) + beta * KL(softmax(logits_clean).detach() ||
                                        log_softmax(logits_adv))

Differences vs standard AT (disclosed):
  * adversarial batches train on BOTH the clean and the adversarial view
    (TRADES' trade-off objective) instead of the adversarial view alone;
  * CutMix is disabled on adversarial batches (the KL term regularizes the
    clean/adv agreement; mixing patches across the two views would change
    the defense being evaluated). Clean batches keep the full v3 recipe.
  * beta = 6.0 (the canonical TRADES default).

Checkpoints: results/trades_checkpoint.pt (resumable per epoch),
final model: results/checkpoint_dual_trades.pt, report:
results/trades_train_report.json. Attack it afterwards with
run_attack.py --checkpoint checkpoint_dual_trades.pt.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(1)

import torch.nn.functional as F

from waveforms import CLASS_NAMES, ATTACK_BANDS
from receiver import FrontEnd, DualStreamModel, smoothed_ce, tf_cutmix
from attack_mask import cconv, waveform_pgd
from run_train import build_dataset
from run_at_defense import generate_adversarial_batch, evaluate, atomic_save

OUT = os.path.join(ROOT, "results")


def _paths(tag):
    """tag=None -> canonical seed-42 paths; tag='s43' -> per-seed files
    (trades_checkpoint_s43.pt, checkpoint_dual_trades_s43.pt,
    trades_train_report_s43.json)."""
    if not tag:
        return (os.path.join(OUT, "trades_checkpoint.pt"),
                os.path.join(OUT, "checkpoint_dual_trades.pt"),
                os.path.join(OUT, "trades_train_report.json"))
    return (os.path.join(OUT, f"trades_checkpoint_{tag}.pt"),
            os.path.join(OUT, f"checkpoint_dual_trades_{tag}.pt"),
            os.path.join(OUT, f"trades_train_report_{tag}.json"))


def train_epoch_trades(model, frontend, opt, sched, tr_w, tr_y, cfg, np_rng):
    """One epoch: standard v3 batches + TRADES adversarial batches."""
    model.train()
    B = tr_w.size(0)
    perm = torch.randperm(B)
    tot_loss, n_adv, n_b = 0.0, 0, 0
    for bi, i in enumerate(range(0, B, cfg["batch_size"])):
        idx = perm[i:i + cfg["batch_size"]]
        wb, yb = tr_w[idx], tr_y[idx]
        if np_rng.random() < cfg["gauss_prob"]:
            wb = wb + cfg["gauss_std"] * torch.complex(
                torch.randn_like(wb.real), torch.randn_like(wb.imag))
        if np_rng.random() < cfg["at_prob"]:
            scenario = "urban" if (bi % 2 == 0) else "highway"
            wb_adv = generate_adversarial_batch(model, frontend, wb, yb,
                                                np_rng, cfg["steps"],
                                                cfg["psr_lo"], cfg["psr_hi"],
                                                cfg["psd_margin"], scenario)
            mag_c, ifr_c = frontend(wb)
            mag_a, ifr_a = frontend(wb_adv)
            logits_c = model(mag_c, ifr_c)
            logits_a = model(mag_a, ifr_a)
            ce = F.cross_entropy(logits_c, yb, label_smoothing=0.1)
            kl = F.kl_div(F.log_softmax(logits_a, dim=1),
                          F.softmax(logits_c, dim=1).detach(),
                          reduction="batchmean")
            loss = ce + cfg["beta"] * kl
            n_adv += 1
        else:
            mag, ifr = frontend(wb)
            if np_rng.random() < cfg["cutmix_prob"]:
                mag, ifr, y_a, y_b, lam = tf_cutmix(mag, ifr, yb)
            else:
                y_a, y_b, lam = yb, yb, 1.0
            logits = model(mag, ifr)
            loss = smoothed_ce(logits, y_a, y_b, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item() * len(idx)
        n_b += 1
    sched.step()
    return tot_loss / B, n_adv, n_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--beta", type=float, default=6.0)
    ap.add_argument("--at-prob", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--psr-lo", type=float, default=-25.0)
    ap.add_argument("--psr-hi", type=float, default=-5.0)
    ap.add_argument("--psd-margin", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--cutmix-prob", type=float, default=0.3)
    ap.add_argument("--gauss-prob", type=float, default=0.3)
    ap.add_argument("--gauss-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default=None,
                    help="suffix for per-seed artifacts (e.g. 's43')")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-time-sec", type=float, default=480.0)
    args = ap.parse_args()

    CKPT, FINAL, REPORT = _paths(args.tag)

    cfg = {"epochs_total": args.epochs, "beta": args.beta,
           "at_prob": args.at_prob, "steps": args.steps,
           "psr_lo": args.psr_lo, "psr_hi": args.psr_hi,
           "psd_margin": args.psd_margin, "batch_size": args.batch_size,
           "lr": args.lr, "cutmix_prob": args.cutmix_prob,
           "gauss_prob": args.gauss_prob, "gauss_std": args.gauss_std,
           "seed": args.seed}

    t_chunk = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed + 1000)

    print("[data] building channelized dataset (seed %d)..." % args.seed,
          flush=True)
    waves, labels, _ = build_dataset(args.seed, 1000)
    n_val = int(0.2 * waves.size(0))
    val_w, val_y = waves[:n_val], labels[:n_val]
    tr_w, tr_y = waves[n_val:], labels[n_val:]
    frontend = FrontEnd()
    with torch.no_grad():
        mags = [frontend(tr_w[i:i + 512])[0]
                for i in range(0, tr_w.size(0), 512)]
        allmag = torch.cat(mags)
        frontend.set_stats(allmag.mean(), allmag.std())
    print(f"[data] train {tr_w.size(0)}  val {val_w.size(0)}  "
          f"({time.time()-t_chunk:.1f}s)", flush=True)

    model = DualStreamModel()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    start_epoch, history, elapsed_total = 0, [], 0.0

    if os.path.exists(CKPT):
        if not args.resume:
            print("ERROR: results/trades_checkpoint.pt exists — pass "
                  "--resume", flush=True)
            sys.exit(2)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        old = ck["config"]
        mism = {k: (old.get(k), cfg[k]) for k in cfg
                if k not in ("epochs_total",) and old.get(k) != cfg[k]}
        if mism:
            print(f"ERROR: config mismatch vs checkpoint: {mism}", flush=True)
            sys.exit(2)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        best_acc = ck["best_acc"]
        best_state = ck["best_state"]
        start_epoch = ck["epoch"]
        history = ck["history"]
        elapsed_total = ck.get("elapsed_total", 0.0)
        frontend.set_stats(ck["mag_mean"], ck["mag_std"])
        torch.set_rng_state(ck["rng_torch"])
        np_rng.bit_generator.state = ck["rng_np"]
        print(f"[resume] continuing from epoch {start_epoch} "
              f"(best val acc {best_acc*100:.2f}%)", flush=True)

    ep = start_epoch
    last_dur = history[-1]["epoch_s"] if history else None
    while ep < args.epochs:
        elapsed = time.time() - t_chunk
        look = last_dur if last_dur is not None else 90.0
        if elapsed + look > args.max_time_sec:
            print(f"[guard] stopping before epoch {ep+1} "
                  f"(elapsed {elapsed:.0f}s + lookahead {look:.0f}s)",
                  flush=True)
            break
        t_ep = time.time()
        loss, n_adv, n_b = train_epoch_trades(model, frontend, opt, sched,
                                              tr_w, tr_y, cfg, np_rng)
        acc = evaluate(model, frontend, val_w, val_y)
        if acc > best_acc + 1e-4:
            best_acc, best_state = acc, \
                {k: v.clone() for k, v in model.state_dict().items()}
        ep += 1
        last_dur = time.time() - t_ep
        history.append({"epoch": ep, "loss": round(loss, 4),
                        "val_acc": round(acc, 4), "n_adv_batches": n_adv,
                        "n_batches": n_b, "epoch_s": round(last_dur, 1)})
        elapsed_total += last_dur
        atomic_save({
            "epoch": ep, "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "best_acc": best_acc, "best_state": best_state,
            "mag_mean": frontend.mag_mean, "mag_std": frontend.mag_std,
            "rng_torch": torch.get_rng_state(),
            "rng_np": np_rng.bit_generator.state,
            "config": cfg, "history": history,
            "elapsed_total": elapsed_total,
        }, CKPT)
        print(f"  epoch {ep:3d}/{args.epochs}: loss {loss:.4f}  "
              f"val acc {acc*100:5.2f}%  best {best_acc*100:5.2f}%  "
              f"TRADES-batches {n_adv}/{n_b}  ({last_dur:.0f}s)", flush=True)

    completed = ep >= args.epochs
    report = {
        "experiment": "TRADES-style ablation (C17): beta-KL clean/adv "
                      "objective with the mask-matched generator",
        "config": cfg,
        "recipe": "v3 recipe; adversarial batches: smooth-CE(clean) + "
                  f"beta({args.beta})*KL(adv||clean.detach()); CutMix "
                  "disabled on adversarial batches (disclosed)",
        "n_train": tr_w.size(0), "n_val": val_w.size(0),
        "classes": CLASS_NAMES,
        "history": history,
        "best_val_acc": round(best_acc, 4),
        "completed": completed,
        "elapsed_total_s": round(elapsed_total, 1),
        "threads": 1,
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)

    if completed:
        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save({"model": model.state_dict(),
                    "mag_mean": frontend.mag_mean,
                    "mag_std": frontend.mag_std,
                    "config": {"seed": args.seed, "epochs": args.epochs,
                               "beta": args.beta, "at_prob": args.at_prob,
                               "at_steps": args.steps,
                               "at_psr_db": [args.psr_lo, args.psr_hi],
                               "psd_margin": args.psd_margin}},
                   FINAL)
        print(f"TRAINING COMPLETE: best val acc {best_acc*100:.2f}% "
              f"-> {FINAL}", flush=True)
    else:
        print(f"CHUNK DONE at epoch {ep}/{args.epochs} — re-run with "
              f"--resume (total {elapsed_total:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
