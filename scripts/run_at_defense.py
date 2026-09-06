#!/usr/bin/env python3
"""run_at_defense.py — MASK-MATCHED ADVERSARIAL TRAINING (paper §Defenses, CLAIMS C16).

Pre-registered scheme (run once, report whatever happens — no tuning to outcome):
  * The standard v3 training recipe from run_train.py is kept EXACTLY
    (build_dataset(42, 1000) channelized waveforms, FrontEnd stats fitted on
    the train split, Adam lr 5e-4 / weight-decay 1e-4, cosine schedule over the
    full epoch budget, label smoothing 0.1 via smoothed_ce, Gaussian waveform
    aug p=0.3 std=0.02, TF-CutMix p=0.3, batch 64, 80/20 split). The ONLY
    change: a random subset of minibatches is adversarially perturbed.
  * AT step — per minibatch, with probability --at-prob (default 0.5):
      1. draw a per-sample attacker PSR uniformly from [--psr-lo, --psr-hi]
         (default [-25, -5] dB);
      2. draw a FRESH attacker->victim channel h_a via channels.draw_channel,
         scenario alternating urban/highway by batch index (bi even = urban);
      3. generate delta with the SAME mask-constrained waveform PGD used in
         the attacks — waveform_pgd(model, frontend, wb, h_a, yb, p_budget,
         steps=--steps, band=ATTACK_BANDS["cv2x_attacker"], psd_margin=2.0,
         targeted=False) — against the CURRENT training model (standard AT,
         not TRADES). p_budget = p_rx * 10^(PSR/10) with p_rx = per-sample
         window energy sum(|wb|^2) of the (post-Gaussian-aug) clean batch,
         the training-time stand-in for the received waveform (note: eval PSR
         is referenced to the pre-noise received energy — ~1-2 dB offset at
         the [5,25] dB training SNRs; disclosed, matches the task spec);
      4. train on (wb + cconv(delta, h_a), yb) with the normal recipe.
    Otherwise the batch trains clean, as usual.
  * Correctness of the AT gradient path: delta is generated under
    torch.enable_grad() but is DATA, not a differentiable path — waveform_pgd
    returns it detached, so the training loss never backpropagates through
    the generator (standard adversarial-example semantics). wb itself is a
    no-grad data leaf.
  * Model mode during generation (documented choice): model is temporarily
    set to eval() (BatchNorm running stats, dropout off) while delta is
    generated, then back to train() for the loss step. Rationale: avoids
    batch-stat leakage into the generated examples and makes the training
    perturbation match the eval-time threat model (attacks run in eval
    mode). BN running stats still update on adversarial batches during the
    loss step.

Sandbox safety: {model, optimizer, scheduler, epoch counter, best acc/state,
frontend stats, RNG states, history} are saved to results/at_checkpoint.pt
after EVERY epoch (atomic tmp+rename). --resume continues exactly where the
last checkpoint stopped (RNG states restored -> deterministic continuation).
--max-time-sec (default 230 s) stops cleanly at an epoch boundary BEFORE the
sandbox 5-7 min kill, using the last epoch's measured duration as a lookahead.

Threading: torch.set_num_threads(1) — this 2-CPU box is shared with concurrent
agents; 1 thread is ~3x faster than 2 under contention and more deterministic.

Outputs:
  results/at_checkpoint.pt        (every epoch, resumable)
  results/at_train_report.json    (refreshed after every chunk; final when done)
  results/checkpoint_dual_at.pt   (only when the epoch target is reached;
                                   best-val-acc state, same format as
                                   checkpoint_dual.pt)
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

from waveforms import CLASS_NAMES, ATTACK_BANDS
import channels as CH
from receiver import FrontEnd, DualStreamModel, tf_cutmix, smoothed_ce
from attack_mask import cconv, waveform_pgd
from run_train import build_dataset

OUT = os.path.join(ROOT, "results")
CKPT = os.path.join(OUT, "at_checkpoint.pt")
FINAL = os.path.join(OUT, "checkpoint_dual_at.pt")
REPORT = os.path.join(OUT, "at_train_report.json")


# ---------------------------------------------------------------------------
def generate_adversarial_batch(model, frontend, wb, yb, np_rng,
                               steps, psr_lo, psr_hi, psd_margin, scenario):
    """Mask-matched AT example generation for one minibatch.

    Returns wb_adv = wb + cconv(delta, h_a) with delta from the SAME
    mask-constrained PGD as the attacks (band [0,10] MHz, psd_margin cap,
    untargeted), generated against the CURRENT model. delta is detached data.
    """
    B = wb.size(0)
    psr_db = np_rng.uniform(psr_lo, psr_hi, size=B)          # per-sample PSR
    p_rx = (wb.abs() ** 2).sum(dim=1)                        # window energy
    p_budget = p_rx * torch.tensor(10.0 ** (psr_db / 10.0),
                                   dtype=torch.float32)
    h_a = CH.draw_channel(scenario, np_rng)                  # fresh channel
    # BN in eval mode during generation (see module docstring); delta comes
    # back detached -> adversarial examples are data, not a grad path.
    model.eval()
    try:
        with torch.enable_grad():
            out = waveform_pgd(model, frontend, wb, h_a, yb, p_budget,
                               steps=steps,
                               band=ATTACK_BANDS["cv2x_attacker"],
                               targeted=False, psd_margin=psd_margin,
                               return_delta=True)
    finally:
        model.train()
    delta = out["delta"]                                     # detached
    wb_adv = wb + cconv(delta, torch.as_tensor(h_a, dtype=wb.dtype))
    return wb_adv


def train_epoch(model, frontend, opt, sched, tr_w, tr_y, cfg, np_rng):
    """One epoch of mask-matched AT (v3 recipe + adversarial minibatches)."""
    model.train()
    B = tr_w.size(0)
    perm = torch.randperm(B)
    tot_loss, n_adv_batches, n_batches = 0.0, 0, 0
    for bi, i in enumerate(range(0, B, cfg["batch_size"])):
        idx = perm[i:i + cfg["batch_size"]]
        wb, yb = tr_w[idx], tr_y[idx]
        # Gaussian waveform augmentation (as in standard training)
        if np_rng.random() < cfg["gauss_prob"]:
            wb = wb + cfg["gauss_std"] * torch.complex(
                torch.randn_like(wb.real), torch.randn_like(wb.imag))
        # mask-matched adversarial training branch
        if np_rng.random() < cfg["at_prob"]:
            scenario = "urban" if (bi % 2 == 0) else "highway"
            wb = generate_adversarial_batch(model, frontend, wb, yb, np_rng,
                                            cfg["steps"], cfg["psr_lo"],
                                            cfg["psr_hi"], cfg["psd_margin"],
                                            scenario)
            n_adv_batches += 1
        mag, ifr = frontend(wb)
        if np_rng.random() < cfg["cutmix_prob"]:
            mag, ifr, y_a, y_b, lam = tf_cutmix(mag, ifr, yb)
        else:
            y_a, y_b, lam = yb, yb, 1.0
        logits = model(mag, ifr)
        loss = smoothed_ce(logits, y_a, y_b, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        tot_loss += loss.item() * len(idx)
        n_batches += 1
    sched.step()
    return tot_loss / B, n_adv_batches, n_batches


def evaluate(model, frontend, val_w, val_y):
    model.eval()
    with torch.no_grad():
        preds = []
        for i in range(0, val_w.size(0), 256):
            lg = model.forward_wave(val_w[i:i + 256], frontend)
            preds.append(lg.argmax(1))
    return (torch.cat(preds) == val_y).float().mean().item()


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=16,
                    help="TOTAL epochs target (across all chunks)")
    ap.add_argument("--at-prob", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=5,
                    help="PGD steps for AT example generation")
    ap.add_argument("--psr-lo", type=float, default=-25.0)
    ap.add_argument("--psr-hi", type=float, default=-5.0)
    ap.add_argument("--psd-margin", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--cutmix-prob", type=float, default=0.3)
    ap.add_argument("--gauss-prob", type=float, default=0.3)
    ap.add_argument("--gauss-std", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true",
                    help="continue from results/at_checkpoint.pt")
    ap.add_argument("--max-time-sec", type=float, default=230.0,
                    help="stop cleanly at an epoch boundary before this")
    args = ap.parse_args()

    cfg = {"epochs_total": args.epochs, "at_prob": args.at_prob,
           "steps": args.steps, "psr_lo": args.psr_lo, "psr_hi": args.psr_hi,
           "psd_margin": args.psd_margin, "batch_size": args.batch_size,
           "lr": args.lr, "cutmix_prob": args.cutmix_prob,
           "gauss_prob": args.gauss_prob, "gauss_std": args.gauss_std,
           "seed": args.seed}

    t_chunk = time.time()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed + 1000)

    # ---- data + frontend stats (deterministic, identical to run_train.py) ----
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

    # ---- resume ----
    if os.path.exists(CKPT):
        if not args.resume:
            print("ERROR: results/at_checkpoint.pt exists — pass --resume "
                  "to continue it (or delete it to start over).", flush=True)
            sys.exit(2)
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        old = ck["config"]
        mism = {k: (old.get(k), cfg[k]) for k in cfg
                if k not in ("epochs_total",) and old.get(k) != cfg[k]}
        if mism:
            print(f"ERROR: config mismatch vs checkpoint: {mism}", flush=True)
            sys.exit(2)
        if old.get("epochs_total", 0) > args.epochs:
            print("ERROR: --epochs lower than checkpoint's total target.",
                  flush=True)
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

    # ---- training loop (time-guarded, checkpoint every epoch) ----
    ep = start_epoch
    last_dur = history[-1]["epoch_s"] if history else None
    while ep < args.epochs:
        elapsed = time.time() - t_chunk
        look = last_dur if last_dur is not None else 90.0
        if elapsed + look > args.max_time_sec:
            print(f"[guard] stopping before epoch {ep+1}: elapsed "
                  f"{elapsed:.0f}s + lookahead {look:.0f}s > "
                  f"{args.max_time_sec:.0f}s (checkpoint is current)",
                  flush=True)
            break
        t_ep = time.time()
        loss, n_adv, n_b = train_epoch(model, frontend, opt, sched,
                                       tr_w, tr_y, cfg, np_rng)
        acc = evaluate(model, frontend, val_w, val_y)
        if acc > best_acc + 1e-4:
            best_acc, best_state = acc, \
                {k: v.clone() for k, v in model.state_dict().items()}
        ep += 1
        last_dur = time.time() - t_ep
        rec = {"epoch": ep, "loss": round(loss, 4),
               "val_acc": round(acc, 4), "n_adv_batches": n_adv,
               "n_batches": n_b, "epoch_s": round(last_dur, 1)}
        history.append(rec)
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
              f"AT-batches {n_adv}/{n_b}  ({last_dur:.0f}s)", flush=True)

    # ---- report (refreshed every chunk; complete when target reached) ----
    completed = ep >= args.epochs
    report = {
        "experiment": "mask-matched adversarial training (standard AT, "
                      "matched to the compliant-attacker threat model)",
        "config": cfg,
        "recipe": "v3 recipe from run_train.py (Adam 5e-4, cosine, label "
                  "smoothing 0.1, TF-CutMix 0.3, Gaussian aug 0.3/0.02) "
                  "+ adversarial minibatches per cfg",
        "at_scheme": {
            "generator": "waveform_pgd against the CURRENT training model "
                         "(standard AT, not TRADES)",
            "band": ATTACK_BANDS["cv2x_attacker"],
            "psd_margin": cfg["psd_margin"],
            "targeted": False,
            "psr_train_db": [cfg["psr_lo"], cfg["psr_hi"]],
            "channel": "fresh draw per AT batch, urban/highway alternating "
                       "by batch index",
            "budget_reference": "per-sample window energy of the clean "
                                "(augmented) training waveform",
            "model_mode_during_generation": "eval (BN running stats, dropout "
                                            "off); train mode for loss step",
        },
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
                    "config": {"seed": args.seed,
                               "n_train": tr_w.size(0),
                               "n_val": val_w.size(0),
                               "epochs": args.epochs,
                               "at_prob": args.at_prob,
                               "at_steps": args.steps,
                               "at_psr_db": [args.psr_lo, args.psr_hi],
                               "psd_margin": args.psd_margin}},
                   FINAL)
        print(f"TRAINING COMPLETE: best val acc {best_acc*100:.2f}%  "
              f"-> {FINAL}", flush=True)
    else:
        print(f"CHUNK DONE at epoch {ep}/{args.epochs} — re-run with "
              f"--resume (total training time so far {elapsed_total:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
