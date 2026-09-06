#!/usr/bin/env python3
"""run_adaptive_attack.py — ADAPTIVE-ATTACK evaluation of the defenses
(P0 item from the 2026-09-06 research council, reviewers 21-a and 21-d).

A defense evaluated only against PGD-10 zero-init is not certified against an
adaptive adversary. This script attacks the defended checkpoints with the
STRONGEST attack in our arsenal: PGD-50 with R random restarts
(best-of-R per sample, src.attack_mask.waveform_pgd_restarts), under the
EXACT baseline protocol (urban, untargeted, attack seed 7, 100 active
samples/class = 300 total, alpha_frac 0.25, psd_margin 2.0, budgets matched
to clean received-signal window energy; eval set identical to
attack_results_merged.json via run_attack.build_eval_set).

Restart 0 of every cell is the deterministic zero init, so the R=1 run is
numerically identical to the canonical single-shot PGD — continuity check.

Targets (--defense):
  at     : results/checkpoint_dual_at.pt     (mask-matched AT,  C16)
  trades : results/checkpoint_dual_trades.pt (TRADES,            C17)
  dual   : results/checkpoint_dual.pt        (undefended, control)

Chunkable/idempotent: results/adaptive_{defense}_s{seed}_r{restarts}.json is
merged after every (setting, psr) cell; computed cells are skipped on relaunch
(--force recomputes). --max-time-sec stops cleanly BETWEEN cells so the script
survives short sandbox process limits; the driver loop re-invokes until DONE.

Suggested full run (2-CPU box, ~150 s per R=10/S=50 cell):
  python3 scripts/run_adaptive_attack.py --defense at --steps 50 --restarts 10
  (repeat until COMPLETE) then --defense trades, then --defense dual.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(2)

from waveforms import ATTACK_BANDS, NOISE_CLASS
from receiver import FrontEnd, DualStreamModel
from attack_mask import (cond_asr, meap_curve, price_of_compliance,
                         waveform_pgd_restarts)
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")
CKPTS = {
    "at": ("checkpoint_dual_at.pt", "mask-matched adversarial training"),
    "trades": ("checkpoint_dual_trades.pt", "TRADES"),
    "dual": ("checkpoint_dual.pt", "undefended control"),
}

DEFAULT_PSR = {
    "at": [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0],
    "trades": [-35.0, -30.0, -25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0],
    "dual": [-40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0],
}


def res_path(defense, seed, restarts):
    return os.path.join(OUT, f"adaptive_{defense}_s{seed}_r{restarts}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--defense", required=True, choices=list(CKPTS))
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--psr", nargs="+", type=float, default=None)
    ap.add_argument("--scenario", default="urban")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--psd-margin", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--settings", nargs="+",
                    default=["genie", "cv2x_mask"],
                    choices=["genie", "cv2x_mask"],
                    help="restrict this invocation to a subset of settings "
                         "(the two settings can use different PSR grids)")
    ap.add_argument("--max-time-sec", type=float, default=420.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    psr_grid = args.psr if args.psr else DEFAULT_PSR[args.defense]

    # ---- load defense ----
    ck_name, desc = CKPTS[args.defense]
    ck = torch.load(os.path.join(OUT, ck_name), map_location="cpu",
                    weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])
    print(f"[adaptive] {ck_name} ({desc}) loaded", flush=True)

    # ---- eval set (identical protocol to baseline runs) ----
    x_rx, y, h_a, p_clean = build_eval_set(args.scenario, args.n_eval,
                                           args.seed)
    with torch.no_grad():
        clean_preds = []
        for i in range(0, x_rx.size(0), 256):
            lg = model.forward_wave(x_rx[i:i + 256], frontend)
            clean_preds.append(lg.argmax(1))
        clean_preds = torch.cat(clean_preds)
    clean_acc = (clean_preds == y).float().mean().item()
    print(f"[adaptive] clean acc on active tx: {clean_acc*100:.2f}%",
          flush=True)

    fname = res_path(args.defense, args.seed, args.restarts)
    if os.path.exists(fname) and not args.force:
        results = json.load(open(fname))
    else:
        results = {
            "experiment": "Adaptive attack (PGD-{steps} x {restarts} restarts, "
                          "best-of-R per sample) on {desc}".format(
                              steps=args.steps, restarts=args.restarts,
                              desc=desc),
            "config": {
                "defense": args.defense, "checkpoint": ck_name,
                "scenario": args.scenario, "mode": "untargeted",
                "steps": args.steps, "restarts": args.restarts,
                "psr_db": psr_grid, "n_eval_per_class": args.n_eval,
                "seed": args.seed, "attack_pgd_alpha_frac": args.alpha,
                "psd_cap_margin": args.psd_margin,
                "restart_init": "r0 zero-init (continuity) + "
                                "CN random inits, frac U[0.2,1.0], projected; "
                                "rng stream default_rng(seed*7919+r)",
                "best_of": "per-sample max attack objective "
                           "(CE to true label, untargeted)",
                "psr_reference": "clean received-signal window energy "
                                 "(sum |r|^2, pre-noise)",
                "mask_band_mhz": ATTACK_BANDS["cv2x_attacker"],
                "genie": "power-only, no PSD cap (corrected 2026-09-05)",
            },
            "clean_acc_active": round(clean_acc, 4),
            "runs": {},
            "summary": {},
        }
    results["config"]["psr_db"] = psr_grid
    results["clean_acc_active"] = round(clean_acc, 4)

    band = ATTACK_BANDS["cv2x_attacker"]
    todo = [(setting, p) for setting in args.settings
            for p in psr_grid]
    done = 0
    for setting, psr in todo:
        key = f"untargeted/{setting}"
        pk = f"psr={psr:+.0f}dB"
        if key in results["runs"] and pk in results["runs"][key] \
                and not args.force:
            continue
        if time.time() - t0 > args.max_time_sec and done > 0:
            print(f"[adaptive] time budget reached — flushing and exiting "
                  f"(re-invoke to continue)", flush=True)
            break

        sband = None if setting == "genie" else band
        p_budget = p_clean * (10 ** (psr / 10.0))
        print(f"    [cell] {key} {pk}  (steps {args.steps} x "
              f"{args.restarts} restarts)", flush=True)
        out = waveform_pgd_restarts(
            model, frontend, x_rx, h_a, y, p_budget,
            steps=args.steps, restarts=args.restarts,
            base_seed=args.seed, band=sband, targeted=False,
            alpha_frac=args.alpha, psd_margin=args.psd_margin, chunk=300)
        c_asr, n_elig = cond_asr(clean_preds, out["preds"], y,
                                 targeted=False)
        rob = (out["preds"] == y).float().mean().item()
        wr = out["win_restart"].tolist()
        n_r0 = sum(1 for w in wr if w == 0)
        results["runs"].setdefault(key, {})[pk] = {
            "cond_asr": round(100 * c_asr, 2),
            "robust_acc": round(100 * rob, 2),
            "n_eligible": n_elig,
            "win_restart_hist": {str(r): wr.count(r)
                                 for r in sorted(set(wr))},
            "frac_zero_init_wins": round(n_r0 / max(n_elig, 1), 4),
        }
        results.setdefault("elapsed_s", 0)
        results["elapsed_s"] = round(results["elapsed_s"] +
                                     (time.time() - t0), 1)
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"    [cell] {key} {pk}: cond-ASR {100*c_asr:5.2f}%  "
              f"robust {100*rob:5.2f}%  zero-init wins "
              f"{n_r0}/{n_elig}", flush=True)
        done += 1
        t0 = time.time()          # per-cell timer for the budget check

    # ---- recompute summaries over ALL stored cells (union of grids) ----
    def stored_psrs(setting):
        cells = results["runs"].get(f"untargeted/{setting}", {})
        out = []
        for k in cells:
            try:
                out.append(float(k[4:-2]))
            except ValueError:
                pass
        return sorted(out)

    have = {s: stored_psrs(s) for s in ("genie", "cv2x_mask")}
    if have["genie"] and have["cv2x_mask"]:
        g = [results["runs"]["untargeted/genie"][f"psr={p:+.0f}dB"]["cond_asr"]
             for p in have["genie"]]
        m = [results["runs"]["untargeted/cv2x_mask"][f"psr={p:+.0f}dB"]
             ["cond_asr"] for p in have["cv2x_mask"]]
        mg, cg = meap_curve(have["genie"], g, 20.0)
        mm, cm = meap_curve(have["cv2x_mask"], m, 20.0)
        poc, cnotes = price_of_compliance(have["genie"], g,
                                          have["cv2x_mask"], m, 20.0)
        results["summary"]["untargeted"] = {
            "meap_genie_db": mg, "meap_genie_censor": cg,
            "meap_cv2x_mask_db": mm, "meap_cv2x_mask_censor": cm,
            "price_of_compliance_db": poc, "poc_censor": cnotes,
            "genie_curve": g, "mask_curve": m,
            "psr_genie": have["genie"], "psr_mask": have["cv2x_mask"],
        }
        with open(fname, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[adaptive] MEAP genie {mg} ({cg})  mask {mm} ({cm})  "
              f"PoC {poc} dB {cnotes or ''}", flush=True)

    n_cells = sum(len(v) for v in results["runs"].values())
    n_need = len(args.settings) * len(psr_grid)
    if n_cells >= n_need:
        print("COMPLETE", flush=True)
    else:
        print(f"PROGRESS {n_cells}/{n_need} cells — re-invoke to continue",
              flush=True)


if __name__ == "__main__":
    main()
