#!/usr/bin/env python3
"""w14b_canonical_capture.py — Wave 14 / 35-b-11 (council 34-d iii): per-window
outcome capture for the CANONICAL PGD-10 protocol, enabling paired-bootstrap
MEAP/PoC CIs (replaces conservative-width CI propagation for the headline).

Re-runs the canonical attack cells (urban, untargeted, PGD-10, zero-init,
alpha 0.25, psd_margin 2.0, budgets = clean window energy, SAME eval sets via
build_eval_set with seeds 7/123/456) and stores, per cell, the per-window
outcome vector win_flags (1 = eligible + attack success, 0 = eligible + fail,
-1 = clean-wrong). The attack is deterministic given the seed, so the
recomputed cond_asr must MATCH the stored canonical cells — asserted as a
continuity check (tolerance: exact on 2dp percentages).

Output: results/w14b_perwindow_s{seed}.json (one per canonical seed).
Chunked/idempotent: computed cells skipped; --max-time-sec stops between cells.
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
from attack_mask import cond_asr, waveform_pgd
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")
SEEDS = [7, 123, 456]
PSR_GRID = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0,
            -5.0, 0.0, 5.0, 10.0]
# the 3-seed protocol is MODEL-seed replication: seed 123/456 runs used the
# retrained checkpoints (checkpoint_dual_s123/456.pt) with the SAME attack
# eval seed 7 (stored config "seed": 7 confirms); seed 7 uses the canonical
# checkpoint_dual.pt. (w14b bug fixed: eval-set seed must be 7 for all.)
CKPT_OF = {7: "checkpoint_dual.pt", 123: "checkpoint_dual_s123.pt",
           456: "checkpoint_dual_s456.pt"}
ATTACK_SEED = 7
CANONICAL = {                      # stored reference files for cross-check
    7: "attack_results_merged.json",
    123: "attack_results_urban_untargeted_s123.json",
    456: "attack_results_urban_untargeted_s456.json",
}


def fmt_psr(p):
    return f"psr={p:+.0f}dB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--max-time-sec", type=float, default=520.0)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(7)

    for seed in args.seeds:
        out_path = os.path.join(OUT, f"w14b_perwindow_s{seed}.json")
        res = json.load(open(out_path)) if os.path.exists(out_path) else {
            "experiment": "per-window outcome capture for the canonical "
                          "PGD-10 urban/untargeted protocol (35-b-11)",
            "config": {
                "scenario": "urban", "mode": "untargeted", "steps": 10,
                "model_seed": seed, "attack_eval_seed": ATTACK_SEED,
                "checkpoint": CKPT_OF[seed],
                "n_eval_per_class": 100,
                "attack_pgd_alpha_frac": 0.25, "psd_cap_margin": 2.0,
                "psr_db": list(PSR_GRID),
                "protocol": "identical to attack_results_merged.json (seed 7) "
                            "/ _s123 / _s456 — deterministic re-run with "
                            "per-window flags; the 3-seed protocol is MODEL-"
                            "seed replication (retrained checkpoints), the "
                            "attack eval seed is 7 for all",
                "win_flags_encoding": "1 = eligible + attack success; "
                                      "0 = eligible + fail; -1 = clean-wrong",
            },
            "clean_acc_active": None,
            "runs": {},
            "checks": {},
        }

        ck = torch.load(os.path.join(OUT, CKPT_OF[seed]),
                        map_location="cpu", weights_only=False)
        model = DualStreamModel()
        model.load_state_dict(ck["model"])
        model.eval()
        frontend = FrontEnd()
        frontend.set_stats(ck["mag_mean"], ck["mag_std"])

        x_rx, y, h_a, p_clean = build_eval_set("urban", 100, ATTACK_SEED)
        with torch.no_grad():
            cp = []
            for i in range(0, x_rx.size(0), 256):
                cp.append(model.forward_wave(x_rx[i:i + 256],
                                             frontend).argmax(1))
            clean_preds = torch.cat(cp)
        res["clean_acc_active"] = round(
            (clean_preds == y).float().mean().item(), 4)

        # stored canonical reference (for the continuity cross-check)
        ref = json.load(open(os.path.join(OUT, CANONICAL[seed])))
        ref_runs = ref["per_scenario"]["urban"]["runs"]

        for setting in ("genie", "cv2x_mask"):
            key = "untargeted/" + setting
            res["runs"].setdefault(key, {})
            sband = None if setting == "genie" else \
                ATTACK_BANDS["cv2x_attacker"]
            for psr in PSR_GRID:
                if fmt_psr(psr) in res["runs"][key]:
                    continue
                if time.time() - t0 > args.max_time_sec:
                    print(f"[s{seed}] time budget — flush and exit",
                          flush=True)
                    json.dump(res, open(out_path, "w"), indent=2)
                    return
                p_budget = p_clean * (10 ** (psr / 10.0))
                preds = []
                for i in range(0, x_rx.size(0), 300):
                    out = waveform_pgd(model, frontend, x_rx[i:i + 300],
                                       h_a[i:i + 300], y[i:i + 300],
                                       p_budget[i:i + 300], steps=10,
                                       band=sband, targeted=False,
                                       target_class=NOISE_CLASS)
                    preds.append(out["preds"])
                adv = torch.cat(preds)
                c_asr, n_elig = cond_asr(clean_preds, adv, y,
                                         targeted=False)
                elig = (clean_preds == y)
                win = ((adv != y) & elig).long()
                flags = torch.where(elig, win,
                                    torch.full_like(win, -1))
                # continuity cross-check vs the stored canonical cell
                ref_cell = ref_runs[key].get(fmt_psr(psr))
                match = (ref_cell is None or
                         abs(ref_cell["cond_asr"] -
                             round(100 * c_asr, 2)) < 0.011)
                res["runs"][key][fmt_psr(psr)] = {
                    "cond_asr": round(100 * c_asr, 2),
                    "n_eligible": n_elig,
                    "win_flags": [int(v) for v in flags.tolist()],
                    "matches_stored_canonical": bool(match),
                }
                json.dump(res, open(out_path, "w"), indent=2)
                print(f"[s{seed}] {key} {fmt_psr(psr)}: cond-ASR "
                      f"{100*c_asr:5.2f}%  stored-match {match}", flush=True)

        n_cells = sum(len(v) for v in res["runs"].values())
        res["checks"]["continuity_all_match"] = all(
            c["matches_stored_canonical"]
            for v in res["runs"].values() for c in v.values())
        json.dump(res, open(out_path, "w"), indent=2)
        print(f"[s{seed}] COMPLETE {n_cells} cells, continuity "
              f"{res['checks']['continuity_all_match']}", flush=True)

    print("ALL CAPTURE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
