#!/usr/bin/env python3
"""w14b_adaptive_capture.py — Wave 14 / 35-b-11 (council 34-d iii): per-window
outcome capture for the CANONICAL ADAPTIVE protocol (PGD-50 x R=5, attack
seed 7, defenses dual/at/trades), extending the paired-bootstrap CIs to the
adaptive grids with n=3 attack seeds per defense (7 captured here; 11/22
were captured during the item-12 grind).

Re-runs the canonical adaptive cells EXACTLY as run_adaptive_attack.py does
(same imports, same eval set via build_eval_set("urban", 100, 7), same
restart stream default_rng(seed*7919+r) inside waveform_pgd_restarts) and
stores, per cell, the win_flags vector (1 = eligible + success, 0 =
eligible + fail, -1 = clean-wrong). The attack is deterministic given the
seed, so the recomputed cond_asr must MATCH the stored canonical cells —
asserted as a continuity cross-check (tolerance: exact on 2dp percentages).

The canonical adaptive_*_s7_r5.json files are NOT modified: the capture goes
to separate files. (run_adaptive_attack.py --force rebuilds its state file
from scratch — it clobbers untouched arms — so the separate-file design is
required; driver quirk documented in the Wave-14 worklog.)

Per-arm PSR grids are read from the canonical file itself, so the capture
mirrors each defense's protocol exactly:
  dual:   genie [-50..-30]      cv2x_mask [-45..-20]
  at:     genie [-45..-25]      cv2x_mask [-35..-10]
  trades: genie [-45..-25]      cv2x_mask [-35..-5]

Output: results/w14b_perwindow_adaptive_{defense}_s7.json
Chunked/idempotent: computed cells skipped; per-cell atomic writes
(tmp + os.replace); --max-time-sec stops cleanly BETWEEN cells.
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

from waveforms import ATTACK_BANDS
from receiver import FrontEnd, DualStreamModel
from attack_mask import cond_asr, waveform_pgd_restarts
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")
CKPTS = {
    "at": "checkpoint_dual_at.pt",
    "trades": "checkpoint_dual_trades.pt",
    "dual": "checkpoint_dual.pt",
}
ATTACK_SEED = 7
R = 5
STEPS = 50


def atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def fmt_psr(p):
    return f"psr={p:+.0f}dB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--defenses", nargs="+",
                    default=["dual", "at", "trades"],
                    choices=list(CKPTS))
    ap.add_argument("--max-time-sec", type=float, default=520.0)
    args = ap.parse_args()

    t0 = time.time()

    for defense in args.defenses:
        out_path = os.path.join(
            OUT, f"w14b_perwindow_adaptive_{defense}_s{ATTACK_SEED}.json")
        ref_path = os.path.join(OUT, f"adaptive_{defense}_s{ATTACK_SEED}_r{R}.json")
        ref = json.load(open(ref_path))
        ref_runs = ref["runs"]

        res = json.load(open(out_path)) if os.path.exists(out_path) else {
            "experiment": "per-window outcome capture for the canonical "
                          "ADAPTIVE protocol (PGD-50 x R=5, attack seed 7; "
                          "35-b-11 / item-12 follow-through)",
            "config": {
                "defense": defense, "checkpoint": CKPTS[defense],
                "scenario": "urban", "mode": "untargeted", "steps": STEPS,
                "restarts": R, "attack_seed": ATTACK_SEED,
                "n_eval_per_class": 100, "attack_pgd_alpha_frac": 0.25,
                "psd_cap_margin": 2.0,
                "attack_call": "identical to run_adaptive_attack.py "
                               "(waveform_pgd_restarts, base_seed=7, "
                               "chunk=300, full 300-window tensors); "
                               "canonical file untouched",
                "win_flags_encoding": "1 = eligible + attack success; "
                                      "0 = eligible + fail; -1 = clean-wrong",
            },
            "clean_acc_active": None,
            "runs": {},
            "checks": {},
        }

        ck = torch.load(os.path.join(OUT, CKPTS[defense]),
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
        clean_acc = (clean_preds == y).float().mean().item()
        res["clean_acc_active"] = round(clean_acc, 4)
        res["checks"]["clean_acc_matches_ref"] = bool(
            abs((ref.get("clean_acc_active") or 0) - round(clean_acc, 4))
            < 1e-9)
        atomic_dump(res, out_path)

        for setting in ("genie", "cv2x_mask"):
            key = "untargeted/" + setting
            res["runs"].setdefault(key, {})
            psr_grid = sorted(float(k[4:-2]) for k in ref_runs[key])
            sband = None if setting == "genie" else \
                ATTACK_BANDS["cv2x_attacker"]
            for psr in psr_grid:
                if fmt_psr(psr) in res["runs"][key]:
                    continue
                if time.time() - t0 > args.max_time_sec:
                    print(f"[{defense}] time budget — flush and exit "
                          f"(re-invoke to continue)", flush=True)
                    atomic_dump(res, out_path)
                    return
                p_budget = p_clean * (10 ** (psr / 10.0))
                out = waveform_pgd_restarts(
                    model, frontend, x_rx, h_a, y, p_budget,
                    steps=STEPS, restarts=R, base_seed=ATTACK_SEED,
                    band=sband, targeted=False, alpha_frac=0.25,
                    psd_margin=2.0, chunk=300)
                c_asr, n_elig = cond_asr(clean_preds, out["preds"], y,
                                         targeted=False)
                elig = (clean_preds == y)
                win = ((out["preds"] != y) & elig).long()
                flags = torch.where(elig, win, torch.full_like(win, -1))
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
                atomic_dump(res, out_path)
                print(f"[{defense}] {key} {fmt_psr(psr)}: cond-ASR "
                      f"{100*c_asr:5.2f}%  stored-match {match}", flush=True)

        n_cells = sum(len(v) for v in res["runs"].values())
        res["checks"]["continuity_all_match"] = all(
            c["matches_stored_canonical"]
            for v in res["runs"].values() for c in v.values())
        res["checks"]["n_cells"] = n_cells
        atomic_dump(res, out_path)
        print(f"[{defense}] COMPLETE {n_cells} cells, continuity "
              f"{res['checks']['continuity_all_match']}", flush=True)

    print("ALL ADAPTIVE CAPTURE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
