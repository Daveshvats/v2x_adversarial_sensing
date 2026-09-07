#!/usr/bin/env python3
"""run_at_eval_seeds.py — Wave 14 / 35-b-13 (council 34-d: the naive-vs-adaptive
gap is only documented on defense seed 42).

Runs the EXACT baseline PGD-10 evaluation protocol (identical to
run_at_eval.py / run_trades.py protocol: urban, untargeted, attack seed 7,
100 samples/class, PGD-10, alpha 0.25, psd_margin 2.0, PSR grid -45..0,
genie + cv2x_mask, budgets = clean received-signal window energy) on the
defense-seed-replication checkpoints s43/s53, so the paper's defense-seed
table shows the PGD-10 (non-adaptive) MEAPs for every defense seed next to
the adaptive PGD-50xR5 MEAPs (w13), and the naive-vs-adaptive margin gap is
disclosed per seed, not only for s42.

Arms (all four, chunked + idempotent — one JSON per arm, per-cell flush):
  at_s43     results/checkpoint_dual_at_s43.pt     -> at_defense_results_s43.json
  at_s53     results/checkpoint_dual_at_s53.pt     -> at_defense_results_s53.json
  trades_s43 results/checkpoint_dual_trades_s43.pt -> trades_defense_results_s43.json
  trades_s53 results/checkpoint_dual_trades_s53.pt -> trades_defense_results_s53.json

Usage: python3 scripts/run_at_eval_seeds.py [--arms at_s43 at_s53 trades_s43
trades_s53] [--max-time-sec 520]   (re-invoke until ALL ARMS COMPLETE)
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
from attack_mask import cond_asr, meap_curve, price_of_compliance
from run_attack import build_eval_set, run_setting

OUT = os.path.join(ROOT, "results")
DEFAULT_PSR = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
               -10.0, -5.0, 0.0]
ARMS = {
    "at_s43": ("checkpoint_dual_at_s43.pt", "at_defense_results_s43.json",
               "mask-matched AT, defense seed 43"),
    "at_s53": ("checkpoint_dual_at_s53.pt", "at_defense_results_s53.json",
               "mask-matched AT, defense seed 53"),
    "trades_s43": ("checkpoint_dual_trades_s43.pt",
                   "trades_defense_results_s43.json",
                   "TRADES, defense seed 43"),
    "trades_s53": ("checkpoint_dual_trades_s53.pt",
                   "trades_defense_results_s53.json",
                   "TRADES, defense seed 53"),
}


def fmt_psr(p):
    return f"psr={p:+.0f}dB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arms", nargs="+", default=list(ARMS),
                    choices=list(ARMS))
    ap.add_argument("--psr", nargs="+", type=float, default=None)
    ap.add_argument("--max-time-sec", type=float, default=520.0)
    args = ap.parse_args()

    t0 = time.time()
    psr_grid = args.psr if args.psr else DEFAULT_PSR

    for arm in args.arms:
        ck_name, res_name, desc = ARMS[arm]
        res_path = os.path.join(OUT, res_name)
        ck = torch.load(os.path.join(OUT, ck_name), map_location="cpu",
                        weights_only=False)
        model = DualStreamModel()
        model.load_state_dict(ck["model"])
        model.eval()
        frontend = FrontEnd()
        frontend.set_stats(ck["mag_mean"], ck["mag_std"])
        print(f"[{arm}] {ck_name} loaded ({desc})", flush=True)

        x_rx, y, h_a, p_clean = build_eval_set("urban", 100, 7)
        with torch.no_grad():
            clean_preds = []
            for i in range(0, x_rx.size(0), 256):
                lg = model.forward_wave(x_rx[i:i + 256], frontend)
                clean_preds.append(lg.argmax(1))
            clean_preds = torch.cat(clean_preds)
        clean_acc = (clean_preds == y).float().mean().item()
        print(f"[{arm}] clean acc on active eval set: {clean_acc*100:.2f}%",
              flush=True)

        # resume
        res = None
        if os.path.exists(res_path):
            res = json.load(open(res_path))
        if res is None or "runs" not in res:
            res = {
                "experiment": f"PGD-10 baseline attack on {desc} "
                              "(naive-vs-adaptive gap per defense seed, "
                              "35-b-13)",
                "config": {
                    "scenario": "urban", "mode": "untargeted", "steps": 10,
                    "seed": 7, "n_eval_per_class": 100,
                    "attack_pgd_alpha_frac": 0.25, "psd_cap_margin": 2.0,
                    "psr_db": list(psr_grid),
                    "psr_reference": "clean received-signal window energy "
                                     "(pre-noise)",
                    "mask_band_mhz": ATTACK_BANDS["cv2x_attacker"],
                    "genie": "power-only, no PSD cap",
                    "protocol": "identical to run_at_eval.py / at_defense_"
                                "results.json (s42) — same eval set, same "
                                "attack seed, same grids",
                },
                "checkpoint": ck_name,
                "clean_acc_active": round(clean_acc, 4),
                "runs": {},
                "summary": {},
            }
        res["config"]["psr_db"] = list(psr_grid)
        res["clean_acc_active"] = round(clean_acc, 4)

        for setting in ("genie", "cv2x_mask"):
            key = "untargeted/" + setting
            res["runs"].setdefault(key, {})
            sband = None if setting == "genie" else \
                ATTACK_BANDS["cv2x_attacker"]
            for psr in psr_grid:
                if fmt_psr(psr) in res["runs"][key]:
                    continue
                if time.time() - t0 > args.max_time_sec:
                    print(f"[{arm}] time budget — flush and exit "
                          "(re-invoke)", flush=True)
                    json.dump(res, open(res_path, "w"), indent=2)
                    return
                p_budget = p_clean * (10 ** (psr / 10.0))
                adv_preds = run_setting(model, frontend, x_rx, h_a, y,
                                        p_budget, sband, False, 10, 2.0)
                c_asr, n_elig = cond_asr(clean_preds, adv_preds, y,
                                         targeted=False)
                rob = (adv_preds == y).float().mean().item()
                res["runs"][key][fmt_psr(psr)] = {
                    "cond_asr": round(100 * c_asr, 2),
                    "robust_acc": round(100 * rob, 2),
                    "n_eligible": n_elig,
                }
                json.dump(res, open(res_path, "w"), indent=2)
                print(f"  [{arm}] {key} {fmt_psr(psr)}: cond-ASR "
                      f"{100*c_asr:5.1f}%  robust {100*rob:5.1f}%",
                      flush=True)

        # summaries
        def stored_psrs(setting):
            cells = res["runs"]["untargeted/" + setting]
            return sorted(float(k[4:-2]) for k in cells)

        have_g, have_m = stored_psrs("genie"), stored_psrs("cv2x_mask")
        if have_g and have_m:
            g = [res["runs"]["untargeted/genie"][fmt_psr(p)]["cond_asr"]
                 for p in have_g]
            m = [res["runs"]["untargeted/cv2x_mask"][fmt_psr(p)]
                 ["cond_asr"] for p in have_m]
            mg, cg = meap_curve(have_g, g, 20.0)
            mm, cm = meap_curve(have_m, m, 20.0)
            poc, cnotes = price_of_compliance(have_g, g, have_m, m, 20.0)
            res["summary"]["untargeted"] = {
                "meap_genie_db": mg, "meap_genie_censor": cg,
                "meap_cv2x_mask_db": mm, "meap_cv2x_mask_censor": cm,
                "price_of_compliance_db": poc, "poc_censor": cnotes,
                "genie_curve": g, "mask_curve": m,
                "psr_genie": have_g, "psr_mask": have_m,
            }
            json.dump(res, open(res_path, "w"), indent=2)
            print(f"[{arm}] MEAP genie {mg} ({cg})  mask {mm} ({cm})  "
                  f"PoC {poc} dB", flush=True)

        n_cells = sum(len(v) for v in res["runs"].values())
        if n_cells < 2 * len(psr_grid):
            print(f"[{arm}] PROGRESS {n_cells}/{2*len(psr_grid)} cells",
                  flush=True)
        else:
            print(f"[{arm}] COMPLETE", flush=True)

    print("ALL ARMS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
