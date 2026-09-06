#!/usr/bin/env python3
"""run_at_eval.py — evaluate the mask-matched-AT model under the EXACT baseline
attack protocol (CLAIMS C16 evidence).

Protocol (identical to the undefended runs behind attack_results_merged.json):
  urban scenario, untargeted (plus optional targeted_noise), attack seed 7,
  100 samples/class of ACTIVE transmissions (300 total), PGD-10,
  alpha_frac 0.25, psd_margin 2.0, PSR grid -45..0 dB for BOTH genie
  (power-only) and cv2x_mask (emission-mask-compliant) attackers, budgets
  matched to the clean received-signal window energy (build_eval_set /
  run_setting imported from run_attack.py). MEAP (20% cond-ASR threshold,
  censoring-aware) and Price-of-Compliance are computed with attack_mask
  metrics, then compared against the UNDEFENDED reference numbers hardcoded
  below (they come from results/attack_results_merged.json — urban block).

Also reported: clean accuracy of the defended model on the attack eval set
(active classes) and on the standard validation split (build_dataset(42,1000),
80/20) — the clean-accuracy COST of the defense.

Chunkable/idempotent: results/at_defense_results.json is merged after every
invocation; already-computed (mode, setting, psr) entries are skipped unless
--force. --max-time-sec stops cleanly between attacks (sandbox 5-7 min kill).

Suggested invocation plan (single-threaded, ~25-30 s per 300-sample PGD-10
attack on this shared 2-CPU box):
  run 1: --modes untargeted --settings genie
  run 2: --modes untargeted --settings cv2x_mask
  run 3: --modes targeted_noise --settings genie cv2x_mask --psr -20 -10 0
         (recomputes all summaries + deltas and finalizes the JSON)

Threading: torch.set_num_threads(1) — 2-CPU box shared with concurrent agents.
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

from waveforms import ATTACK_BANDS, NOISE_CLASS, CLASS_NAMES
from receiver import FrontEnd, DualStreamModel
from attack_mask import cond_asr, meap_curve, price_of_compliance
from run_attack import build_eval_set, run_setting
from run_train import build_dataset

OUT = os.path.join(ROOT, "results")
CKPT_AT = os.path.join(OUT, "checkpoint_dual_at.pt")
RES = os.path.join(OUT, "at_defense_results.json")

DEFAULT_PSR = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
               -10.0, -5.0, 0.0]
TARGETED_PSR = [-20.0, -10.0, 0.0]

# ---------------------------------------------------------------------------
# UNDEFENDED reference numbers — VERBATIM from results/attack_results_merged.json
# (urban block; config: PGD-10, seed 7, 100/class, psd_margin 2.0, threshold 20%)
# and results/train_report.json (clean val acc, seed 42). Do not edit: these are
# the baseline the defense must be compared against.
# ---------------------------------------------------------------------------
UNDEFENDED = {
    "source": "results/attack_results_merged.json (urban) + results/train_report.json",
    "clean_acc_val": 1.0,
    "clean_acc_active_urban": 1.0,
    "untargeted": {
        "meap_genie_db": -44.22475707646814,
        "meap_genie_censor": None,
        "meap_cv2x_mask_db": -37.08333333333358,
        "meap_cv2x_mask_censor": None,
        "price_of_compliance_db": 7.14142374313456,
        "asr_at_minus10_cv2x_mask": 96.67,
        "asr_at_minus10_genie": 100.0,
    },
    "targeted_noise": {
        "meap_genie_db": -30.8340909090911,
        "meap_genie_censor": None,
        "meap_cv2x_mask_db": -14.823684210526325,
        "meap_cv2x_mask_censor": None,
        "price_of_compliance_db": 16.010406698564772,
        "asr_at_minus10_cv2x_mask": 38.33,
        "asr_at_minus10_genie": 100.0,
    },
}


def fmt_psr(p):
    return f"psr={p:+.0f}dB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modes", nargs="+", default=["untargeted"],
                    choices=["untargeted", "targeted_noise"])
    ap.add_argument("--settings", nargs="+", default=["genie", "cv2x_mask"],
                    choices=["genie", "cv2x_mask"])
    ap.add_argument("--psr", nargs="+", type=float, default=None,
                    help="PSR grid for this invocation (default: full "
                         "10-point grid for untargeted, {-20,-10,0} for "
                         "targeted_noise)")
    ap.add_argument("--scenario", default="urban")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--psd-margin", type=float, default=2.0)
    ap.add_argument("--force", action="store_true",
                    help="recompute entries even if already stored")
    ap.add_argument("--max-time-sec", type=float, default=230.0)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- load defended model ----
    ck = torch.load(CKPT_AT, map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])
    at_cfg = ck["config"]
    print(f"[model] checkpoint_dual_at.pt loaded "
          f"(AT {at_cfg['at_prob']}, steps {at_cfg['at_steps']}, "
          f"PSR {at_cfg['at_psr_db']} dB)", flush=True)

    # ---- eval set (identical to undefended protocol) ----
    x_rx, y, h_a, p_clean = build_eval_set(args.scenario, args.n_eval,
                                           args.seed)
    with torch.no_grad():
        clean_preds = []
        for i in range(0, x_rx.size(0), 256):
            lg = model.forward_wave(x_rx[i:i + 256], frontend)
            clean_preds.append(lg.argmax(1))
        clean_preds = torch.cat(clean_preds)
    clean_acc_active = (clean_preds == y).float().mean().item()
    print(f"[clean] acc on active eval set ({args.scenario}, "
          f"seed {args.seed}): {clean_acc_active*100:.2f}%", flush=True)

    # ---- clean acc on the standard validation split (cost of the defense) ----
    waves, labels, _ = build_dataset(42, 1000)
    n_val = int(0.2 * waves.size(0))
    val_w, val_y = waves[:n_val], labels[:n_val]
    with torch.no_grad():
        preds = []
        for i in range(0, val_w.size(0), 256):
            lg = model.forward_wave(val_w[i:i + 256], frontend)
            preds.append(lg.argmax(1))
    val_acc = (torch.cat(preds) == val_y).float().mean().item()
    print(f"[clean] val-split acc (seed 42): {val_acc*100:.2f}% "
          f"(undefended: 100.00%)", flush=True)

    # ---- attack sweeps (chunkable, idempotent) ----
    attack_cfg = {"scenario": args.scenario, "steps": args.steps,
                  "psd_margin": args.psd_margin, "seed": args.seed,
                  "n_eval_per_class": args.n_eval,
                  "psr_reference": "clean received-signal window energy "
                                   "(pre-noise)",
                  "mask_band_mhz": ATTACK_BANDS["cv2x_attacker"],
                  "genie": "power-only, no PSD cap",
                  "alpha_frac": 0.25}
    # ---- skip conditions already stored in the output file (resume) ----
    done = set()
    if os.path.exists(RES) and not args.force:
        try:
            prev = json.load(open(RES))
            for mode, mdata in prev.get("defended", {}).items():
                if isinstance(mdata, dict):
                    for setting, runs in mdata.get("runs", {}).items():
                        for psr_key in runs:
                            done.add((mode, setting, psr_key))
        except Exception:
            pass

    dur = []
    todo = []
    for mode in args.modes:
        psr_list = (args.psr if args.psr is not None else
                    (DEFAULT_PSR if mode == "untargeted" else TARGETED_PSR))
        for setting in args.settings:
            for psr in psr_list:
                if (mode, setting, fmt_psr(psr)) in done:
                    continue
                todo.append((mode, setting, psr))

    print(f"[plan] {len(todo)} attack(s) to run this invocation", flush=True)
    for mode, setting, psr in todo:
        look = max(dur) if dur else 30.0
        if time.time() - t0 + look > args.max_time_sec:
            print(f"[guard] stopping before {mode}/{setting}/"
                  f"{fmt_psr(psr)}: elapsed {time.time()-t0:.0f}s + "
                  f"lookahead {look:.0f}s > {args.max_time_sec:.0f}s",
                  flush=True)
            break
        ta = time.time()
        targeted = (mode == "targeted_noise")
        sband = None if setting == "genie" else ATTACK_BANDS["cv2x_attacker"]
        p_budget = p_clean * (10 ** (psr / 10.0))
        adv_preds = run_setting(model, frontend, x_rx, h_a, y, p_budget,
                                sband, targeted, args.steps, args.psd_margin)
        c_asr, n_elig = cond_asr(clean_preds, adv_preds, y,
                                 targeted=targeted, target=NOISE_CLASS)
        rob = (adv_preds == y).float().mean().item()
        rec = {"cond_asr": round(100 * c_asr, 2),
               "robust_acc": round(100 * rob, 2), "n_eligible": n_elig}
        dur.append(time.time() - ta)
        print(f"  {mode}/{setting:10s} {fmt_psr(psr)}: cond-ASR "
              f"{100*c_asr:5.1f}%  robust {100*rob:5.1f}%  "
              f"({dur[-1]:.0f}s)", flush=True)
        store(mode, setting, fmt_psr(psr), rec, attack_cfg, at_cfg,
              clean_acc_active, val_acc, args)

    finalize(attack_cfg, at_cfg, clean_acc_active, val_acc, args,
             time.time() - t0)
    print(f"[done] wrote {RES}", flush=True)


def store(mode, setting, key, rec, attack_cfg, at_cfg, clean_active, val_acc,
          args):
    """Merge one attack result into results/at_defense_results.json."""
    res = load_or_init(attack_cfg, at_cfg, clean_active, val_acc)
    res["defended"].setdefault(mode, {}).setdefault("runs", {}) \
        .setdefault(setting, {})[key] = rec
    res["defended"][mode]["grid_complete"] = None   # recomputed in finalize
    with open(RES, "w") as f:
        json.dump(res, f, indent=2)


def load_or_init(attack_cfg, at_cfg, clean_active, val_acc):
    if os.path.exists(RES):
        with open(RES) as f:
            res = json.load(f)
        res["config"]["attack"] = attack_cfg
        res["defended"]["clean_acc"]["eval_active_urban"] = \
            round(clean_active, 4)
        res["defended"]["clean_acc"]["val_split_seed42"] = round(val_acc, 4)
        return res
    return {
        "experiment": "mask-matched adversarial training defense evaluation "
                      "(CLAIMS C16)",
        "config": {"attack": attack_cfg, "at_training": at_cfg,
                   "at_model": "results/checkpoint_dual_at.pt"},
        "undefended": UNDEFENDED,
        "defended": {
            "model": "DualStreamModel after mask-matched AT "
                     "(results/checkpoint_dual_at.pt)",
            "clean_acc": {"eval_active_urban": round(clean_active, 4),
                          "val_split_seed42": round(val_acc, 4)},
        },
        "deltas": {},
        "notes": {},
    }


def finalize(attack_cfg, at_cfg, clean_active, val_acc, args, chunk_s):
    """Recompute MEAP/PoC summaries and undefended-vs-defended deltas for all
    stored modes; annotate grid completeness; write the final JSON."""
    res = load_or_init(attack_cfg, at_cfg, clean_active, val_acc)
    full_grid = {m: (DEFAULT_PSR if m == "untargeted" else TARGETED_PSR)
                 for m in ("untargeted", "targeted_noise")}
    for mode, mdata in res["defended"].items():
        if not isinstance(mdata, dict) or "runs" not in mdata:
            continue
        out = {}
        asr = {}
        for setting in ("genie", "cv2x_mask"):
            runs = mdata["runs"].get(setting, {})
            keys = sorted(runs.keys(), key=lambda k: float(k[4:-2]))
            psrs = [float(k[4:-2]) for k in keys]
            asr[setting] = [runs[k]["cond_asr"] for k in keys]
            out[f"meap_{setting}_db"], out[f"meap_{setting}_censor"] = \
                meap_curve(psrs, asr[setting], 20.0)
            m10 = runs.get("psr=-10dB")
            out[f"asr_at_minus10_{setting}"] = \
                (m10["cond_asr"] if m10 else None)
            out[f"psr_grid_{setting}"] = psrs
        if "genie" in asr and "cv2x_mask" in asr and asr["genie"] and \
                asr["cv2x_mask"]:
            gk = sorted(mdata["runs"]["genie"].keys(),
                        key=lambda k: float(k[4:-2]))
            mk = sorted(mdata["runs"]["cv2x_mask"].keys(),
                        key=lambda k: float(k[4:-2]))
            pg = [float(k[4:-2]) for k in gk]
            pm = [float(k[4:-2]) for k in mk]
            poc, cnotes = price_of_compliance(pg, asr["genie"], pm,
                                              asr["cv2x_mask"], 20.0)
            out["price_of_compliance_db"] = poc
            out["poc_censor"] = cnotes
        expected = [fmt_psr(p) for p in full_grid[mode]]
        have = [fmt_psr(p) for p in out.get("psr_grid_cv2x_mask", [])]
        out["grid_complete"] = (expected == have)
        res["defended"][mode]["summary"] = out

        # ---- deltas vs UNDEFENDED (positive MEAP delta = defense raises the
        #      power the attacker needs; negative ASR delta = defense helps) ----
        d = {}
        u = UNDEFENDED.get(mode, {})
        for k in ("meap_genie_db", "meap_cv2x_mask_db",
                  "price_of_compliance_db"):
            if out.get(k) is not None and u.get(k) is not None:
                d[k] = round(out[k] - u[k], 3)
        for k in ("asr_at_minus10_cv2x_mask", "asr_at_minus10_genie"):
            if out.get(k) is not None and u.get(k) is not None:
                d[k] = round(out[k] - u[k], 3)
        res["deltas"][mode] = d

    res["deltas"]["clean_acc_val"] = round(
        val_acc - UNDEFENDED["clean_acc_val"], 4)
    res["notes"] = {
        "interpretation": "delta>0 on MEAP = defense pushes the minimum "
                          "effective attack power UP (good for defense); "
                          "delta<0 on ASR@-10dB = fewer successful attacks "
                          "(good for defense). grid_complete flags whether "
                          "the stored PSR grid covers the baseline grid "
                          "(MEAP comparable only when True).",
        "at_training_psr_db": at_cfg.get("at_psr_db"),
        "chunk_elapsed_s": round(chunk_s, 1),
    }
    with open(RES, "w") as f:
        json.dump(res, f, indent=2)
    # console summary
    for mode, mdata in res["defended"].items():
        if isinstance(mdata, dict) and "summary" in mdata:
            s = mdata["summary"]
            print(f"[summary] {mode}: MEAP genie {s.get('meap_genie_db')} "
                  f"({s.get('meap_genie_censor')})  mask "
                  f"{s.get('meap_cv2x_mask_db')} "
                  f"({s.get('meap_cv2x_mask_censor')})  PoC "
                  f"{s.get('price_of_compliance_db')}  "
                  f"grid_complete={s.get('grid_complete')}", flush=True)
            print(f"           vs UNDEFENDED: {UNDEFENDED.get(mode, {})}",
                  flush=True)


if __name__ == "__main__":
    main()
