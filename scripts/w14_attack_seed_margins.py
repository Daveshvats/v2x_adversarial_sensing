#!/usr/bin/env python3
"""Attack-seed replication of adaptive-attack MEAPs / PoCs / margins
(Wave 14, item 12; closes council 34-d blocker i: "attack seed fixed at 7").

Mirror of w13_seed_replication.py (which varied DEFENSE seeds at fixed
attack seed 7). Here the defense models are the canonical seed-42 trio
(dual / AT-s42 / TRADES-s42) and the ATTACK seed varies over {7, 11, 22}.

Protocol: per-arm PSR grids IDENTICAL across attack seeds (per defense):
  dual:   genie [-50,-45,-40,-35,-30]   cv2x_mask [-45,-40,-35,-30,-25,-20]
  at:     genie [-45,-40,-35,-30,-25]   cv2x_mask [-35,-30,-25,-20,-15,-10]
  trades: genie [-45,-40,-35,-30,-25]   cv2x_mask [-35,-30,-25,-20,-15,-10,-5]

Deliverables (results/w14_attack_seed_margins.json):
  * per (defense, attack seed): MEAP genie / mask, PoC — re-derived from
    stored curves via meap_curve / price_of_compliance (independent of the
    stored summaries; cross-checked against them)
  * margin(defense, s) = mask_MEAP(defense, s) - mask_MEAP(dual, s)
    (seed-matched undefended reference — same formula as w13)
  * aggregates over attack seeds {7,11,22}: mean / range / std (ddof=1)
    for margins AND PoCs, per defense
  * checks: grid identity across seeds (per defense, per arm); win_flags
    capture presence and length; censor flags; stored-summary match
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
from attack_mask import meap_curve, price_of_compliance  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "results")
ATTACK_SEEDS = [7, 11, 22]
DEFENSES = ["dual", "at", "trades"]
R = 5
N_EVAL = 300  # 100/class x 3 classes


def fname_for(defense, seed):
    return os.path.join(OUT, f"adaptive_{defense}_s{seed}_r{R}.json")


def meaps_from_curves(doc):
    """Re-derive (grids, curves, genie MEAP, mask MEAP, PoC) from stored
    cells — independent of the stored summary block."""
    runs = doc["runs"]

    def curve(setting):
        cells = runs[f"untargeted/{setting}"]
        ps = sorted(float(k[4:-2]) for k in cells)
        asr = [cells[f"psr={p:+.0f}dB"]["cond_asr"] for p in ps]
        return ps, asr

    pg, g = curve("genie")
    pm, m = curve("cv2x_mask")
    mg, cg = meap_curve(pg, g, 20.0)
    mm, cm = meap_curve(pm, m, 20.0)
    poc, cnotes = price_of_compliance(pg, g, pm, m, 20.0)
    return pg, g, pm, m, mg, cg, mm, cm, poc, cnotes


def win_flags_stats(doc):
    """win_flags presence / lengths / -1 rate per arm."""
    out = {}
    for setting in ("genie", "cv2x_mask"):
        cells = doc["runs"].get(f"untargeted/{setting}", {})
        recs = [c for c in cells.values() if "win_flags" in c]
        if not recs:
            out[setting] = {"capture": False}
            continue
        lens = {len(c["win_flags"]) for c in recs}
        n_minus1 = sum(sum(1 for v in c["win_flags"] if v == -1)
                       for c in recs)
        n_tot = sum(len(c["win_flags"]) for c in recs)
        out[setting] = {
            "capture": True, "n_cells": len(recs),
            "flag_len_uniform": (len(lens) == 1),
            "flag_len": sorted(lens), "frac_clean_wrong": round(n_minus1 / n_tot, 4),
        }
    return out


def main():
    report = {
        "experiment": "Attack-seed replication of adaptive-attack margins "
                      "(council 34-d blocker i: attack seed was fixed at 7)",
        "protocol": {
            "attack": "PGD-50 x R=5 best-of restarts, urban, untargeted",
            "attack_seeds": ATTACK_SEEDS,
            "defense_models": "canonical seed-42 trio (checkpoint_dual.pt, "
                              "checkpoint_dual_at.pt, "
                              "checkpoint_dual_trades.pt)",
            "n_eval_per_class": 100,
            "margin_definition": "mask_MEAP(defense, s) - mask_MEAP(dual, s), "
                                 "seed-matched undefended reference "
                                 "(same formula as w13 defense-seed study)",
            "per_arm_grids": {
                "dual":   {"genie": [-50, -45, -40, -35, -30],
                           "cv2x_mask": [-45, -40, -35, -30, -25, -20]},
                "at":     {"genie": [-45, -40, -35, -30, -25],
                           "cv2x_mask": [-35, -30, -25, -20, -15, -10]},
                "trades": {"genie": [-45, -40, -35, -30, -25],
                           "cv2x_mask": [-35, -30, -25, -20, -15, -10, -5]},
            },
        },
        "defenses": {},
        "checks": {},
    }

    # ---- load all 9 grids ----
    data = {}
    for d in DEFENSES:
        for s in ATTACK_SEEDS:
            path = fname_for(d, s)
            ok = os.path.exists(path)
            report["checks"][f"{d}_s{s}_present"] = ok
            if not ok:
                print(f"[MISS] {path}")
                continue
            doc = json.load(open(path))
            data[(d, s)] = (doc, meaps_from_curves(doc), win_flags_stats(doc))

    # ---- per-defense tables + margins ----
    for d in DEFENSES:
        rows = []
        for s in ATTACK_SEEDS:
            if (d, s) not in data:
                continue
            doc, (pg, g, pm, m, mg, cg, mm, cm, poc, cnotes), wf = data[(d, s)]
            stored = doc.get("summary", {}).get("untargeted", {})
            match = (abs(mg - (stored.get("meap_genie_db") or 1e9)) < 1e-6 and
                     abs(mm - (stored.get("meap_cv2x_mask_db") or 1e9)) < 1e-6)
            report["checks"][f"{d}_s{s}_summary_match"] = bool(match)
            report["checks"][f"{d}_s{s}_uncensored"] = (cg is None and
                                                        cm is None)
            rows.append({
                "attack_seed": s,
                "meap_genie_db": round(mg, 3), "meap_genie_censor": cg,
                "meap_mask_db": round(mm, 3), "meap_mask_censor": cm,
                "poc_db": round(poc, 3) if poc is not None else None,
                "poc_censor": cnotes,
                "genie_grid": pg, "mask_grid": pm,
                "genie_curve": g, "mask_curve": m,
                "win_flags": wf,
                "elapsed_s": doc.get("elapsed_s"),
            })

        # grid identity across attack seeds (per arm)
        for arm, key in (("genie", "genie_grid"), ("cv2x_mask", "mask_grid")):
            grids = [r[key] for r in rows]
            report["checks"][f"{d}_{arm}_grid_identity_across_seeds"] = (
                all(x == grids[0] for x in grids))
            if not all(x == grids[0] for x in grids):
                for r in rows:
                    print(f"[WARN] {d} s{r['attack_seed']} {arm} grid "
                          f"{r[key]}")

        # seed-matched dual reference for margins
        drows = []
        for r in rows:
            dual_ref = None
            if (("dual", r["attack_seed"]) in data and d != "dual"):
                dual_ref = data[("dual", r["attack_seed"])][1][6]  # mm
            if d == "dual":
                dual_ref = r["meap_mask_db"]
            if dual_ref is None:
                continue
            r["margin_vs_dual_db"] = round(r["meap_mask_db"] - dual_ref, 3)
            drows.append(r)

        margins = [r["margin_vs_dual_db"] for r in drows]
        pocs = [r["poc_db"] for r in drows if r["poc_db"] is not None]
        agg = {
            "n_attack_seeds": len(drows),
            "margin_mean_db": round(float(np.mean(margins)), 3),
            "margin_range_db": round(float(np.max(margins) -
                                                  np.min(margins)), 3),
            "margin_std_db": round(float(np.std(margins, ddof=1)), 3)
            if len(margins) > 1 else None,
            "per_seed_margins_db": margins,
            "poc_mean_db": round(float(np.mean(pocs)), 3),
            "poc_range_db": round(float(np.max(pocs) - np.min(pocs)), 3),
            "poc_std_db": round(float(np.std(pocs, ddof=1)), 3)
            if len(pocs) > 1 else None,
            "per_seed_poc_db": pocs,
        }
        report["defenses"][d] = {"rows": drows, "aggregates": agg}

        print(f"\n=== {d} ===")
        for r in drows:
            wf = ("s%s: capture" % r["attack_seed"]) if \
                r["win_flags"]["genie"].get("capture") else \
                f"s{r['attack_seed']}: NO capture"
            print(f"  s{r['attack_seed']:>2}: genie {r['meap_genie_db']:8.3f}"
                  f"  mask {r['meap_mask_db']:8.3f}  PoC {r['poc_db']:6.3f}"
                  f"  margin {r['margin_vs_dual_db']:+7.3f}  [{wf}]")
        print(f"  margin mean {agg['margin_mean_db']:+.3f}  range "
              f"{agg['margin_range_db']:.3f}  std {agg['margin_std_db']}"
              f"  | PoC mean {agg['poc_mean_db']:.3f} range "
              f"{agg['poc_range_db']:.3f}")

    # ---- headline: attack-seed vs defense-seed variance comparison ----
    if all(len(report["defenses"][d]["rows"]) == 3 for d in DEFENSES):
        at_m = report["defenses"]["at"]["aggregates"]
        tr_m = report["defenses"]["trades"]["aggregates"]
        report["headline"] = {
            "at_margin_over_attack_seeds_db": at_m,
            "trades_margin_over_attack_seeds_db": tr_m,
            "comparison_note": "defense-seed spread (w13, attack seed 7): "
                               "AT +9.76 +/- 2.80 dB over defense seeds "
                               "{42,43,53}; attack-seed spread here is "
                               "measured on the same margin formula with "
                               "the canonical seed-42 models",
        }

    path = os.path.join(OUT, "w14_attack_seed_margins.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    os.replace(tmp, path)
    n_bad = sum(1 for k, v in report["checks"].items()
                if v is not True and k.endswith(("_present", "_match",
                                                 "_uncensored",
                                                 "_identity_across_seeds")))
    print(f"\n[w14] checks failing: {n_bad}")
    print(f"[w14] wrote {path}")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
