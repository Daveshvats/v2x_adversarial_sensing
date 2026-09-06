#!/usr/bin/env python3
"""w12_cis.py — Wave 12 part 2: binomial 95% confidence intervals for every
headline conditional-ASR / MEAP / PoC number (council P0, reviewer 21-d:
"no error bars anywhere — binomial CI on a single MEAP cell is ±1.0–1.9 dB at
n=300, LARGER than the 1.06 dB 3-seed spread used as the stability argument").

Method
  * Every attack cell stores cond_asr (percent, 2dp) and n_eligible. We
    reconstruct k = round(asr*n/100) and verify the round-trip |100k/n - asr|
    <= 0.01 (flag any cell where 2dp storage loses the exact k).
  * Cell-level Wilson score interval (z = 1.959964, 95%).
  * MEAP (20%-crossing interpolation, src.attack_mask.meap_curve) is
    recomputed on the per-cell lower-bound curve and upper-bound curve:
    MEAP_lo = meap_curve(asr_lo), MEAP_hi = meap_curve(asr_hi). Censoring
    flags inherit the same semantics as the point estimate.
  * PoC = MEAP_mask - MEAP_genie; its interval is propagated conservatively:
    [MEAP_mask_lo - MEAP_genie_hi, MEAP_mask_hi - MEAP_genie_lo]. The two
    curves share the same eval set (correlated), so this is conservative —
    disclosed in the JSON notes.
  * IMPORTANT scope note: these intervals quantify SAMPLING noise at fixed
    model/data/protocol. Seed-to-seed model variance (the 1.06 dB 3-seed
    spread) is a separate, additive source of uncertainty.

Outputs results/w12_confidence_intervals.json (full provenance) and prints a
compact headline table. Pure numpy; safe to run while torch jobs use the CPUs
(imports attack_mask for meap_curve — torch import is side-effect-free here).
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from attack_mask import meap_curve

OUT = os.path.join(ROOT, "results")
Z = 1.959964                          # 95% two-sided
MEAP_THRESHOLD = 20.0


def wilson(k, n):
    """Wilson score interval in PERCENT."""
    if n == 0:
        return None, None
    p = k / n
    denom = 1.0 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return 100 * (center - half), 100 * (center + half)


def k_of_cell(asr_pct, n):
    k = round(asr_pct * n / 100.0)
    k = min(max(k, 0), n)
    rt = abs(100.0 * k / n - asr_pct)
    return k, rt


def find_curves(node, path=""):
    """Yield (curve_path, [(psr_db, asr, n), ...]) for every run subtree whose
    leaves are cond_asr cells keyed 'psr=XdB'."""
    curves = []
    if isinstance(node, dict):
        leaves = {k: v for k, v in node.items() if k.startswith("psr=")}
        if leaves and all(isinstance(v, dict) and "cond_asr" in v
                          for v in leaves.values()):
            pts = []
            for k in sorted(leaves, key=lambda s: float(s[4:-2])):
                v = leaves[k]
                pts.append((float(k[4:-2]), v["cond_asr"], v["n_eligible"]))
            curves.append((path, pts))
            return curves
        for k, v in node.items():
            if k in ("config", "experiment", "summary", "history",
                     "recipe", "at_scheme", "at_training", "at_model",
                     "source", "elapsed_s", "undefended_reference",
                     "clean_acc_active", "clean_acc_val"):
                continue
            curves += find_curves(v, path + "/" + str(k))
    return curves


def curve_ci(pts, label):
    """Wilson CIs per cell + MEAP interval via bound-curve interpolation."""
    cells, rt_bad = [], []
    for psr, asr, n in pts:
        k, rt = k_of_cell(asr, n)
        if rt > 0.011:
            rt_bad.append({"psr": psr, "asr": asr, "n": n, "k": k,
                           "roundtrip_err": round(rt, 4)})
        lo, hi = wilson(k, n)
        cells.append({"psr_db": psr, "k": k, "n": n, "asr_pct": asr,
                      "ci95_pct": [round(lo, 2), round(hi, 2)]})
    psrs = [p for p, _, _ in pts]
    a_pt = [a for _, a, _ in pts]
    a_lo = [c["ci95_pct"][0] for c in cells]
    a_hi = [c["ci95_pct"][1] for c in cells]
    m_pt, c_pt = meap_curve(psrs, a_pt, MEAP_THRESHOLD)
    m_lo, c_lo = meap_curve(psrs, a_lo, MEAP_THRESHOLD)
    m_hi, c_hi = meap_curve(psrs, a_hi, MEAP_THRESHOLD)
    # canonical orientation: MEAP is a PSR value; lower MEAP = stronger attack.
    # interval = [m_hi, m_lo] in dB (attack-power axis), censored bounds kept.
    return {
        "label": label,
        "n_cells": len(cells),
        "meap_db": m_pt, "meap_censor": c_pt,
        "meap_ci95_db": [None if m_hi is None else round(m_hi, 2),
                         None if m_lo is None else round(m_lo, 2)],
        "meap_ci_censors": [c_hi, c_lo],
        "asr_at_lowest_psr": cells[0] if cells else None,
        "k_roundtrip_flags": rt_bad,
        "cells": cells,
    }


FILES = {
    "merged_canonical": "attack_results_merged.json",
    "at_defense_C16": "at_defense_results.json",
    "trades_C17": "attack_results_urban_untargeted_trades.json",
    "at_urban_alt_C16": "attack_results_urban_untargeted_at.json",
    "seed123_C12": "attack_results_urban_untargeted_s123.json",
    "seed456_C12": "attack_results_urban_untargeted_s456.json",
    "seed11_C12": "attack_results_urban_untargeted_seed11.json",
    "real_wifi_C21": "real_wifi_attack.json",
    "victim2_transfer_C22C23": "victim2_transfer.json",
    "csi_mismatch_C13": "csi_mismatch.json",
    "atrural_C24": "attack_results_rural_untargeted_atrural.json",
    "pgd50_R1": "pgd50_urban_untargeted.json",
    "margin10_C15": "sensitivity_margin10_urban_untargeted.json",
}
# adaptive + grid55 files, if present at run time
for tag, fn in [("adaptive_at_R5", "adaptive_at_s7_r5.json"),
                ("adaptive_trades_R5", "adaptive_trades_s7_r5.json"),
                ("adaptive_dual_R5", "adaptive_dual_s7_r5.json"),
                ("adaptive_at_R1", "adaptive_at_s7_r1.json"),
                ("grid55_C12", "attack_results_urban_untargeted_grid55.json")]:
    if os.path.exists(os.path.join(OUT, fn)):
        FILES[tag] = fn


def poc_interval(mask_ci, genie_ci):
    """PoC = MEAP_mask - MEAP_genie (dB), conservative propagation.
    Inputs are [lower_dB, upper_dB] MEAP intervals (see method.meap_ci).
    PoC_min = mask_lower_dB - genie_upper_dB (mask strongest, genie weakest);
    PoC_max = mask_upper_dB - genie_lower_dB. Returns [PoC_min, PoC_max]."""
    if mask_ci is None or genie_ci is None:
        return None
    if mask_ci[0] is None or mask_ci[1] is None or \
            genie_ci[0] is None or genie_ci[1] is None:
        return None
    poc_min = mask_ci[0] - genie_ci[1]
    poc_max = mask_ci[1] - genie_ci[0]
    return [round(poc_min, 2), round(poc_max, 2)]


def main():
    report = {
        "experiment": "Wave 12 part 2: binomial 95% confidence intervals "
                      "for headline attack numbers",
        "method": {
            "cell_ci": "Wilson score interval, z=1.959964 (95%), "
                       "k = round(cond_asr * n_eligible / 100)",
            "meap_ci": "meap_curve (20% threshold) recomputed on per-cell "
                       "Wilson bound curves; stored as [lower_dB, upper_dB]: "
                       "lower dB = upper-ASR curve (stronger attack), upper dB "
                       "= lower-ASR curve (weaker attack)",
            "poc_ci": "conservative bound propagation "
                      "(mask_lo - genie_hi, mask_hi - genie_lo); curves share "
                      "the eval set so this overstates width",
            "scope": "sampling noise at fixed model/data/protocol ONLY; "
                     "model-seed variance (3-seed spread 1.06 dB) is "
                     "additive and reported separately",
            "meap_threshold_pct": MEAP_THRESHOLD,
        },
        "files": {},
    }
    for tag, fn in FILES.items():
        path = os.path.join(OUT, fn)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        curves = find_curves(d)
        entry = {"file": fn, "curves": {}}
        for cpath, pts in curves:
            entry["curves"][cpath.strip("/")] = curve_ci(pts, cpath.strip("/"))
        report["files"][tag] = entry

    # ---- paired MEAP/PoC summary for the headline pairs ----
    pairs = []
    def get_curve(tag, name):
        f = report["files"].get(tag)
        if not f:
            return None
        c = f["curves"].get(name)
        return c

    def pair(tag, genie_name, mask_name, label):
        g, m = get_curve(tag, genie_name), get_curve(tag, mask_name)
        if g and m:
            pairs.append({
                "label": label,
                "meap_genie_db": g["meap_db"], "meap_genie_ci95": g["meap_ci95_db"],
                "meap_mask_db": m["meap_db"], "meap_mask_ci95": m["meap_ci95_db"],
                "poc_db": (None if (g["meap_db"] is None or m["meap_db"] is None)
                           else round(m["meap_db"] - g["meap_db"], 2)),
                "poc_ci95": poc_interval(m["meap_ci95_db"], g["meap_ci95_db"]),
                "censors": [g["meap_censor"], m["meap_censor"]],
            })

    pair("merged_canonical", "per_scenario/urban/runs/untargeted/genie",
         "per_scenario/urban/runs/untargeted/cv2x_mask",
         "C8/C9 urban untargeted (canonical)")
    pair("merged_canonical", "per_scenario/urban/runs/targeted_noise/genie",
         "per_scenario/urban/runs/targeted_noise/cv2x_mask",
         "C10 urban targeted-noise")
    pair("merged_canonical", "per_scenario/highway/runs/untargeted/genie",
         "per_scenario/highway/runs/untargeted/cv2x_mask",
         "highway untargeted")
    pair("merged_canonical", "per_scenario/rural/runs/untargeted/genie",
         "per_scenario/rural/runs/untargeted/cv2x_mask",
         "rural untargeted")
    pair("at_defense_C16", "defended/untargeted/runs/genie",
         "defended/untargeted/runs/cv2x_mask", "C16 AT urban untargeted")
    pair("trades_C17", "per_scenario/urban/runs/untargeted/genie",
         "per_scenario/urban/runs/untargeted/cv2x_mask",
         "C17 TRADES urban untargeted")
    pair("seed123_C12", "per_scenario/urban/runs/untargeted/genie",
         "per_scenario/urban/runs/untargeted/cv2x_mask", "C12 seed 123")
    pair("seed456_C12", "per_scenario/urban/runs/untargeted/genie",
         "per_scenario/urban/runs/untargeted/cv2x_mask", "C12 seed 456")
    pair("seed11_C12", "per_scenario/urban/runs/untargeted/genie",
         "per_scenario/urban/runs/untargeted/cv2x_mask", "C12 attack seed 11")
    pair("real_wifi_C21", "frozen/genie/curve", "frozen/cv2x_mask/curve",
         "C21 real-WiFi frozen")
    pair("real_wifi_C21", "finetuned/genie/curve", "finetuned/cv2x_mask/curve",
         "C21 real-WiFi finetuned")
    pair("csi_mismatch_C13", "runs/perfect_csi/genie",
         "runs/perfect_csi/cv2x_mask", "C13 perfect CSI")
    if get_curve("csi_mismatch_C13", "runs/csi_mismatch/cv2x_mask"):
        pair("csi_mismatch_C13", "runs/csi_mismatch/genie",
             "runs/csi_mismatch/cv2x_mask", "C13 CSI mismatch")
    pair("atrural_C24", "per_scenario/rural/runs/untargeted/genie",
         "per_scenario/rural/runs/untargeted/cv2x_mask", "C24 AT-rural")
    pair("pgd50_R1", "per_scenario/urban/runs/untargeted/genie",
         "per_scenario/urban/runs/untargeted/cv2x_mask",
         "PGD-50 R=1 urban")
    pair("adaptive_at_R5", "runs/untargeted/genie",
         "runs/untargeted/cv2x_mask", "ADAPTIVE PGD-50x5 vs AT")
    pair("adaptive_trades_R5", "runs/untargeted/genie",
         "runs/untargeted/cv2x_mask", "ADAPTIVE PGD-50x5 vs TRADES")
    pair("adaptive_dual_R5", "runs/untargeted/genie",
         "runs/untargeted/cv2x_mask",
         "ADAPTIVE PGD-50x5 vs undefended")
    if os.path.exists(os.path.join(OUT, "adaptive_at_s7_r1.json")):
        pair("adaptive_at_R1", "runs/untargeted/genie",
             "runs/untargeted/cv2x_mask",
             "PGD-50 R=1 vs AT (step-starvation)")

    report["headline_pairs"] = pairs

    with open(os.path.join(OUT, "w12_confidence_intervals.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"{'pair':44s} {'genie':>20s} {'mask':>20s} {'PoC [95% CI]':>22s}")
    for p in pairs:
        def f_(v, ci):
            if v is None:
                return "censored/ge".rjust(20)
            s = f"{v:7.1f}"
            if ci and ci[0] is not None and ci[1] is not None:
                s += f" [{ci[0]:.1f},{ci[1]:.1f}]"
            return s.rjust(20)
        poc = "-"
        if p["poc_db"] is not None:
            poc = f"{p['poc_db']:.1f}"
            if p["poc_ci95"]:
                poc += f" [{p['poc_ci95'][0]:.1f},{p['poc_ci95'][1]:.1f}]"
        print(f"{p['label'][:44]:44s} {f_(p['meap_genie_db'], p['meap_genie_ci95'])}"
              f" {f_(p['meap_mask_db'], p['meap_mask_ci95'])} {poc:>22s}")
    nflags = sum(len(c["k_roundtrip_flags"])
                 for f in report["files"].values()
                 for c in f["curves"].values())
    print(f"\nk-reconstruction round-trip flags: {nflags} (cells where 2dp "
          f"storage could not exactly recover k)")
    print("wrote results/w12_confidence_intervals.json")


if __name__ == "__main__":
    main()
