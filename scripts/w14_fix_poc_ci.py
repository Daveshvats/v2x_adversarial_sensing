"""w14_fix_poc_ci.py — F5 (council round-2, Part IV): honest PoC CI upper tail.

The headline genie MEAP CI lower bound was censored 'le' at the −45 dB grid
edge (w12 headline pair 0 reported censors [null,null], hiding it). The
committed grid55 run (attack_results_urban_untargeted_grid55.json) contains
genie cells at −55/−50 dB; merging them into the canonical curve un-censors
the stronger-attack (upper-Wilson) bound and widens the PoC CI upper tail
from 9.80 to ≈10.1 dB.

Method is bit-identical to scripts/w12_cis.py (same Wilson, same k
reconstruction, same meap_curve interpolation); this script reuses those
functions by import. Updates results/w12_confidence_intervals.json in place
(headline pair 0 + method note + w14_f5_update provenance block).
Idempotent: re-running on an already-updated file is a no-op (checks flag).
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))

from w12_cis import wilson, k_of_cell, curve_ci, meap_curve  # noqa: E402
from attack_mask import meap_curve  # noqa: F811,E402  (re-exported symbol)

ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
W12 = os.path.join(RES, "w12_confidence_intervals.json")
GRID55 = os.path.join(RES, "attack_results_urban_untargeted_grid55.json")


def curve_points_from_w12(curve):
    """[(psr_db, asr_pct, n), ...] from a stored w12 curve (cells carry k/n)."""
    return [(c["psr_db"], c["asr_pct"], c["n"]) for c in curve["cells"]]


def curve_points_from_result(node):
    """[(psr_db, asr, n), ...] from a raw run subtree keyed 'psr=XdB'."""
    pts = []
    for k in sorted(node, key=lambda s: float(s[4:-2])):
        v = node[k]
        pts.append((float(k[4:-2]), v["cond_asr"], v["n_eligible"]))
    return pts


def main():
    w12 = json.load(open(W12))
    pair0 = w12["headline_pairs"][0]
    if pair0.get("genie_merged_grid55"):
        print("already updated (w14_f5 flag present) — no-op")
        return

    # 1. canonical genie curve (urban untargeted) as stored points
    gkey = "per_scenario/urban/runs/untargeted/genie"
    genie = w12["files"]["merged_canonical"]["curves"][gkey]
    canon_pts = curve_points_from_w12(genie)
    canon_psrs = {round(p, 1) for p, _, _ in canon_pts}

    # 2. grid55 genie cells
    g55 = json.load(open(GRID55))
    node = g55["per_scenario"]["urban"]["runs"]["untargeted/genie"]
    extra_pts = [pt for pt in curve_points_from_result(node)
                 if round(pt[0], 1) not in canon_psrs]
    merged_pts = sorted(canon_pts + extra_pts)

    print("canonical genie cells:", len(canon_pts),
          "range", canon_pts[0][0], "..", canon_pts[-1][0])
    print("grid55 additions:", extra_pts)

    # 3. recompute the curve CI on the merged set (same machinery as w12)
    merged = curve_ci(merged_pts, gkey + " + grid55 merge")

    old_lo, old_hi = pair0["meap_genie_ci95"]
    new_lo, new_hi = merged["meap_ci95_db"]
    print(f"genie MEAP CI: [{old_lo}, {old_hi}] -> [{new_lo}, {new_hi}]"
          f"  (censors {pair0.get('censors')} ->"
          f" {merged['meap_ci_censors']})")

    # 4. recompute PoC CI (conservative bound propagation, same formula)
    mask_lo, mask_hi = pair0["meap_mask_ci95"]
    poc_lo = round(mask_lo - new_hi, 2)
    poc_hi = round(mask_hi - new_lo, 2)
    print(f"PoC CI: {pair0['poc_ci95']} -> [{poc_lo}, {poc_hi}]")

    # 5. update the file (preserving everything else)
    pair0["meap_genie_ci95"] = [new_lo, new_hi]
    pair0["poc_ci95"] = [poc_lo, poc_hi]
    pair0["censors"] = list(merged["meap_ci_censors"])
    pair0["genie_merged_grid55"] = True
    pair0["genie_cells_before"] = len(canon_pts)
    pair0["genie_cells_after"] = len(merged_pts)
    w12["method"]["poc_ci"] = (
        "conservative bound propagation (mask_lo - genie_hi, mask_hi - genie_lo);"
        " curves share the eval set so this overstates width. W14/F5 update: the"
        " canonical genie CI bounds are computed on the 12-cell curve MERGED with"
        " the committed grid55 cells (-55/-50 dB), un-censoring the stronger-attack"
        " bound that w12 reported at the -45 dB floor")
    w12["w14_f5_update"] = {
        "finding": "F5 (council round 2): headline genie MEAP CI lower bound was "
                   "'le'-censored at the -45 dB grid edge while censors reported "
                   "[null,null]; PoC CI upper tail understated",
        "fix": "merged grid55 genie cells (-55 dB: 0/300, -50 dB: 3.33%) into the "
               "canonical curve before recomputing Wilson-bound MEAPs",
        "genie_meap_point_db_unchanged": True,
        "genie_meap_ci95_db": [new_lo, new_hi],
        "genie_ci_censors": merged["meap_ci_censors"],
        "poc_ci95_db": [poc_lo, poc_hi],
        "merged_curve": {c["psr_db"]: {"k": c["k"], "n": c["n"],
                                       "asr_pct": c["asr_pct"]}
                         for c in merged["cells"]},
        "note": "point MEAP unchanged (20% crossing bracketed above -45 dB); "
                "only the CI lower bound moves",
    }
    json.dump(w12, open(W12, "w"), indent=1)
    print("w12_confidence_intervals.json updated")


if __name__ == "__main__":
    main()
