"""ci_recompute_meaps.py — CI gate step (F4, council round-2 Part IV; Wave 14).

Recomputes five headline MEAPs from the stored result JSONs in PURE PYTHON
(stdlib only — no torch, no numpy, ~1 s) and asserts they match the stored
summaries. This is the gate step the council demanded in P0-4 and the round-2
reviewers confirmed was never landed (finding F4). The MEAP interpolation is
byte-equivalent to src.attack_mask.meap_curve (same 20%-crossing linear
interpolation, same censor semantics), re-implemented locally so CI does not
need the torch wheel.

Exits non-zero on any mismatch > 0.05 dB or on missing files.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
THRESH = 20.0
TOL_DB = 0.05


def meap_curve(psr_db_list, asr_list, threshold=THRESH):
    """Pure-python replica of src.attack_mask.meap_curve."""
    if len(asr_list) == 0:
        return None, "ge"
    if asr_list[0] >= threshold:
        return float(psr_db_list[0]), "le"
    for i in range(len(asr_list) - 1):
        if asr_list[i] < threshold <= asr_list[i + 1]:
            t = (threshold - asr_list[i]) / (asr_list[i + 1] - asr_list[i] + 1e-12)
            return float(psr_db_list[i] + t * (psr_db_list[i + 1] - psr_db_list[i])), None
    return None, "ge"


def cells(node):
    """Sorted [(psr, asr)] from a run subtree keyed 'psr=XdB'."""
    pts = [(float(k[4:-2]), v["cond_asr"])
           for k, v in node.items() if k.startswith("psr=")]
    return sorted(pts)


def check(name, node, expected):
    pts = cells(node)
    meap, censor = meap_curve([p for p, _ in pts], [a for _, a in pts])
    ok = meap is not None and expected is not None and abs(meap - expected) <= TOL_DB
    print(f"  {name}: recomputed {None if meap is None else round(meap, 3)}"
          f" vs stored {None if expected is None else round(expected, 3)}"
          f" (censor={censor}) -> {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    failures = 0

    # 1. canonical urban untargeted mask MEAP (C8/C9 headline, -37.08)
    d = json.load(open(os.path.join(RES, "attack_results_merged.json")))
    u = d["per_scenario"]["urban"]
    failures += not check("canonical urban mask (C8/C9)",
                          u["runs"]["untargeted/cv2x_mask"],
                          u["summary"]["untargeted"]["meap_cv2x_mask_db"])

    # 2. ETSI Table-7 shaped MEAP (C34, -36.72)
    d = json.load(open(os.path.join(RES, "etsi_mask_results.json")))
    failures += not check("ETSI Table-7 mask (C34)",
                          d["runs"]["untargeted/etsi_table7"],
                          d["summary"]["meap_etsi_table7_db"])

    # 3-5. adaptive mask MEAPs (C33: dual / AT / TRADES)
    for fname, label in [("adaptive_dual_s7_r5.json", "adaptive dual  (-39.43)"),
                         ("adaptive_at_s7_r5.json", "adaptive AT    (-32.70)"),
                         ("adaptive_trades_s7_r5.json", "adaptive TRADES(-29.56)")]:
        d = json.load(open(os.path.join(RES, fname)))
        failures += not check(label,
                              d["runs"]["untargeted/cv2x_mask"],
                              d["summary"]["untargeted"]["meap_cv2x_mask_db"])

    print("\n" + ("MEAP-GATE PASS (5/5)" if failures == 0
                  else f"MEAP-GATE FAIL ({failures} mismatches)"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
