#!/usr/bin/env python3
"""w11_check2_meap.py — W11 auditor CHECK 2 (independent MEAP recomputation).

Loads results/papr_pa_results.json, takes the stored psr_grid and the stored
cond-ASR curves, and re-derives MEAP with THIS script's own implementation
(threshold 20%, piecewise-linear interpolation, first crossing from below,
censor flags). Nothing is imported from run_papr_pa.py or attack_mask.py.

Verifies:
  * control_no_pa  -> control_meap_db  (expected -37.00)
  * pa_p3_ibo0     -> -36.40   (shift +0.6)
  * pa_p3_ibo6     -> -37.09   (shift -0.09)
  * pa_p3_ibo12/13, pa_p2_ibo6/15 -> -37.00 (shift 0.0)
  * pa_aware_p3_ibo13 -> -38.23 (shift -1.23), 100% strict, -60.2 dBr p95
  * pa_aware_p3_ibo0  -> -38.32 (shift -1.32)
  * shift_vs_control_db == round(meap - control, 2) for every __meap entry
  * curve lengths match the psr grid; curves are monotone non-decreasing
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results", "papr_pa_results.json")

def meap_own(psr, asr, thr=20.0):
    """MEAP: PSR where ASR first crosses `thr` from below, linear interp.
    Returns (meap, censor). Own implementation (no attack_mask import)."""
    psr = [float(p) for p in psr]
    a = [float(x) for x in asr]
    if not a:
        return None, "ge"
    if a[0] >= thr:
        return psr[0], "le"
    for i in range(len(a) - 1):
        if a[i] < thr <= a[i + 1]:
            t = (thr - a[i]) / (a[i + 1] - a[i])
            return psr[i] + t * (psr[i + 1] - psr[i]), None
    return None, "ge"

def main():
    res = json.load(open(RES))
    psr = res["psr_grid"]
    curves = res["curves"]
    ctrl = res["control_meap_db"]

    fails = []

    def check(name, expect_meap, expect_shift=None):
        cur = curves[name]
        entry = curves[name + "__meap"]
        if len(cur) != len(psr):
            fails.append(f"{name}: curve len {len(cur)} != grid {len(psr)}")
        m, c = meap_own(psr, cur)
        st_m = entry["meap_db"]
        st_c = entry.get("censor")
        st_shift = entry.get("shift_vs_control_db")
        my_shift = None if (m is None or ctrl is None) else m - ctrl
        line = (f"{name:22s} own {m:9.4f} stored {st_m:9.4f} "
                f"(d {m-st_m:+.5f})  shift own {my_shift:+.4f} "
                f"stored {st_shift:+.2f} (d {my_shift-st_shift:+.5f})")
        print(line)
        if abs(m - st_m) > 1e-6:
            fails.append(f"{name}: own MEAP {m} != stored {st_m}")
        if c != st_c:
            fails.append(f"{name}: censor own {c} != stored {st_c}")
        if st_shift is not None and abs(my_shift - st_shift) > 5e-3:
            fails.append(f"{name}: shift mismatch {my_shift} vs {st_shift}")
        # shift consistency: stored shift == round(stored meap - ctrl, 2)
        if st_shift is not None and abs(
                round(st_m - ctrl, 2) - st_shift) > 1e-9:
            fails.append(f"{name}: stored shift {st_shift} != "
                         f"round(meap-ctrl,2)={round(st_m-ctrl,2)}")
        if expect_meap is not None and abs(st_m - expect_meap) > 0.005:
            fails.append(f"{name}: stored {st_m} != expected {expect_meap}")
        if expect_shift is not None and abs(st_shift - expect_shift) > 0.005:
            fails.append(f"{name}: shift {st_shift} != expected {expect_shift}")
        # monotonicity of the cond-ASR curve (sanity)
        if any(cur[i + 1] < cur[i] - 1e-9 for i in range(len(cur) - 1)):
            fails.append(f"{name}: curve not monotone")

    print(f"control_meap_db stored: {ctrl}")
    # control arm
    m_ctrl, c_ctrl = meap_own(psr, curves["control_no_pa"])
    print(f"control_no_pa          own {m_ctrl:9.4f} stored {ctrl:9.4f} "
          f"(d {m_ctrl-ctrl:+.5f})  censor own {c_ctrl} "
          f"stored {res['control_censor']}")
    if abs(m_ctrl - ctrl) > 1e-6:
        fails.append(f"control: own {m_ctrl} != stored {ctrl}")
    if c_ctrl != res["control_censor"]:
        fails.append("control censor mismatch")

    check("pa_p3_ibo0", -36.40, 0.60)
    check("pa_p3_ibo6", -37.09, -0.09)
    check("pa_p3_ibo12", -37.00, 0.0)
    check("pa_p3_ibo13", -37.00, 0.0)
    check("pa_p2_ibo6", -37.00, 0.0)
    check("pa_p2_ibo15", -37.00, 0.0)
    check("pa_aware_p3_ibo13", -38.23, -1.23)
    check("pa_aware_p3_ibo0", -38.32, -1.32)

    # pa-aware extra stats quoted by the paper
    e13 = curves["pa_aware_p3_ibo13__meap"]
    e0 = curves["pa_aware_p3_ibo0__meap"]
    print(f"\npa_aware ibo13: strict pass {e13['post_tx_pass_strict']:.4f} "
          f"(paper: 100%)  p95 excess {e13['post_tx_mask_excess_p95_dbr']} dBr "
          f"(paper: -60)")
    print(f"pa_aware ibo0 : strict pass {e0['post_tx_pass_strict']:.4f}  "
          f"p95 excess {e0['post_tx_mask_excess_p95_dbr']} dBr (paper: "
          f"non-compliant)")
    if abs(e13["post_tx_pass_strict"] - 1.0) > 1e-9:
        fails.append("pa_aware ibo13 strict pass != 1.0")
    if e13["post_tx_mask_excess_p95_dbr"] > -55:
        fails.append("pa_aware ibo13 p95 excess not ~ -60 dBr")

    print("\n" + ("ALL CHECK2 PASS" if not fails else "CHECK2 FAILURES:"))
    for f in fails:
        print("  " + f)
    return 1 if fails else 0

if __name__ == "__main__":
    raise SystemExit(main())
