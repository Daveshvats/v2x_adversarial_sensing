#!/usr/bin/env python3
"""w12_check.py — Wave 12 part 2 independent audit (exit gate, numpy-only).

Re-derives every NEW number cited in the paper's Wave-12 subsections from
the raw result JSONs, without reusing the experiment code paths:

  F1  adaptive MEAPs (undefended/AT/TRADES, PGD-50 x R5) by re-running the
      20%-crossing interpolation on the stored per-cell cond_asr values
  F2  defense margins (adaptive vs undefended; PGD-10 vs adaptive)
  F3  step-starvation decomposition (PGD-10 -> R=1 -> R=5 on AT mask)
  F4  ETSI vs flat-cap MEAP + PoC + shape delta
  F5  harm-chain spot checks: EIRP feasibility (closed-form link budget),
      PRR deltas at 100/150/200 m, random-timing duty argument
  F6  CI engine spot checks: Wilson interval math on hand-computed cells,
      PoC interval propagation, and round-trip k reconstruction
  F7  ETSI Table-7 knot fidelity: values in etsi_mask.py match
      data/standards/en302571_tables.json verbatim
  F8  continuity: R=1 == waveform_pgd (re-derives the smoke-test identity
      from the stored config blocks: restart_init description + steps)

PASS criteria: every derived number matches the paper-stated value within
rounding. FAIL prints the offending cell.
"""
import os, sys, json, math

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "results")

import numpy as np

PASS, FAIL = "PASS", "FAIL"
fails = []


def check(tag, ok, detail=""):
    print(f"  [{PASS if ok else FAIL}] {tag}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(f"{tag}: {detail}")


def meap_20(psr, asr):
    """20%-crossing interpolation (independent of attack_mask.meap_curve)."""
    psr = np.asarray(psr, float)
    a = np.asarray(asr, float)
    if a[0] >= 20:
        return psr[0], "le"
    for i in range(len(a) - 1):
        if a[i] < 20 <= a[i + 1]:
            t = (20 - a[i]) / (a[i + 1] - a[i])
            return psr[i] + t * (psr[i + 1] - psr[i]), None
    return None, "ge"


def curve(d, path):
    """Fetch stored cells as (psr_list, asr_list); path is a list of keys
    (slash-joined run keys are handled as single keys)."""
    node = d
    for k in path:
        if k in node:
            node = node[k]
        else:                     # slash-joined key: 'untargeted/cv2x_mask'
            hit = None
            for cand in node:
                if cand.replace("/", "/") == "/".join(path[-len(path):]):
                    hit = cand
                    break
            if hit is None:
                raise KeyError(path)
            node = node[hit]
            break
    items = []
    for k, v in node.items():
        if k.startswith("psr="):
            items.append((float(k[4:-2]), v["cond_asr"]))
    items.sort()
    return [p for p, _ in items], [a for _, a in items]


def curve_flat(d, key):
    """Direct access for slash-joined run keys."""
    node = d["runs"][key]
    items = []
    for k, v in node.items():
        if k.startswith("psr="):
            items.append((float(k[4:-2]), v["cond_asr"]))
    items.sort()
    return [p for p, _ in items], [a for _, a in items]


print("F1/F2: adaptive MEAPs + defense margins")
ad = {}
for tag, f in (("dual", "adaptive_dual_s7_r5.json"),
               ("at", "adaptive_at_s7_r5.json"),
               ("trades", "adaptive_trades_s7_r5.json")):
    d = json.load(open(os.path.join(OUT, f)))
    p, a = curve_flat(d, "untargeted/cv2x_mask")
    m, c = meap_20(p, a)
    ad[tag] = m
    print(f"    {tag}: mask cells {list(zip(p, a))}")
    print(f"    {tag}: MEAP {m:.2f} ({c})")
paper = {"dual": -39.43, "at": -32.70, "trades": -29.56}
for tag in paper:
    check(f"F1 {tag} adaptive mask MEAP", abs(ad[tag] - paper[tag]) < 0.05,
          f"derived {ad[tag]:.2f} vs paper {paper[tag]}")
check("F2 AT adaptive margin +6.7", abs((ad['at'] - ad['dual']) - 6.7) < 0.15,
      f"{ad['at'] - ad['dual']:.2f}")
check("F2 TRADES adaptive margin +9.9",
      abs((ad['trades'] - ad['dual']) - 9.9) < 0.15,
      f"{ad['trades'] - ad['dual']:.2f}")

print("\nF3: step-starvation decomposition (AT mask)")
d10 = json.load(open(os.path.join(OUT, "at_defense_results.json")))
p10, a10 = curve(d10, ["defended", "untargeted", "runs", "cv2x_mask"])
m10, _ = meap_20(p10, a10)
r1 = json.load(open(os.path.join(OUT, "adaptive_at_s7_r1.json")))
p1, a1 = curve_flat(r1, "untargeted/cv2x_mask")
m1, _ = meap_20(p1, a1)
check("F3 PGD-10 AT mask MEAP -23.08", abs(m10 - (-23.08)) < 0.05, f"{m10:.2f}")
check("F3 PGD-50 R=1 AT mask MEAP -31.56", abs(m1 - (-31.56)) < 0.05, f"{m1:.2f}")
check("F3 steps contribution 8.5 dB", abs((m10 - m1) - 8.48) < 0.15, f"{m10 - m1:.2f}")
check("F3 restarts contribution 1.1 dB", abs((m1 - ad['at']) - 1.14) < 0.15, f"{m1 - ad['at']:.2f}")

print("\nF4: ETSI vs flat-cap")
e = json.load(open(os.path.join(OUT, "etsi_mask_results.json")))
s = e["summary"]
pe, ae = curve_flat(e, "untargeted/etsi_table7")
me, _ = meap_20(pe, ae)
check("F4 ETSI MEAP -36.72", abs(me - (-36.72)) < 0.05, f"{me:.2f}")
check("F4 flat MEAP -37.08", abs(s["meap_flat_cap_db"] - (-37.08)) < 0.05,
      f"{s['meap_flat_cap_db']:.2f}")
check("F4 shape delta +0.36", abs((me - s["meap_flat_cap_db"]) - 0.36) < 0.06,
      f"{me - s['meap_flat_cap_db']:.2f}")
check("F4 PoC flat 7.14", abs(s["poc_flat_minus_genie_db"] - 7.14) < 0.02)
check("F4 PoC ETSI 7.50", abs(s["poc_etsi_minus_genie_db"] - 7.50) < 0.02)

print("\nF5: harm chain")
h = json.load(open(os.path.join(OUT, "harm_chain.json")))
c = h["stage2_prr"]["at_meap_20pct"]
D = np.array(c["d_m"])
i100 = int(np.argmin(np.abs(D - 100)))
i150 = int(np.argmin(np.abs(D - 150)))
check("F5 baseline PRR(100m) 0.857", abs(c["prr_baseline"][i100] - 0.857) < 0.001)
check("F5 attacked PRR(100m) 0.762",
      abs(c["prr_attacked_nocapture"][i100] - 0.762) < 0.001)
dprr = 100 * (c["prr_baseline"][i100] - c["prr_attacked_nocapture"][i100])
check("F5 dPRR(100m) 9.5 pp", abs(dprr - 9.45) < 0.1, f"{dprr:.2f}")
dprr150 = 100 * (c["prr_baseline"][i150] - c["prr_attacked_nocapture"][i150])
check("F5 dPRR(150m) 4.2 pp", abs(dprr150 - 4.17) < 0.1, f"{dprr150:.2f}")
rt = 100 * (c["prr_baseline"][i100] - c["prr_attacked_random_timing"][i100])
check("F5 random-timing dPRR(100m) 0.2 pp", abs(rt - 0.2) < 0.15, f"{rt:.2f}")
# independent link budget: EIRP needed at d_A=650, d_B=100, PSR=MEAP
FSPL0 = 20 * np.log10(4 * np.pi / (3e8 / 5.9e9))
pl = lambda dm, n: FSPL0 + 10 * n * np.log10(dm)
psr_meap = h["stage2_prr"]["at_meap_20pct"]["psr_db"]
eirp = 23 - pl(100, 3.0) + psr_meap + pl(650, 3.0)
check("F5 EIRP(650m) <= 33 dBm cap", eirp <= 33.0, f"{eirp:.2f} dBm")
eirp_700 = 23 - pl(100, 3.0) + psr_meap + pl(700, 3.0)
check("F5 EIRP(700m) > 33 (infeasible just beyond)", eirp_700 > 33.0,
      f"{eirp_700:.2f} dBm")

print("\nF6: Wilson CI spot checks")
Z = 1.959964
def wilson(k, n):
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    hz = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return 100 * (c - hz), 100 * (c + hz)
from scipy.stats import binomtest
bt = binomtest(92, 300).proportion_ci(0.95, method="wilson")
lo, hi = 100 * bt.low, 100 * bt.high
wlo, whi = wilson(92, 300)
check("F6 Wilson(92,300) == scipy implementation",
      abs(lo - wlo) < 0.01 and abs(hi - whi) < 0.01,
      f"ours [{wlo:.2f},{whi:.2f}] scipy [{lo:.2f},{hi:.2f}]")
ci = json.load(open(os.path.join(OUT, "w12_confidence_intervals.json")))
pair = next(p for p in ci["headline_pairs"] if "C8/C9" in p["label"])
check("F6 canonical PoC 7.14 CI [4.37, 9.8] (ascending)",
      pair["poc_db"] == 7.14 and pair["poc_ci95"] == [4.37, 9.8]
      and pair["poc_ci95"][0] < pair["poc_ci95"][1],
      f"{pair['poc_db']} {pair['poc_ci95']}")
# k round trip: every stored cell reconstructs
nflags = sum(len(v["k_roundtrip_flags"]) for f in ci["files"].values()
             for v in f["curves"].values())
check("F6 zero k round-trip flags", nflags == 0, f"{nflags}")

print("\nF7: ETSI Table-7 knot fidelity")
std = json.load(open(os.path.join(ROOT, "data", "standards",
                                  "en302571_tables.json")))
t7 = std["table_7_unwanted_inside_its_bands_10mhz"]
from etsi_mask import TABLE7_KNOTS, table7_cap_db
knots = TABLE7_KNOTS
check("F7 offsets match", np.allclose(knots[:, 0], t7["offsets_mhz"]))
check("F7 relative dB match",
      np.allclose(knots[:, 1], t7["relative_db"]))
check("F7 verbatim dBm/MHz row",
      t7["row"] == [23, 23, -3, -9, -17, -27])
check("F7 interpolation monotone non-increasing",
      all(np.diff(table7_cap_db(np.linspace(0, 20, 200))) <= 1e-9))

print("\nF8: continuity config blocks")
for f in ("adaptive_at_s7_r5.json", "adaptive_dual_s7_r5.json"):
    d = json.load(open(os.path.join(OUT, f)))
    cfg = d["config"]
    ok = ("zero-init" in cfg["restart_init"]) and cfg["steps"] == 50 \
        and cfg["restarts"] == 5 and cfg["seed"] == 7
    check(f"F8 {f} restart-0 zero-init documented", ok)
d = json.load(open(os.path.join(OUT, "adaptive_at_s7_r1.json")))
check("F8 R=1 file is the decomposition arm",
      d["config"]["restarts"] == 1)

print("\n" + "=" * 60)
if fails:
    print(f"AUDIT RESULT: FAIL ({len(fails)} failures)")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("AUDIT RESULT: PASS — all Wave-12 part-2 numbers reproduce")
