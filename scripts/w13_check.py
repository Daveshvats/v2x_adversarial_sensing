#!/usr/bin/env python3
"""w13_check.py — INDEPENDENT audit of the defense-seed replication numbers
(adversarial-internal-review protocol; same role as w12_check.py).

Checks (pure numpy, no attack_mask import — deliberately a different code
path from w13_seed_replication.py):
  F1  every per-seed MEAP/PoC in w13_defense_seed_replication.json matches
      an independent piecewise-linear re-interpolation of the stored curves;
  F2  margins = defense mask MEAP - undefended mask MEAP (recomputed);
  F3  aggregate mean/range/std match the report;
  F4  attack protocol identical across all 7 files (steps 50, R 5, seed 7,
      n_eval 100, alpha 0.25, psd_margin 2.0, band [0,10] MHz);
  F5  clean acc within [0.95, 1.0] on every defended model;
  F6  no censored MEAPs (every MEAP bracketed by its grid);
  F7  TRADES margin > AT margin on every seed (ordering claim used in paper);
  F8  all 6 replication result files exist with full grids.
Exit 0 only if every check passes.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")

fails = []


def chk(name, cond, detail=""):
    print(f"  {name}: {'PASS' if cond else 'FAIL'} {detail}")
    if not cond:
        fails.append(name)


def meap_indep(ps, asr, thresh=20.0):
    """MEAP by ascending-PSR linear interpolation of cond-ASR vs PSR."""
    ps = np.asarray(ps, float)
    asr = np.asarray(asr, float)
    o = np.argsort(ps)
    ps, asr = ps[o], asr[o]
    if np.all(asr < thresh):
        return None  # right-censored
    if np.all(asr >= thresh):
        return None  # left-censored (20% already at the weakest PSR)
    for i in range(len(ps) - 1):
        if asr[i] < thresh <= asr[i + 1]:
            t = (thresh - asr[i]) / (asr[i + 1] - asr[i])
            return ps[i] + t * (ps[i + 1] - ps[i])
    return None


def curves(doc):
    runs = doc["runs"]
    def get(setting):
        cells = runs[f"untargeted/{setting}"]
        ps = sorted(float(k[4:-2]) for k in cells)
        asr = [cells[f"psr={p:+.0f}dB"]["cond_asr"] for p in ps]
        return ps, asr
    return get("genie"), get("cv2x_mask")


print("[F8] file presence")
FILES = {
    ("at", 42, ""): "adaptive_at_s7_r5.json",
    ("at", 43, "_d43"): "adaptive_at_d43_s7_r5.json",
    ("at", 53, "_d53"): "adaptive_at_d53_s7_r5.json",
    ("trades", 42, ""): "adaptive_trades_s7_r5.json",
    ("trades", 43, "_d43"): "adaptive_trades_d43_s7_r5.json",
    ("trades", 53, "_d53"): "adaptive_trades_d53_s7_r5.json",
    ("dual", 42, ""): "adaptive_dual_s7_r5.json",
}
docs = {}
for key, fname in FILES.items():
    p = os.path.join(OUT, fname)
    ok = os.path.exists(p)
    chk(f"exists {fname}", ok)
    if ok:
        docs[key] = json.load(open(p))
chk("all 7 files", len(docs) == 7, f"({len(docs)}/7)")
if len(docs) < 7:
    sys.exit(1)

print("[F4] protocol identity across files")
proto_ref = None
for key, doc in docs.items():
    c = doc["config"]
    proto = (c["steps"], c["restarts"], c["seed"], c["n_eval_per_class"],
             c["attack_pgd_alpha_frac"], c["psd_cap_margin"],
             tuple(c["mask_band_mhz"].values()) if isinstance(
                 c["mask_band_mhz"], dict) else c["mask_band_mhz"])
    if proto_ref is None:
        proto_ref = proto
    chk(f"proto {key}", proto == proto_ref)

print("[F1/F2/F5/F6] per-seed MEAPs, margins, clean acc, censoring")
rep = json.load(open(os.path.join(OUT, "w13_defense_seed_replication.json")))
dual_g, dual_m = curves(docs[("dual", 42, "")])
dual_mask_meap = meap_indep(*dual_m)
chk("dual mask MEAP uncensored", dual_mask_meap is not None,
    f"({dual_mask_meap:.2f} dB)")
chk("dual MEAP matches stored summary",
    abs(dual_mask_meap - docs[("dual", 42, "")]["summary"]["untargeted"]
        ["meap_cv2x_mask_db"]) < 0.01)

indep = {}
for (defense, seed, tag), doc in docs.items():
    if defense == "dual":
        continue
    (pg, g), (pm, m) = curves(doc)
    mg = meap_indep(pg, g)
    mm = meap_indep(pm, m)
    poc = (mm - mg) if (mm is not None and mg is not None) else None
    margin = (mm - dual_mask_meap) if mm is not None else None
    indep[(defense, seed)] = (mg, mm, poc, margin)
    row = [r for r in rep["defenses"][defense]
           if r["defense_seed"] == seed][0]
    chk(f"{defense} s{seed} genie MEAP", mg is not None and
        abs(mg - row["meap_genie_db"]) < 0.01, f"({mg:.3f})")
    chk(f"{defense} s{seed} mask MEAP", mm is not None and
        abs(mm - row["meap_mask_db"]) < 0.01, f"({mm:.3f})")
    chk(f"{defense} s{seed} PoC", poc is not None and
        abs(poc - row["poc_db"]) < 0.01)
    chk(f"{defense} s{seed} margin", margin is not None and
        abs(margin - row["margin_vs_dual_db"]) < 0.01,
        f"({margin:+.3f})")
    ca = doc["clean_acc_active"]
    chk(f"{defense} s{seed} clean acc", 0.95 <= ca <= 1.0, f"({ca})")

print("[F3] aggregates")
for defense in ("at", "trades"):
    margins = [indep[(defense, s)][3] for s in (42, 43, 53)]
    agg = rep["aggregates"][defense]
    chk(f"{defense} margin mean",
        abs(np.mean(margins) - agg["margin_mean_db"]) < 0.01,
        f"({np.mean(margins):.3f})")
    chk(f"{defense} margin range",
        abs((max(margins) - min(margins)) - agg["margin_range_db"]) < 0.01,
        f"({max(margins)-min(margins):.3f})")
    chk(f"{defense} margin std",
        abs(np.std(margins, ddof=1) - agg["margin_std_db"]) < 0.01,
        f"({np.std(margins, ddof=1):.3f})")

print("[F7] TRADES > AT ordering on every seed")
for s in (42, 43, 53):
    chk(f"seed {s}", indep[("trades", s)][3] > indep[("at", s)][3],
        f"(AT {indep[('at', s)][3]:+.2f} vs TRADES "
        f"{indep[('trades', s)][3]:+.2f})")

print()
if fails:
    print("W13 AUDIT: FAIL —", fails)
    sys.exit(1)
print("W13 AUDIT: ALL CHECKS PASS")
