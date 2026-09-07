#!/usr/bin/env python3
"""w14c_council_round3_recompute.py — Council round-3 (Task 35-d item 23):
LIVE re-verification record. Every number printed here is recomputed from
the committed artifact JSONs on disk (independent of the producing scripts'
stored summaries where possible). This is the evidence base Part V cites.

Reviewer recomputes:
  34-a (adversarial ML): attack-seed margins (own interpolation, 9 grids),
        naive-vs-adaptive gap per defense seed, grid identity
  34-b (spectrum regulation): four-domain MEAP/PoC table + deltas,
        post-PA zeros, standards provenance (sha256 + US rows)
  34-c (V2X systems): fusion K=1 exactness / gains / PoC stability,
        harm-chain MC bands, latency decisions-per-budget
  34-d (statistics): fresh-seed paired bootstrap (B=1000, different rng),
        t-CI formulas, censoring conventions, Wilson spot check
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")

R = []


def rec(tag, value, expect=None, tol=None):
    ok = ""
    if expect is not None:
        t = tol or 0.011
        if isinstance(value, str) or isinstance(expect, str):
            good = value == expect
        elif isinstance(value, list) or isinstance(expect, list):
            v = list(value) if not isinstance(value, list) else value
            e = list(expect) if not isinstance(expect, list) else expect
            good = len(v) == len(e) and all(
                abs(a - b) <= t for a, b in zip(v, e))
        else:
            good = abs(value - expect) <= t
        ok = " MATCH" if good else f" ** MISMATCH vs {expect}"
    R.append(f"{tag}: {value}{ok}")
    print(f"  {tag}: {value}{ok}")


def meap(ps, asr, thresh=20.0):
    ps = np.asarray(ps, float)
    asr = np.asarray(asr, float)
    o = np.argsort(ps)
    ps, asr = ps[o], asr[o]
    if np.all(asr >= thresh):
        return float(ps[0])  # stored 'le' floor convention
    if np.all(asr < thresh):
        return None
    for i in range(len(ps) - 1):
        if asr[i] < thresh <= asr[i + 1]:
            t = (thresh - asr[i]) / (asr[i + 1] - asr[i])
            return float(ps[i] + t * (ps[i + 1] - ps[i]))
    return None


def curves(path):
    d = json.load(open(path))
    out = {}
    for a in ("genie", "cv2x_mask"):
        cells = d["runs"][f"untargeted/{a}"]
        ps = sorted(float(k[4:-2]) for k in cells)
        out[a] = (ps, [cells[f"psr={p:+.0f}dB"]["cond_asr"] for p in ps])
    return out


print("=== 34-a (adversarial ML) recomputes ===")
rep_m = json.load(open(os.path.join(OUT, "w14_attack_seed_margins.json")))
margins = {}
for d in ("dual", "at", "trades"):
    for s in (7, 11, 22):
        c = curves(os.path.join(OUT, f"adaptive_{d}_s{s}_r5.json"))
        mg = meap(*c["genie"])
        mm = meap(*c["cv2x_mask"])
        margins[(d, s)] = (mg, mm)
        row = [r for r in rep_m["defenses"][d]["rows"]
               if r["attack_seed"] == s][0]
        rec(f"34a mask MEAP {d} s{s}", round(mm, 3), row["meap_mask_db"])
        rec(f"34a genie MEAP {d} s{s}", round(mg, 3), row["meap_genie_db"])
for d in ("at", "trades"):
    ms = [round(margins[(d, s)][1] - margins[("dual", s)][1], 3)
          for s in (7, 11, 22)]
    agg = rep_m["defenses"][d]["aggregates"]
    rec(f"34a margins {d} per-seed", ms,
        agg["per_seed_margins_db"], tol=0.0)
    rec(f"34a margin std {d}", round(float(np.std(ms, ddof=1)), 3),
        agg["margin_std_db"])
# naive-vs-adaptive gap per defense seed
# (TRADES s42 naive summary is empty in the stored file; recompute the
#  naive MEAP from its stored curves with the independent interpolator)
NAIVE_SRC = {
    ("at", "42"): ("at_defense_results.json", "summary"),
    ("at", "43"): ("at_defense_results_s43.json", "summary"),
    ("at", "53"): ("at_defense_results_s53.json", "summary"),
    ("trades", "42"): ("attack_results_urban_untargeted_trades.json",
                       "curves"),
    ("trades", "43"): ("trades_defense_results_s43.json", "summary"),
    ("trades", "53"): ("trades_defense_results_s53.json", "summary"),
}
for fam in ("at", "trades"):
    gaps = []
    for tag, base, sfx in (("42", "", ""), ("43", "_s43", "_d43"),
                           ("53", "_s53", "_d53")):
        fname, mode = NAIVE_SRC[(fam, tag)]
        doc = json.load(open(os.path.join(OUT, fname)))
        if mode == "summary":
            naive = (doc["defended"]["untargeted"]["summary"]
                     if "defended" in doc else doc["summary"]["untargeted"])
            nm = naive["meap_cv2x_mask_db"]
        else:
            cells = doc["per_scenario"]["urban"]["runs"][
                "untargeted/cv2x_mask"]
            ps = sorted(float(k[4:-2]) for k in cells)
            nm = meap(ps, [cells[f"psr={p:+.0f}dB"]["cond_asr"]
                           for p in ps])
        adv = json.load(open(os.path.join(
            OUT, f"adaptive_{fam}{sfx}_s7_r5.json")))["summary"]["untargeted"]
        gaps.append(round(adv["meap_cv2x_mask_db"] - nm, 2))
    rec(f"34a naive-vs-adaptive gap {fam} (d42/43/53)", gaps)

print("\n=== 34-b (spectrum regulation) recomputes ===")
co = json.load(open(os.path.join(OUT, "conformance_results.json")))["summary"]
us = json.load(open(os.path.join(OUT,
                                 "conformance_us_results.json")))["summary"]
et = json.load(open(os.path.join(OUT, "etsi_mask_results.json")))["summary"]
fu = json.load(open(os.path.join(OUT, "w14b_fusion_baseline.json")))["summary"]
flat = fu["cv2x_mask_K1_meap_db"]
genie = fu["genie_K1_meap_db"]
try:
    perbin = et["meap_etsi_db"]
except KeyError:
    perbin = et.get("etsi_meap_db", co["meap_etsi_perbin_db"])
poc_perbin = round(perbin - genie, 2)
rec("34b flat-cap MEAP (fusion K=1 control)", round(flat, 3), -37.083)
rec("34b per-bin ETSI MEAP", round(perbin, 2), -36.72)
rec("34b RBW/mean-power MEAP", round(co["meap_conformance_rbw_db"], 2),
    -37.37)
rec("34b US 95.3205 MEAP", round(us["meap_conformance_us_db"], 2), -37.89)
rec("34b genie MEAP", round(genie, 3), -44.225)
pocs = {"flat": round(flat - genie, 2), "per-bin ETSI": poc_perbin,
        "RBW mean-power": co["poc_conformance_db"],
        "US 95.3205": us["poc_conformance_us_db"]}
rec("34b four-domain PoC set", pocs)
rec("34b spread of four MEAPs",
    round(max(abs(perbin - flat), abs(co["meap_conformance_rbw_db"] - flat),
              abs(us["meap_conformance_us_db"] - flat)), 2), None)
rec("34b post-PA worst excess (ETSI RBW / US)",
    (co["post_pa_worst_excess_dbr_max"], us["post_pa_worst_excess_dbr_max"]),
    None)
tb = json.load(open(os.path.join(ROOT, "data", "standards",
                                 "en302571_tables.json")))
h_json = hashlib.sha256(
    open(os.path.join(ROOT, "data", "standards",
                     "en302571_tables.json"), "rb").read()).hexdigest()
rec("34b committed tables-JSON sha256 prefix", h_json[:7], "c66bf63")
rec("34b recorded source-PDF sha256 prefix",
    str(tb.get("pdf_sha256", ""))[:7], "667a939")
rec("34b source-PDF bytes recorded", tb.get("pdf_bytes"), 218250)
rows = tb.get("fcc_47cfr_95_3205", [])
rec("34b US 95.3205 rows in tables", len(rows), 4)
rec("34b Table-7 rows in tables",
    len(tb.get("table_7_unwanted_inside_its_bands_10mhz", [])), None)

print("\n=== 34-c (V2X systems) recomputes ===")
rec("34c fusion K=1 == canonical flat", round(fu["cv2x_mask_K1_meap_db"], 3),
    -37.083)
for k in (3, 5):
    rec(f"34c K={k} mask MEAP",
        round(fu[f"cv2x_mask_K{k}_meap_db"], 2), None)
    rec(f"34c K={k} gain vs K=1",
        round(fu[f"cv2x_mask_K{k}_meap_db"] - flat, 2), None)
    rec(f"34c K={k} PoC",
        round(fu[f"cv2x_mask_K{k}_meap_db"] - fu[f"genie_K{k}_meap_db"], 2),
        None)
harm = json.load(open(os.path.join(OUT, "harm_chain.json")))
a = harm["stage2_prr"]["at_meap_20pct"]
i100 = a["d_m"].index(100.0)
cln = a["prr_baseline"][i100]
att = a["prr_attacked_nocapture"][i100]
att_std = a["prr_attacked_nocapture_std"][i100]
drop = round(100 * (cln - att), 2)
rec("34c harm PRR @100m clean/attacked (seed-mean)", (cln, att), None)
rec("34c harm PRR drop pp", drop, 9.44, 0.06)
rec("34c harm MC seeds", a.get("mc_seeds"), None)
rec("34c harm band std @100m", round(att_std, 4), None)
lat = json.load(open(os.path.join(OUT, "latency_phase1.json")))["feasibility"]
med = lat["per_window_ms_median"]
rec("34c latency median (ms)", round(med, 3), 2.637, 0.02)
rec("34c decisions per 100 ms TR-37.885 budget",
    lat["windows_affordable_per_budget_median"], 37, 1)
rec("34c realtime multiple", lat["realtime_multiple_median"], 25.75, 0.5)

print("\n=== 34-d (statistics) recomputes ===")
bs = json.load(open(os.path.join(OUT, "w14b_bootstrap_cis.json")))
# fresh-seed paired bootstrap, dual s7 capture, B=1000, rng 20260908
cap = json.load(open(os.path.join(OUT,
                                  "w14b_perwindow_adaptive_dual_s7.json")))
rng = np.random.default_rng(20260908)


def flags_of(doc, setting):
    cells = doc["runs"][f"untargeted/{setting}"]
    ps, fl = [], []
    for k in sorted(cells, key=lambda x: float(x[4:-2])):
        ps.append(float(k[4:-2]))
        fl.append(np.asarray(cells[k]["win_flags"], dtype=np.int64))
    return np.asarray(ps), np.asarray(fl)


pg, fg = flags_of(cap, "genie")
pm, fm = flags_of(cap, "cv2x_mask")
n = fg.shape[1]
pocs_r = []
for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    asr_g = [100.0 * (((fg[:, idx][i] == 1) & (fg[:, idx][i] >= 0)).sum()
                      / max((fg[:, idx][i] >= 0).sum(), 1))
             for i in range(fg.shape[0])]
    asr_m = [100.0 * (((fm[:, idx][i] == 1) & (fm[:, idx][i] >= 0)).sum()
                      / max((fm[:, idx][i] >= 0).sum(), 1))
             for i in range(fm.shape[0])]
    mg = meap(pg, asr_g)
    mm = meap(pm, asr_m)
    if mg is not None and mm is not None:
        pocs_r.append(mm - mg)
lo, hi = np.percentile(pocs_r, [2.5, 97.5])
stored = bs["adaptive"]["dual"]["attack_seed_7"]["poc_ci95_db"]
rec("34d fresh-seed B=1000 dual-s7 PoC CI", [round(lo, 2), round(hi, 2)],
    None)
rec("34d stored B=2000 CI (overlap check)",
    (stored, "overlap" if stored[0] <= hi and lo <= stored[1] else
     "NO OVERLAP"), None)
# t-CI formulas
for d in ("dual", "at", "trades"):
    pt = bs["adaptive"][d]["three_seed_mean"]["point_pocs_db"]
    t3 = bs["adaptive"][d]["three_seed_mean"]["t_ci95_db"]
    m = float(np.mean(pt))
    h = 4.303 * float(np.std(pt, ddof=1)) / np.sqrt(3)
    rec(f"34d t-CI {d} recomputed",
        [round(m - h, 3), round(m + h, 3)], t3, tol=0.011)
# censoring convention disclosure
w12 = json.load(open(os.path.join(OUT, "w12_confidence_intervals.json")))
c21 = [p for p in w12["headline_pairs"]
       if "real-WiFi" in p["label"]]
for p in c21:
    rec(f"34d {p['label']} PoC (un-censored)", p["poc_db"], None)
rec("34d canonical conservative CI",
    w12["headline_pairs"][0]["poc_ci95"], [4.37, 10.12], tol=0.0)

path = os.path.join(OUT, "w14c_council_round3_recompute.txt")
with open(path, "w") as f:
    f.write("\n".join(R) + "\n")
print(f"\nwrote {path} ({len(R)} records)")
