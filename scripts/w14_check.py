#!/usr/bin/env python3
"""w14_check.py — INDEPENDENT audit of the Wave-14 artifacts
(adversarial-internal-review protocol; same role as w12/w13_check.py).

Audited artifacts and checks (pure numpy — deliberately a different code
path from the producing scripts; MEAPs re-interpolated here from raw
curves and per-window flags):
  A  attack-seed replication grids adaptive_{dual,at,trades}_s{11,22}_r5.json
     A1 full canonical per-arm grids; A2 win_flags 300/cell, values in
        {-1,0,1}; A3 flag-derived cond-ASR == stored cond-ASR (2dp);
        A4 independent MEAP/PoC match stored summaries; A5 protocol config
        (PGD-50 x R5, seed 11/22, n_eval 100, alpha 0.25, psd 2.0);
        A6 no censoring; A7 grid identity across seeds incl. canonical s7.
  B  seed-7 per-window captures w14b_perwindow_adaptive_*_s7.json
     B1 presence + full grids; B2 continuity_all_match; B3 capture
        cond-ASR == canonical file cond-ASR (exact); B4 flag-derived ASR;
        B5 independent PoC from flags == canonical summary PoC; B6 clean
        acc match.
  C  w14_attack_seed_margins.json — checks all true; per-seed MEAPs,
     margins (mask def - mask dual, seed-matched) and aggregates match
     independent recompute; n=3 per defense.
  D  w14b_bootstrap_cis.json — canonical PGD-10 point PoCs + 3-seed t-CI
     match independent recompute from the per-window captures; adaptive
     point PoCs per (defense, seed) and 3-seed t-CIs match; continuity
     flags true; bootstrap-CI sanity.
  E  PGD-10 captures w14b_perwindow_s{7,123,456}.json — continuity flags.
  F  fusion baseline (C43) — K=1 control == canonical flat-cap/genie
     MEAPs; K3/K5 mask MEAP > K1; PoC stable across K in [6.0, 7.6].
  G  conformance artifacts (C41/C42) — internal delta arithmetic, shared
     references equal across files, post-PA 0.00 dBr, no censoring, and
     the four pinned headline MEAPs.
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
    """MEAP by ascending-PSR linear interpolation of cond-ASR vs PSR
    (independent of attack_mask.meap_curve). Mirrors the stored censoring
    convention: ASR >= threshold already at the lowest grid PSR returns the
    grid floor (left-censored "le"); never-crossing returns None ("ge").
    The censored-floor case is exactly the s123/s456 genie arms of the
    canonical PGD-10 captures (disclosed in w14b_bootstrap_cis.json)."""
    ps = np.asarray(ps, float)
    asr = np.asarray(asr, float)
    o = np.argsort(ps)
    ps, asr = ps[o], asr[o]
    if np.all(asr >= thresh):
        return float(ps[0])
    if np.all(asr < thresh):
        return None
    for i in range(len(ps) - 1):
        if asr[i] < thresh <= asr[i + 1]:
            t = (thresh - asr[i]) / (asr[i + 1] - asr[i])
            return float(ps[i] + t * (ps[i + 1] - ps[i]))
    return None


def curve_of(doc, setting):
    cells = doc["runs"][f"untargeted/{setting}"]
    ps = sorted(float(k[4:-2]) for k in cells)
    asr = [cells[f"psr={p:+.0f}dB"]["cond_asr"] for p in ps]
    return ps, asr


def flags_curve(doc, setting):
    """(ps, asr, n_flags) recomputed from win_flags."""
    cells = doc["runs"][f"untargeted/{setting}"]
    ps, asr = [], []
    lens = set()
    for k in sorted(cells, key=lambda s: float(s[4:-2])):
        c = cells[k]
        f = np.asarray(c["win_flags"], dtype=np.int64)
        lens.add(f.size)
        elig = f >= 0
        a = 100.0 * ((f == 1) & elig).sum() / max(elig.sum(), 1)
        ps.append(float(k[4:-2]))
        asr.append(round(float(a), 2))
    return ps, asr, lens


GRIDS = {  # canonical per-arm protocol (mirrors the s7 files)
    "dual": {"genie": [-50, -45, -40, -35, -30],
             "cv2x_mask": [-45, -40, -35, -30, -25, -20]},
    "at": {"genie": [-45, -40, -35, -30, -25],
           "cv2x_mask": [-35, -30, -25, -20, -15, -10]},
    "trades": {"genie": [-45, -40, -35, -30, -25],
               "cv2x_mask": [-35, -30, -25, -20, -15, -10, -5]},
}
DEFENSES = ("dual", "at", "trades")
SEEDS = (11, 22)

print("[A] attack-seed replication grids (item 12)")
ind = {}  # (defense, seed) -> (mg, mm, poc) independent
for d in DEFENSES:
    for s in SEEDS:
        path = os.path.join(OUT, f"adaptive_{d}_s{s}_r5.json")
        tag = f"{d} s{s}"
        if not os.path.exists(path):
            chk(f"A1 {tag} present", False)
            continue
        doc = json.load(open(path))
        ok_grids = all(
            sorted(float(k[4:-2]) for k in doc["runs"][f"untargeted/{a}"])
            == GRIDS[d][a] for a in ("genie", "cv2x_mask"))
        chk(f"A1 {tag} canonical grids", ok_grids)
        ok_flags, ok_asr = True, True
        for a in ("genie", "cv2x_mask"):
            for k, c in doc["runs"][f"untargeted/{a}"].items():
                f = c.get("win_flags")
                if f is None or len(f) != 300 or \
                        not set(f) <= {-1, 0, 1}:
                    ok_flags = False
                fa = np.asarray(f)
                if abs(round(100.0 * ((fa == 1) & (fa >= 0)).sum()
                             / max((fa >= 0).sum(), 1), 2)
                       - c["cond_asr"]) > 0.011:
                    ok_asr = False
        chk(f"A2 {tag} win_flags 300/cell", ok_flags)
        chk(f"A3 {tag} flag-ASR == stored", ok_asr)
        pg, g = curve_of(doc, "genie")
        pm, m = curve_of(doc, "cv2x_mask")
        mg = meap_indep(pg, g)
        mm = meap_indep(pm, m)
        st = doc["summary"]["untargeted"]
        chk(f"A4 {tag} MEAPs match", mg is not None and mm is not None
            and abs(mg - st["meap_genie_db"]) < 0.01
            and abs(mm - st["meap_cv2x_mask_db"]) < 0.01)
        ind[(d, s)] = (mg, mm, mm - mg)
        cfg = doc["config"]
        chk(f"A5 {tag} protocol", cfg["steps"] == 50 and
            cfg["restarts"] == 5 and cfg["seed"] == s and
            cfg["n_eval_per_class"] == 100 and
            abs(cfg["attack_pgd_alpha_frac"] - 0.25) < 1e-9 and
            abs(cfg["psd_cap_margin"] - 2.0) < 1e-9)
        uncens = (g[0] < 20.0 and m[0] < 20.0 and g[-1] >= 20.0
                  and m[-1] >= 20.0)
        chk(f"A6 {tag} uncensored", uncens)

# A7: grid identity across seeds incl. canonical s7 (and collect s7 refs)
s7 = {}
for d in DEFENSES:
    doc = json.load(open(os.path.join(OUT, f"adaptive_{d}_s7_r5.json")))
    ok = all(sorted(float(k[4:-2])
                    for k in doc["runs"][f"untargeted/{a}"]) == GRIDS[d][a]
             for a in ("genie", "cv2x_mask"))
    chk(f"A7 {d} s7 canonical grids (identity across seeds)", ok)
    pg, g = curve_of(doc, "genie")
    pm, m = curve_of(doc, "cv2x_mask")
    s7[d] = (meap_indep(pg, g), meap_indep(pm, m))
    st = doc["summary"]["untargeted"]
    chk(f"A7 {d} s7 stored MEAPs match", s7[d][0] is not None and
        abs(s7[d][0] - st["meap_genie_db"]) < 0.01 and
        abs(s7[d][1] - st["meap_cv2x_mask_db"]) < 0.01)

print("[B] seed-7 adaptive per-window captures")
cap = {}
for d in DEFENSES:
    tag = f"{d}"
    path = os.path.join(OUT, f"w14b_perwindow_adaptive_{d}_s7.json")
    if not os.path.exists(path):
        chk(f"B1 {tag} capture present", False)
        continue
    doc = json.load(open(path))
    canon = json.load(open(os.path.join(OUT, f"adaptive_{d}_s7_r5.json")))
    n_cells = sum(len(v) for v in doc["runs"].values())
    need = sum(len(v) for v in GRIDS[d].values())
    chk(f"B1 {tag} {n_cells}/{need} cells", n_cells == need)
    chk(f"B2 {tag} continuity_all_match",
        doc["checks"].get("continuity_all_match") is True)
    ok_b3, ok_b4 = True, True
    for a in ("genie", "cv2x_mask"):
        for k, c in doc["runs"][f"untargeted/{a}"].items():
            rc = canon["runs"][f"untargeted/{a}"][k]
            if c["cond_asr"] != rc["cond_asr"]:
                ok_b3 = False
            f = np.asarray(c["win_flags"])
            if f.size != 300 or not set(c["win_flags"]) <= {-1, 0, 1}:
                ok_b4 = False
            elif abs(round(100.0 * ((f == 1) & (f >= 0)).sum()
                           / max((f >= 0).sum(), 1), 2)
                     - c["cond_asr"]) > 0.011:
                ok_b4 = False
    chk(f"B3 {tag} capture ASR == canonical (exact)", ok_b3)
    chk(f"B4 {tag} flags consistent", ok_b4)
    pg, g, _ = flags_curve(doc, "genie")
    pm, m, _ = flags_curve(doc, "cv2x_mask")
    mg, mm = meap_indep(pg, g), meap_indep(pm, m)
    st = canon["summary"]["untargeted"]
    chk(f"B5 {tag} PoC from flags == canonical",
        mg is not None and mm is not None and
        abs(mm - mg - st["price_of_compliance_db"]) < 0.01,
        f"{None if mg is None or mm is None else round(mm - mg, 3)}")
    chk(f"B6 {tag} clean acc", doc["checks"].get("clean_acc_matches_ref")
        is True)
    cap[d] = (mm - mg, doc["checks"].get("continuity_all_match") is True)

print("[C] w14_attack_seed_margins.json")
rep = json.load(open(os.path.join(OUT, "w14_attack_seed_margins.json")))
chk("C1 report checks all true", all(v is True for v in
                                     rep["checks"].values()))
ok_rows, ok_marg, ok_agg = True, True, True
for d in DEFENSES:
    rows = rep["defenses"][d]["rows"]
    chk(f"C5 {d} n=3 seeds", len(rows) == 3)
    for r in rows:
        s = r["attack_seed"]
        mine = ind.get((d, s)) if s in SEEDS else (
            s7[d][0], s7[d][1], s7[d][1] - s7[d][0])
        if mine is None or abs(mine[0] - r["meap_genie_db"]) > 0.01 or \
                abs(mine[1] - r["meap_mask_db"]) > 0.01 or \
                abs(mine[2] - r["poc_db"]) > 0.01:
            ok_rows = False
        dm = s7[d if d == "dual" else "dual"][1] if s == 7 else \
            ind[("dual", s)][1]
        if abs((r["meap_mask_db"] - dm) - r["margin_vs_dual_db"]) > 0.011:
            ok_marg = False
    agg = rep["defenses"][d]["aggregates"]
    mvals = [r["margin_vs_dual_db"] for r in rows]
    pvals = [r["poc_db"] for r in rows]
    if abs(float(np.mean(mvals)) - agg["margin_mean_db"]) > 0.01 or \
            abs(float(np.ptp(mvals)) - agg["margin_range_db"]) > 0.01 or \
            abs(float(np.std(mvals, ddof=1)) - agg["margin_std_db"]) > 0.01 or \
            abs(float(np.mean(pvals)) - agg["poc_mean_db"]) > 0.01 or \
            abs(float(np.ptp(pvals)) - agg["poc_range_db"]) > 0.01:
        ok_agg = False
chk("C2 per-seed MEAPs/PoCs match", ok_rows)
chk("C3 margins = mask(def) - mask(dual, seed-matched)", ok_marg)
chk("C4 aggregates match", ok_agg)

print("[D] w14b_bootstrap_cis.json")
bs = json.load(open(os.path.join(OUT, "w14b_bootstrap_cis.json")))
ok_d1 = True
cpocs = []
for s in (7, 123, 456):
    p = os.path.join(OUT, f"w14b_perwindow_s{s}.json")
    doc = json.load(open(p))
    pg, g, _ = flags_curve(doc, "genie")
    pm, m, _ = flags_curve(doc, "cv2x_mask")
    mg, mm = meap_indep(pg, g), meap_indep(pm, m)
    poc = None if (mg is None or mm is None) else mm - mg
    got = bs["canonical"][f"model_seed_{s}"]["poc_db"]
    if poc is None or abs(poc - got) > 0.011:
        ok_d1 = False
    cpocs.append(poc)
chk("D1 canonical PGD-10 point PoCs match", ok_d1)
t3 = bs["canonical"]["three_seed_mean"]
m = float(np.mean(cpocs))
h = 4.303 * float(np.std(cpocs, ddof=1)) / np.sqrt(3)
chk("D2 canonical 3-seed t-CI matches",
    abs(m - t3["point_mean_db"]) < 0.01 and
    abs(m - h - t3["t_ci95_db"][0]) < 0.01 and
    abs(m + h - t3["t_ci95_db"][1]) < 0.01)
ok_d3, ok_d4 = True, True
for d in DEFENSES:
    e = bs["adaptive"].get(d, {})
    if not all(k in e for k in ("attack_seed_7", "attack_seed_11",
                                "attack_seed_22", "three_seed_mean")):
        ok_d3 = False
        continue
    if abs(e["attack_seed_7"]["poc_db"] - cap[d][0]) > 0.011:
        ok_d3 = False
    if abs(e["attack_seed_11"]["poc_db"] - ind[(d, 11)][2]) > 0.011 or \
            abs(e["attack_seed_22"]["poc_db"] - ind[(d, 22)][2]) > 0.011:
        ok_d3 = False
    pt = [e[f"attack_seed_{s}"]["poc_db"] for s in (7, 11, 22)]
    tm = bs["adaptive"][d]["three_seed_mean"]
    mm_ = float(np.mean(pt))
    hh = 4.303 * float(np.std(pt, ddof=1)) / np.sqrt(3)
    if abs(mm_ - tm["point_mean_db"]) > 0.01 or \
            abs(mm_ - hh - tm["t_ci95_db"][0]) > 0.01 or \
            abs(mm_ + hh - tm["t_ci95_db"][1]) > 0.01:
        ok_d3 = False
    if not e["attack_seed_7"].get("continuity_all_match"):
        ok_d4 = False
    lo, hi = e["attack_seed_7"].get("poc_ci95_db", [9e9, -9e9])
    if not (lo - 2.0 <= e["attack_seed_7"]["poc_db"] <= hi + 2.0) or \
            (hi - lo) > 8.0:
        ok_d4 = False
chk("D3 adaptive point PoCs + 3-seed t-CIs match", ok_d3)
chk("D4 continuity + bootstrap-CI sanity", ok_d4)

print("[E] PGD-10 per-window captures")
for s in (7, 123, 456):
    doc = json.load(open(os.path.join(OUT, f"w14b_perwindow_s{s}.json")))
    chk(f"E1 s{s} continuity_all_match",
        doc["checks"].get("continuity_all_match") is True)

print("[F] fusion baseline (C43)")
fu = json.load(open(os.path.join(OUT, "w14b_fusion_baseline.json")))["summary"]
chk("F1 K=1 control == flat-cap MEAP",
    abs(fu["cv2x_mask_K1_meap_db"] + 37.0833) < 0.01,
    f"{fu['cv2x_mask_K1_meap_db']:.3f}")
chk("F1b K=1 genie == canonical genie",
    abs(fu["genie_K1_meap_db"] + 44.2248) < 0.01,
    f"{fu['genie_K1_meap_db']:.3f}")
chk("F2 K3 > K1 and K5 > K1 (fusion gain)",
    fu["cv2x_mask_K3_meap_db"] > fu["cv2x_mask_K1_meap_db"] and
    fu["cv2x_mask_K5_meap_db"] > fu["cv2x_mask_K1_meap_db"],
    f"K3 +{fu['cv2x_mask_K3_meap_db'] - fu['cv2x_mask_K1_meap_db']:.1f} dB")
pocs_k = [fu["cv2x_mask_K%d_meap_db" % k] - fu["genie_K%d_meap_db" % k]
          for k in (1, 3, 5)]
chk("F3 PoC stable across K in [6.0, 7.6]",
    all(6.0 <= p <= 7.6 for p in pocs_k), f"{[round(p,2) for p in pocs_k]}")

print("[G] conformance artifacts (C41/C42)")
co = json.load(open(os.path.join(OUT, "conformance_results.json")))["summary"]
us = json.load(open(os.path.join(OUT,
                                 "conformance_us_results.json")))["summary"]
chk("G1 ETSI-RBW MEAP pinned", abs(co["meap_conformance_rbw_db"]
                                   + 37.3678) < 0.01)
chk("G2 US 95.3205 MEAP pinned", abs(us["meap_conformance_us_db"]
                                     + 37.8949) < 0.01)
chk("G3 shared references equal across files",
    abs(co["meap_flat_cap_db"] - us["meap_flat_cap_db"]) < 1e-9 and
    abs(co["meap_etsi_perbin_db"] - us["meap_etsi_perbin_db"]) < 1e-9 and
    abs(co["meap_genie_db"] - us["meap_genie_db"]) < 1e-9)
chk("G4 delta arithmetic",
    abs((us["meap_conformance_us_db"] - co["meap_flat_cap_db"]) -
        us["delta_us_minus_flat_db"]) < 1e-3 and
    abs((co["meap_conformance_rbw_db"] - co["meap_flat_cap_db"]) -
        co["delta_conf_minus_flat_db"]) < 1e-3)
chk("G5 post-PA 0.00 dBr everywhere",
    co["post_pa_worst_excess_dbr_max"] == 0.0 and
    us["post_pa_worst_excess_dbr_max"] == 0.0)
chk("G6 PoCs (6.86 ETSI / 6.33 US)",
    abs(co["poc_conformance_db"] - 6.86) < 0.01 and
    abs(us["poc_conformance_us_db"] - 6.33) < 0.01)
chk("G7 no censoring",
    co["meap_conformance_censor"] is None and
    us["meap_conformance_us_censor"] is None)

print(f"\n[w14_check] {len(fails)} FAIL" + ("S" if len(fails) != 1 else "")
      if fails else f"\n[w14_check] ALL PASS")
sys.exit(0 if not fails else 1)
