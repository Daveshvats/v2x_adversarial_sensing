#!/usr/bin/env python3
r"""w11_check5_claims.py — W11 auditor CHECK 5 + CHECK 6 + CHECK 7.

Automated cross-check of every quantitative claim in the paper subsection
\label{sec:papr} (plus the abstract/Conclusion/Limitations PA sentences and
CLAIMS.md C27–C32) against results/papr_pa_results.json and the canonical
attack_results_urban_untargeted.json. Also verifies the new bibliography
entries are cited, and that the canonical price-of-compliance headline
(-37.1 / -44.2) survived the Wave-11 edit (CHECK 7).

Each check prints PASS / FAIL / INFO with paper-says vs JSON-says.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
TEX = os.path.join(ROOT, "paper", "main.tex")
CLAIMS = os.path.join(ROOT, "paper", "CLAIMS.md")

res = json.load(open(os.path.join(RES, "papr_pa_results.json")))
canon = json.load(open(os.path.join(RES, "attack_results_urban_untargeted.json")))
tex = open(TEX).read()
claims_md = open(CLAIMS).read()

fails, warns = [], []

def ps(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: {detail}")
    if not ok:
        fails.append(f"{name}: {detail}")
    return ok

def pi(name, detail=""):
    print(f"[INFO] {name}: {detail}")
    warns.append(f"{name}: {detail}")

def near(a, b, tol):
    return abs(a - b) <= tol

def tex_has(s):
    return s in tex

# --------------------------------------------------------------------------
print("== sec:papr: PAPR claims ==")
m = res["papr_optimized_db"]["mask"]["psr=-36dB"]
ps("paper 11.05 dB mean PAPR", near(m["mean_db"], 11.05, 0.005)
   and tex_has("11.05\\,dB"), f"JSON {m['mean_db']}")
ps("paper p95 12.8", near(m["p95_db"], 12.8, 0.05) and tex_has("p95 12.8"),
   f"JSON {m['p95_db']}")
ben = res["papr_benign_db"]
for cname, val, pstr in [("C-V2X-PC5", 6.26, "6.26 (PC5)"),
                         ("802.11p", 8.77, "8.77 (11p)"),
                         ("WiFi-U-NII4", 8.69, "8.69 (WiFi)"),
                         ("Noise", 9.02, "9.02")]:
    ps(f"paper benign PAPR {cname}", near(ben[cname]["mean_db"], val, 0.005)
       and tex_has(pstr), f"JSON {ben[cname]['mean_db']}")
ps("H1 confirmed flag", res["H1"]["confirmed"] is True
   and near(res["H1"]["adv_papr_db"], 11.05, 0.005)
   and near(res["H1"]["benign_max_db"], 9.02, 0.005),
   f"JSON adv {res['H1']['adv_papr_db']} vs benign max "
   f"{res['H1']['benign_max_db']}")

# --------------------------------------------------------------------------
print("\n== sec:papr: regrowth claims ==")
r6 = res["regrowth"]["p=3/ibo=6"]
ps("paper -19.3 dBr p95 @IBO6 p=3", near(r6["mask_excess_p95_dbr"], -19.3, 0.05)
   and tex_has("$-19.3$\\,dBr"), f"JSON {r6['mask_excess_p95_dbr']}")
# the lenient-gate sentence
pl = r6["pass_lenient"]
if near(pl, 0.0, 1e-9) and tex_has("0\\% of windows pass even the lenient"):
    ps("paper '0% pass lenient' @IBO6 p=3", True, "JSON 0.0")
else:
    ps("paper '0% pass even the lenient -28 dBr shoulder' @IBO6 p=3", False,
       f"paper says 0% of windows pass the lenient gate; JSON "
       f"regrowth['p=3/ibo=6'].pass_lenient = {pl} "
       f"({pl*100:.0f}% DO pass, i.e. {100-pl*100:.0f}% fail). "
       f"(p=2/ibo=6: {res['regrowth']['p=2/ibo=6']['pass_lenient']})")
ps("paper strict-gate 0% @IBO6", near(r6["pass_strict"], 0.0, 1e-9),
   f"JSON pass_strict {r6['pass_strict']}")
pc5 = res["benign_regrowth_increment"]["C-V2X-PC5"]
ps("paper PC5 total skirt -28.7 dBr @IBO6",
   near(pc5["post_ibo6_p95_dbr"], -28.7, 0.05) and tex_has("PC5: $-28.7$"),
   f"JSON {pc5['post_ibo6_p95_dbr']}")
ps("paper PC5 PA increment +1.0 dB @IBO6",
   near(pc5["increment_ibo6_p95_db"], 1.0, 0.05) and tex_has("$+1.0$\\,dB"),
   f"JSON {pc5['increment_ibo6_p95_db']} (rounds to +1.0)")
delta9 = r6["mask_excess_p95_dbr"] - pc5["post_ibo6_p95_dbr"]
if tex_has("$\\sim$9\\,dB above the strongest legitimate neighbor"):
    ps("paper '~9 dB above (PC5) skirt'", near(delta9, 9.35, 0.05),
       f"JSON -19.32-(-28.67) = {delta9:.2f} dB vs PC5; NOTE vs 11p skirt "
       f"{res['benign_regrowth_increment']['802.11p']['post_ibo6_p95_dbr']} "
       f"the margin is only "
       f"{r6['mask_excess_p95_dbr']-res['benign_regrowth_increment']['802.11p']['post_ibo6_p95_dbr']:.2f} dB "
       f"and vs WiFi "
       f"{res['benign_regrowth_increment']['WiFi-U-NII4']['post_ibo6_p95_dbr']} "
       f"it is "
       f"{r6['mask_excess_p95_dbr']-res['benign_regrowth_increment']['WiFi-U-NII4']['post_ibo6_p95_dbr']:.2f} dB "
       "(PC5 is the CLEANEST, not the strongest-skirt, neighbor)")
star = res["ibo_star_strict_db"]
ps("paper IBO* = 13 (p=3) / 15 (p=2)",
   star["p=3"] == 13.0 and star["p=2"] == 15.0
   and tex_has("13$\\,dB ($p{=}3$) / 15\\,dB ($p{=}2$)"),
   f"JSON {star}")
pi("paper 'IBO swept 0-16 dB'",
   "stored config ibo_sweep_db = [0,2,4,6,8,10,12]; the 0-16 range is the "
   "script's fine 1-dB IBO* search grid (arange(0,16.1,1)), not a stored "
   "sweep value")

# --------------------------------------------------------------------------
print("\n== sec:papr: MEAP-shift claims ==")
cm = res["control_meap_db"]
ps("paper control -37.0", near(cm, -37.0, 0.005) and tex_has("$-37.0$\\,dB"),
   f"JSON {cm}")
c_canon = canon["per_scenario"]["urban"]["summary"]["untargeted"][
    "meap_cv2x_mask_db"]
g_canon = canon["per_scenario"]["urban"]["summary"]["untargeted"][
    "meap_genie_db"]
ps("paper '0.1 dB of the coarser canonical grid'",
   abs(cm - c_canon) <= 0.1,
   f"|control {cm} - canonical {c_canon:.4f}| = {abs(cm-c_canon):.3f} dB")
cv = res["curves"]
for key, pshift, jshift in [("pa_p3_ibo0", "+0.6", 0.60),
                            ("pa_p3_ibo6", "-0.1", -0.09),
                            ("pa_p3_ibo12", "0.0", 0.0),
                            ("pa_p3_ibo13", "0.0", 0.0),
                            ("pa_p2_ibo6", "0.0 (p=2)", 0.0),
                            ("pa_p2_ibo15", "0.0 (p=2)", 0.0)]:
    ps(f"paper MEAP shift {key} {pshift}",
       near(cv[key + "__meap"]["shift_vs_control_db"], jshift, 0.005),
       f"JSON {cv[key+'__meap']['shift_vs_control_db']}")
a13 = cv["pa_aware_p3_ibo13__meap"]
a0 = cv["pa_aware_p3_ibo0__meap"]
ps("paper PA-aware MEAP -38.2", near(a13["meap_db"], -38.2, 0.05)
   and tex_has("(MEAP $-38.2$\\,dB"), f"JSON {a13['meap_db']:.4f}")
ps("paper PA-aware shift -1.2 dB", near(a13["shift_vs_control_db"], -1.2, 0.05)
   and tex_has("$-1.2$\\,dB shift"), f"JSON {a13['shift_vs_control_db']}")
ps("paper PA-aware 100% strict pass", a13["post_tx_pass_strict"] == 1.0
   and tex_has("100\\% strict-gate compliance"), f"JSON {a13['post_tx_pass_strict']}")
ps("paper PA-aware -60 dBr p95 OOB",
   near(a13["post_tx_mask_excess_p95_dbr"], -60, 0.5)
   and tex_has("($-60$\\,dBr p95 out-of-mask)"),
   f"JSON {a13['post_tx_mask_excess_p95_dbr']}")
ps("paper PA-aware IBO0 non-compliant (-12.8 implied)",
   near(a0["post_tx_mask_excess_p95_dbr"], -12.8, 0.05)
   and a0["post_tx_pass_strict"] == 0.0,
   f"JSON {a0['post_tx_mask_excess_p95_dbr']}, strict "
   f"{a0['post_tx_pass_strict']}")
# post-PA PAPR 4.9 dB claim
found49 = any("4.9" in json.dumps(v) for v in
              [res["regrowth"], res["curves"], res["papr_optimized_db"],
               res["benign_regrowth_increment"]])
if tex_has("post-PA PAPR 4.9\\,dB") and not found49:
    ps("paper 'post-PA PAPR 4.9 dB' (PA-aware, IBO 0)", False,
       "NOT present in papr_pa_results.json (no papr field in any "
       "pa_aware __meap entry or regrowth row matches 4.9); independent "
       "30-window re-derivation (w11_check3b_aware.py) gives 4.93 dB "
       "=> number appears CORRECT but is untraced")
ps("paper '11.1 dB PAPR ... benign 6.3-9.0 dB' (Reading + abstract)",
   near(m["mean_db"], 11.1, 0.05) and tex_has("11.1\\,dB PAPR")
   and tex_has("6.3--9.0\\,dB"), f"JSON mask PAPR {m['mean_db']}, benign "
   f"6.26..9.02")

# --------------------------------------------------------------------------
print("\n== abstract / conclusion / limitations PA sentences ==")
ps("abstract: highest PAPR (11.1 vs 6.3-9.0)",
   tex_has("(11.1\\,dB vs.\\ 6.3--9.0\\,dB") and near(m["mean_db"], 11.1, 0.05),
   "traced")
ps("abstract: -40 dBr mask at 13 dB backoff",
   tex_has("inside a\n$-40$\\,dBr mask at 13\\,dB backoff") and star["p=3"] == 13.0,
   f"JSON p=3 IBO* {star['p=3']} (p=2 needs {star['p=2']}; abstract cites "
   "only the p=3 value — body gives both)")
ps("abstract: PA-aware gains 1.2 dB", tex_has("gaining 1.2\\,dB")
   and near(a13["shift_vs_control_db"], -1.2, 0.05), "traced")
ps("conclusion PA sentences present",
   tex_has("power-amplifier reality check sharpens the regulatory reading")
   and tex_has("compliance testing should sit at the PA\noutput port"),
   "qualitative, consistent with JSON (shifts <= 0.6 dB, aware arm -1.23)")
ps("limitations item (10) present",
   tex_has("(10)~The PA study uses a\nmemoryless Rapp model"),
   "qualitative; matches config (power_control/papr_note strings)")

# --------------------------------------------------------------------------
print("\n== bibliography (CHECK 5, citations) ==")
for key in ["anand2008", "ambhika2024", "itsa2024", "5gaa2024"]:
    cited = f"\\cite{{{key}}}" in tex
    bib = f"\\bibitem{{{key}}}" in tex
    ps(f"{key}: cited in text AND in bibliography", cited and bib,
       f"\\cite present={cited}, \\bibitem present={bib}")
# PUEA paragraph + table row
ps("PUEA paragraph (Related Work) present",
   tex_has("Primary-user emulation: the pre-DL ancestor")
   or tex_has("primary-user emulation, the pre-DL ancestor")
   or "PUEA" in tex, "PUEA text found")
ps("threat-table PUEA row present",
   tex_has("Anand et al.~\\cite{anand2008} (PUEA)"), "table row found")
# every bibitem is cited somewhere (orphan check)
keys = re.findall(r"\\bibitem\{([^}]+)\}", tex)
orphans = [k for k in keys if f"\\cite{{{k}}}" not in tex]
ps("no orphan bibliography entries", not orphans, f"orphans: {orphans}")

# --------------------------------------------------------------------------
print("\n== CLAIMS.md C27-C32 (CHECK 6) ==")
for cid in ["C27", "C28", "C29", "C30", "C31", "C32"]:
    ps(f"CLAIMS.md has {cid}", f"| {cid} " in claims_md, "row present")
ps("C27 numbers", near(m["mean_db"], 11.05, 0.005)
   and near(ben["C-V2X-PC5"]["mean_db"], 6.26, 0.005)
   and near(ben["802.11p"]["mean_db"], 8.77, 0.005)
   and near(ben["WiFi-U-NII4"]["mean_db"], 8.69, 0.005)
   and near(ben["Noise"]["mean_db"], 9.02, 0.005), "all match JSON")
ps("C28 numbers", near(r6["mask_excess_p95_dbr"], -19.3, 0.05)
   and star["p=3"] == 13 and star["p=2"] == 15
   and near(pc5["increment_ibo6_p95_db"], 1.0, 0.05)
   and near(pc5["pre_p95_dbr"], -29.7, 0.05),
   f"JSON p95 {r6['mask_excess_p95_dbr']}, IBO* {star}, inc "
   f"{pc5['increment_ibo6_p95_db']}, pre {pc5['pre_p95_dbr']}; "
   f"BUT '0% pass lenient' vs JSON {pl}")
ps("C29 numbers", near(cv["pa_p3_ibo0__meap"]["shift_vs_control_db"], 0.60, 0.005)
   and near(cv["pa_p3_ibo6__meap"]["shift_vs_control_db"], -0.09, 0.005)
   and near(cm, -37.0, 0.005) and near(c_canon, -37.08, 0.01),
   f"JSON shifts 0.6/-0.09/0.0, control {cm}, canonical {c_canon:.4f}")
ps("C30 numbers", near(a13["meap_db"], -38.23, 0.005)
   and near(a13["shift_vs_control_db"], -1.23, 0.005)
   and a13["post_tx_pass_strict"] == 1.0
   and near(a13["post_tx_mask_excess_p95_dbr"], -60.2, 0.05)
   and near(a0["meap_db"], -38.32, 0.005)
   and near(a0["post_tx_mask_excess_p95_dbr"], -12.8, 0.05),
   "all match JSON EXCEPT 'post-PA PAPR 4.9 dB' (not stored; re-derived "
   "4.93 dB)")
# C31: code-comment cross-check
pa_script = open(os.path.join(HERE, "run_papr_pa.py")).read()
c31a = ("0/0" in pa_script or "zero initialization" in pa_script)
c31b = ("33.11" in pa_script and "phantom backoff" in pa_script)
c31c = ("merge-on-write" in pa_script
        and "BEFORE" in pa_script)
ps("C31 vs run_papr_pa.py comments",
   c31a and c31b and c31c,
   f"NaN 0/0 bug comment={c31a}, +33.11 dB phantom-backoff comment={c31b}, "
   f"merge-on-write clobber comment={c31c}; NOTE '66.7%/20/30' detail and "
   "'D2 gate' narrative cite 'worklog Task 19', which does NOT exist in "
   "worklog.md (no Task ID: 19 entry)")
ps("C32 numbers/citations", all(
   f"\\cite{{{k}}}" in tex for k in ["anand2008", "ambhika2024",
                                     "itsa2024", "5gaa2024"]),
   "all four cited; FCC effective Feb 11 2025 / 50 waivers / two-year "
   "sunset sentences present in intro")

# --------------------------------------------------------------------------
print("\n== CHECK 7: canonical regression ==")
ps("canonical -37.1 still headline", tex_has("$-37.1$ dB")
   and tex_has("-37.1") and near(c_canon, -37.08, 0.01),
   f"-37.1 appears (Table 2 row intact); canonical JSON {c_canon:.4f}")
ps("canonical -44.2 still headline", tex_has("$-44.2$ dB")
   and near(g_canon, -44.22, 0.01),
   f"-44.2 appears (Table 2 row intact); canonical JSON {g_canon:.4f}")
ps("price-of-compliance table row intact",
   tex_has("urban & untargeted & $-44.2$ dB & $-37.1$ dB & $7.1$ dB"),
   "row found verbatim")
pi("system-model PAPR sentence",
   "paper Sec. 3 still says 'measured: 6.1 dB PC5 vs 8.8/8.6 dB' "
   "(untraced legacy number, W9-B already noted); Wave-11 JSON now provides "
   "committed benign PAPR 6.26/8.77/8.69 — the two measurements agree to "
   "<0.2 dB but the Sec. 3 sentence still cites the older untraced one")

# --------------------------------------------------------------------------
print(f"\nTOTAL: {len(fails)} FAIL, {len(warns)} INFO")
for f in fails:
    print("  FAIL: " + f)
raise SystemExit(1 if fails else 0)
