#!/usr/bin/env python3
"""run_plot.py — merge the per-scenario/mode attack JSONs, recompute the
summary metrics (MEAP with censoring, Price of Compliance), and draw the
money figure: conditional ASR vs PSR (dB), genie vs compliant attacker.

Merges every results/attack_results_{scenario}_{mode}.json it finds (the
run_attack.py output convention), so extra scenarios (rural) or seeds can be
added incrementally without renaming files.
"""
import sys, os, json, glob, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from attack_mask import meap_curve, price_of_compliance

RES = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")
os.makedirs(FIGS, exist_ok=True)

# canonical baseline sweeps ONLY (excludes seed11/sensitivity/pgd50 experiment
# files and the merged output itself)
CANON = re.compile(
    r"attack_results_(urban|highway|rural)_(untargeted|targeted_noise)\.json$")

# --- merge all per-scenario/mode files ---
merged = None
for p in sorted(glob.glob(os.path.join(RES, "attack_results_*_*.json"))):
    base = os.path.basename(p)
    if not CANON.match(base):
        continue
    d = json.load(open(p))
    scen = list(d["per_scenario"].keys())[0]
    if merged is None:
        merged = {"experiment": d.get("experiment"),
                  "config": d.get("config"), "per_scenario": {}}
    if scen in merged["per_scenario"]:
        merged["per_scenario"][scen]["runs"].update(
            d["per_scenario"][scen]["runs"])
        merged["per_scenario"][scen].setdefault("summary", {}).update(
            d["per_scenario"][scen].get("summary", {}))
    else:
        merged["per_scenario"][scen] = d["per_scenario"][scen]

if merged is None:
    raise SystemExit("no attack results found")

scen_list = list(merged["per_scenario"].keys())
mode_list = []
for scen in scen_list:
    mode_list += [m for m in merged["per_scenario"][scen].get("summary", {})
                  if m not in mode_list]

# --- recompute summary metrics (censoring-aware) ---
psrs = merged["config"]["psr_db"]
for scen, sd in merged["per_scenario"].items():
    for mode in ("untargeted", "targeted_noise"):
        if f"{mode}/genie" not in sd["runs"]:
            continue
        g = [sd["runs"][f"{mode}/genie"][f"psr={p:+.0f}dB"]["cond_asr"]
             for p in psrs]
        m = [sd["runs"][f"{mode}/cv2x_mask"][f"psr={p:+.0f}dB"]["cond_asr"]
             for p in psrs]
        mg, cg = meap_curve(psrs, g, 20.0)
        mm, cm = meap_curve(psrs, m, 20.0)
        poc, cnotes = price_of_compliance(psrs, g, psrs, m, 20.0)
        sd.setdefault("summary", {})[mode] = {
            "meap_genie_db": mg, "meap_genie_censor": cg,
            "meap_cv2x_mask_db": mm, "meap_cv2x_mask_censor": cm,
            "price_of_compliance_db": poc, "poc_censor": cnotes,
            "genie_curve": g, "mask_curve": m,
        }

with open(os.path.join(RES, "attack_results_merged.json"), "w") as f:
    json.dump(merged, f, indent=2)
print("merged scenarios:", scen_list)
for scen, sd in merged["per_scenario"].items():
    for mode, s in sd.get("summary", {}).items():
        print(f"  {scen}/{mode}: MEAP genie {s['meap_genie_db']} dB "
              f"({s['meap_genie_censor']}) | MEAP mask "
              f"{s['meap_cv2x_mask_db']} dB ({s['meap_cv2x_mask_censor']}) "
              f"| PoC {s['price_of_compliance_db']} dB "
              f"({s['poc_censor']})")

# --- money figure ---
C_GENIE, C_MASK = "#c0392b", "#2471a3"
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), constrained_layout=True)
titles = {"untargeted": "Untargeted (any misclassification)",
          "targeted_noise": "Targeted: cloak active TX as Noise"}
styles = {"urban": ("-", "o"), "highway": ("--", "s"), "rural": (":", "^")}
poc_vals = [sd["summary"][m]["price_of_compliance_db"]
            for sd in merged["per_scenario"].values()
            for m in sd.get("summary", {}) if m == "untargeted"]
poc_txt = (f"{np.mean(poc_vals):.1f}" if poc_vals and all(
    v is not None for v in poc_vals) else "n/a")
for ax, mode in zip(axes, ("untargeted", "targeted_noise")):
    for scen in scen_list:
        sd = merged["per_scenario"][scen]
        if mode not in sd.get("summary", {}):
            continue
        ls, mk = styles.get(scen, ("-", "o"))
        ax.plot(psrs, sd["summary"][mode]["genie_curve"], ls, marker=mk,
                color=C_GENIE, label=f"genie attacker ({scen})", ms=4)
        ax.plot(psrs, sd["summary"][mode]["mask_curve"], ls, marker=mk,
                color=C_MASK,
                label=f"C-V2X-mask compliant ({scen})", ms=4)
    ax.set_xlabel("Attack power budget  PSR (dB)")
    ax.set_ylabel("Conditional ASR (%)")
    ax.set_title(titles[mode], fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-3, 103)
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle(f"Emission-mask compliance shifts the attack curve by "
             f"~{poc_txt} dB — but does not close it", fontsize=11)
fig.savefig(os.path.join(FIGS, "price_of_compliance.png"), dpi=200)
print("figure saved:", os.path.join(FIGS, "price_of_compliance.png"))
