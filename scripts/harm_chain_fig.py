#!/usr/bin/env python3
"""harm_chain_fig.py — two-panel harm-chain figure from results/harm_chain.json.

(a) PRR vs legit-link distance: baseline vs compliant-attacked at the MEAP
    and at PSR = -10 dB (attacker-synchronized worst case, no-capture model;
    capture model nearly identical).
(b) Required attacker EIRP vs attacker-victim distance d_A (cloaked TX at
    100 m), with the 33 dBm ITS / 36 dBm U-NII-4 EIRP caps.

Style: matches run_plot.py conventions (constrained_layout, 300 dpi,
paper/figs/). English labels (paper language).
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")

import matplotlib.font_manager as fm
for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
    if os.path.exists(f):
        fm.fontManager.addfont(f)
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

d = json.load(open(os.path.join(OUT, "harm_chain.json")))

fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), constrained_layout=True)

# ---- (a) PRR curves ----
ax = axes[0]
c = d["stage2_prr"]["at_meap_20pct"]
D = c["d_m"]
ax.plot(D, c["prr_baseline"], color="#444444", lw=1.8, ls="--",
        label="no attacker (noise-limited)")
ax.plot(D, c["prr_attacked_nocapture"], color="#c0392b", lw=1.8,
        label="attacked @ MEAP (20\\% false-idle)")
c2 = d["stage2_prr"]["at_psr_minus10"]
ax.plot(D, c2["prr_attacked_nocapture"], color="#8e44ad", lw=1.5,
        ls=":", label="attacked @ PSR $-10$ dB (38\\%)")
ax.set_xlabel("legit BSM link distance $D$ (m)")
ax.set_ylabel("packet reception ratio (PRR)")
ax.set_xlim(0, 300)
ax.set_ylim(0, 1.02)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=6.2, loc="upper right", framealpha=0.9)
ax.set_title("(a) BSM PRR, urban NLOS", fontsize=8.5)

# ---- (b) attacker EIRP feasibility ----
ax = axes[1]
s1 = d["stage1_attacker_feasibility"]
d_A = np.arange(25.0, 1000.0, 25.0)
colors = {"meap_20pct": "#c0392b", "asr38_psr": "#8e44ad",
          "asr90_psr": "#d35400"}
labels = {"meap_20pct": "MEAP ($-14.8$ dB, 20\\%)",
          "asr38_psr": "PSR $-10$ dB (38\\%)",
          "asr90_psr": "PSR $0$ dB (68\\%)"}
for tag in ("meap_20pct", "asr38_psr", "asr90_psr"):
    e = s1["psr_points"][tag]["dB=100"]["required_eirp_dbm"]
    ax.plot(d_A, e, lw=1.6, color=colors[tag], label=labels[tag])
ax.axhline(33.0, color="#2c3e50", lw=1.2, ls="--")
ax.text(960, 33.6, "33 dBm ITS EIRP cap", fontsize=6.2, ha="right",
        color="#2c3e50")
ax.axhline(36.0, color="#7f8c8d", lw=0.9, ls=":")
ax.text(960, 36.6, "36 dBm U-NII-4", fontsize=6.2, ha="right",
        color="#7f8c8d")
ax.set_xlabel("attacker--victim distance $d_A$ (m)")
ax.set_ylabel("required attacker EIRP (dBm)")
ax.set_xlim(0, 1000)
ax.grid(alpha=0.25, lw=0.4)
ax.legend(fontsize=6.2, loc="lower right", framealpha=0.9)
ax.set_title("(b) attack feasibility, cloaked TX at 100 m", fontsize=8.5)

fig.savefig(os.path.join(FIGS, "harm_chain.png"), dpi=300)
print("figure saved:", os.path.join(FIGS, "harm_chain.png"))
