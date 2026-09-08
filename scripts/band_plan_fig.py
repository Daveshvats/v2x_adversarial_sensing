#!/usr/bin/env python3
"""band_plan_fig.py — Figure 1 source for "The Price of Compliance".

Post-FCC 5.9 GHz band plan as seen by the boundary-straddling sensing window.

Frequencies (absolute GHz) mirror src/waveforms.py BAND_PLAN exactly:
  - Regulatory blocks (FCC 20-164): U-NII-4 unlicensed [5.850, 5.895];
    dedicated ITS / C-V2X [5.895, 5.925]; boundary at 5.895.
  - Sensing window: [5.885, 5.905] (2048 samples @ 20 MHz).
  - C-V2X PC5 SC-FDMA:  [5.8955, 5.9045]  (450 x 20 kHz, center +5 MHz rel.)
  - 802.11p OFDM:       52 subcarriers, 156.25 kHz spacing, center 5.900,
                        DC nulled  -> stems at 5.900 +/- k*0.15625, k=1..26.
  - Wi-Fi U-NII-4:      26 in-window subcarriers, 312.5 kHz spacing, channel
                        center 5.885 (10 MHz below window center), DC nulled
                        -> stems at 5.885 + k*0.3125, k=1..26 (upper half of a
                        20 MHz channel [5.875, 5.895]); out-of-window half
                        drawn as a dashed ghost clipped by the window edge.
  - Noise floor: flat low line across the window.

Style: Okabe-Ito colorblind-safe palette, 300 dpi, 7.0 x 3.4 in (double
column), English labels, no chart junk. Stylized PSD axis (no ticks).

Run:  python scripts/band_plan_fig.py
Out:  paper/figs/band_plan.png + paper/figs/band_plan.pdf
      (deterministic: numpy default_rng(7) fixes the stylized stem heights)
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_DIR = os.path.join(ROOT, "paper", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------- constants
F_MIN, F_MAX = 5.850, 5.925          # axis (GHz)
UNII_LO, UNII_HI = 5.850, 5.895      # U-NII-4 unlicensed block
ITS_LO, ITS_HI = 5.895, 5.925        # dedicated ITS / C-V2X block
BOUND = 5.895                        # 5895 MHz boundary
WIN_LO, WIN_HI = 5.885, 5.905        # 20 MHz sensing window

PC5_LO, PC5_HI = 5.8955, 5.9045      # C-V2X PC5 (SC-FDMA, 9 MHz)
P11P_C = 5.900                         # 802.11p center (52 subcarr., DC nulled)
P11P_DF = 0.15625e-3                   # 156.25 kHz spacing, in GHz
WIFI_C = 5.885                         # Wi-Fi U-NII-4 channel center (10 MHz
WIFI_DF = 0.3125e-3                    #  below window center); 312.5 kHz, GHz
WIFI_CH_LO = 5.875                     # nominal 20 MHz channel lower edge

# Okabe-Ito palette
C_PC5, C_11P, C_WIFI = "#0072B2", "#D55E00", "#009E73"
C_NOISE, C_DARK, C_MID = "#555555", "#1A1A1A", "#333333"
BG_UNII, BG_ITS = "#F3EBDD", "#DFE9F2"   # regulatory block shades

# stylized PSD levels (axis units, y in [0, 1])
Y_TOP_LABEL, Y_SUB_LABEL = 0.975, 0.925   # regulatory labels (2 lines)
Y_BRACKET, Y_BRACKET_TICK = 0.030, 0.075  # sensing-window bracket
Y_WIN_LABEL = -0.045
Y_NOISE = 0.090
Y_WIFI_BASE, H_WIFI = 0.240, (0.100, 0.220)
Y_PC5_LO, Y_PC5_HI = 0.160, 0.400
Y_11P_BASE, H_11P = 0.500, (0.100, 0.220)

rng = np.random.default_rng(7)  # deterministic stylized stem heights

# ------------------------------------------------------------------ figure
fig, ax = plt.subplots(figsize=(7.0, 3.4))
fig.subplots_adjust(left=0.065, right=0.985, top=0.97, bottom=0.14)

# --- regulatory background blocks ----------------------------------------
ax.axvspan(UNII_LO, BOUND, color=BG_UNII, zorder=0)
ax.axvspan(BOUND, ITS_HI, color=BG_ITS, zorder=0)
ax.text((UNII_LO + BOUND) / 2, Y_TOP_LABEL, "U-NII-4 unlicensed",
        ha="center", va="center", fontsize=7.5, color=C_MID)
ax.text((UNII_LO + BOUND) / 2, Y_SUB_LABEL, "5.850--5.895 GHz",
        ha="center", va="center", fontsize=7.0, color=C_MID)
ax.text((BOUND + ITS_HI) / 2, Y_TOP_LABEL, "Dedicated ITS / C-V2X",
        ha="center", va="center", fontsize=7.5, color=C_MID)
ax.text((BOUND + ITS_HI) / 2, Y_SUB_LABEL, "5.895--5.925 GHz",
        ha="center", va="center", fontsize=7.0, color=C_MID)

# --- 5895 MHz boundary (strong vertical line) ----------------------------
ax.plot([BOUND, BOUND], [0.0, 1.01], color=C_DARK, lw=1.8, zorder=4)
ax.text(BOUND - 0.0007, 0.56, "5895 MHz boundary (FCC 20-164)",
        rotation=90, ha="center", va="center", fontsize=7.0,
        color=C_MID, zorder=5)

# --- 20 MHz sensing window: dashed edges + bottom bracket ----------------
for xw in (WIN_LO, WIN_HI):
    ax.plot([xw, xw], [Y_BRACKET_TICK, 0.86], color="#777777",
            lw=1.0, ls=(0, (4, 3)), zorder=3)
ax.plot([WIN_LO, WIN_HI], [Y_BRACKET, Y_BRACKET], color=C_MID,
        lw=1.2, zorder=3)
for xw in (WIN_LO, WIN_HI):
    ax.plot([xw, xw], [Y_BRACKET, Y_BRACKET_TICK], color=C_MID,
            lw=1.2, zorder=3)
ax.text((WIN_LO + WIN_HI) / 2, Y_WIN_LABEL,
        "20 MHz sensing window (2048 samples @ 20 MHz)",
        ha="center", va="center", fontsize=7.5, color=C_MID)

# --- C-V2X PC5: flat-topped block with comb texture ----------------------
ax.add_patch(Rectangle((PC5_LO, Y_PC5_LO), PC5_HI - PC5_LO, Y_PC5_HI - Y_PC5_LO,
                       facecolor=C_PC5, alpha=0.38, edgecolor="none", zorder=2))
comb = np.arange(PC5_LO, PC5_HI + 1e-9, 0.0002)       # every 0.2 MHz
ax.vlines(comb, Y_PC5_LO, Y_PC5_HI, colors=C_PC5, lw=0.45, alpha=0.35, zorder=2)
ax.plot([PC5_LO, PC5_HI], [Y_PC5_HI, Y_PC5_HI], color=C_PC5, lw=1.7, zorder=3)
ax.plot([PC5_LO, PC5_HI], [Y_PC5_LO, Y_PC5_LO], color=C_PC5, lw=0.9, zorder=3)

# --- 802.11p: 52 raised subcarrier stems (co-channel with PC5) ------------
ks = np.concatenate([np.arange(-26, 0), np.arange(1, 27)])   # DC nulled
p11p_x = P11P_C + ks * P11P_DF
h11 = H_11P[0] + (H_11P[1] - H_11P[0]) * rng.random(52)
ax.vlines(p11p_x, Y_11P_BASE, Y_11P_BASE + h11, colors=C_11P, lw=1.1, zorder=3)
ax.plot([p11p_x[0], p11p_x[-1]], [Y_11P_BASE, Y_11P_BASE],
        color=C_11P, lw=1.0, alpha=0.85, zorder=3)

# --- Wi-Fi U-NII-4: 26 in-window stems + dashed ghost of out-of-window half
kw = np.arange(1, 27)                                       # DC nulled at 5.885
wifi_x = WIFI_C + kw * WIFI_DF
hw = H_WIFI[0] + (H_WIFI[1] - H_WIFI[0]) * rng.random(26)
ax.vlines(wifi_x, Y_WIFI_BASE, Y_WIFI_BASE + hw, colors=C_WIFI, lw=1.2, zorder=3)
ax.plot([wifi_x[0], wifi_x[-1]], [Y_WIFI_BASE, Y_WIFI_BASE],
        color=C_WIFI, lw=1.0, alpha=0.85, zorder=3)
# ghost: lower half of the 20 MHz channel, clipped by the window edge (5.885)
ax.add_patch(Rectangle((WIFI_CH_LO, Y_WIFI_BASE), WIN_LO - WIFI_CH_LO,
                       (Y_WIFI_BASE + H_WIFI[1]) - Y_WIFI_BASE,
                       facecolor=C_WIFI, alpha=0.06, edgecolor=C_WIFI,
                       lw=1.0, linestyle=(0, (3, 3)), zorder=1))
kw_out = -np.arange(1, 27)
wifi_out_x = WIFI_C + kw_out * WIFI_DF
hw_out = H_WIFI[0] + (H_WIFI[1] - H_WIFI[0]) * rng.random(26)
ax.vlines(wifi_out_x, Y_WIFI_BASE, Y_WIFI_BASE + hw_out, colors=C_WIFI,
          lw=0.8, alpha=0.28, linestyles=(0, (2, 2)), zorder=1)
ax.text((WIFI_CH_LO + WIN_LO) / 2, Y_WIFI_BASE + H_WIFI[1] + 0.045,
        "out-of-window (not received)", ha="center", va="center",
        fontsize=6.5, color="#02695A", zorder=2)

# --- noise floor across the window ----------------------------------------
ax.plot([WIN_LO, WIN_HI], [Y_NOISE, Y_NOISE], color=C_NOISE, lw=1.4, zorder=3)

# --- legend ----------------------------------------------------------------
handles = [
    Patch(facecolor=C_PC5, alpha=0.45, edgecolor=C_PC5, lw=1.2,
          label="C-V2X PC5 (SC-FDMA)"),
    Line2D([0], [0], color=C_11P, lw=1.4, label="802.11p (52 subcarriers)"),
    Line2D([0], [0], color=C_WIFI, lw=1.4,
           label="Wi-Fi U-NII-4 (26 subcarr.)"),
    Line2D([0], [0], color=C_NOISE, lw=1.4, label="noise floor"),
]
ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.002, 0.88),
          fontsize=7, frameon=True, framealpha=0.95,
          edgecolor="#BBBBBB", borderpad=0.5, handlelength=1.4,
          labelspacing=0.35)

# --- axes cosmetics ---------------------------------------------------------
ax.set_xlim(F_MIN, F_MAX)
ax.set_ylim(-0.09, 1.03)
ax.set_xticks(np.arange(5.850, 5.9251, 0.010))
ax.set_xticklabels([f"{f:.3f}" for f in np.arange(5.850, 5.9251, 0.010)],
                   fontsize=8)
ax.set_yticks([])
ax.set_xlabel("Frequency (GHz)", fontsize=9)
ax.set_ylabel("PSD", fontsize=9, rotation=0, labelpad=18)
ax.yaxis.set_label_coords(-0.035, 0.62)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="x", length=3, color="#888888")

for ext, dpi in (("png", 300), ("pdf", None)):
    fig.savefig(os.path.join(OUT_DIR, f"band_plan.{ext}"), dpi=dpi)
    print(f"wrote {os.path.join(OUT_DIR, 'band_plan.' + ext)}")
print("stems: 11p =", len(p11p_x), "| wifi in =", len(wifi_x),
      "| wifi out =", len(wifi_out_x), "| pc5 comb =", len(comb))
