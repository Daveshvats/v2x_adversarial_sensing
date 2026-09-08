#!/usr/bin/env python3
"""Horizontal bar chart: self-overlap 5-gram containment by source (DM-1 palette)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Data from scripts/toolcheck/self_overlap_results.json (post-edit re-run, 2026-09-09)
sources = [
    "ICE2CT-2026 conference paper (published)",
    "repo README",
    "repo RESEARCH_COUNCIL",
    "repo REGULATORY_COMMENT_DRAFT",
    "paper COVER_LETTER",
    "repo ONE_PAGER",
]
vals = [0.05, 1.82, 1.33, 0.87, 0.42, 0.15]

fig, ax = plt.subplots(figsize=(10, 5.0), dpi=200, constrained_layout=True)
bars = ax.barh(sources, vals, color="#1B6B7A", height=0.62, edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
            va="center", ha="left", fontsize=10, color="#1C2A3D")
ax.axvline(3.0, color="#B13D3D", linestyle="--", linewidth=1.2)
ax.text(3.06, 5.28, "3% single-source attention level", fontsize=9,
        color="#B13D3D", va="center")
ax.set_xlabel("Share of manuscript 5-grams also found in source (%)", fontsize=11)
ax.set_title("Self-overlap scan: journal manuscript vs prior and public text",
             fontsize=13, pad=12)
ax.set_xlim(0, 3.4)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3, color="#C8DDE2")
ax.tick_params(axis="y", labelsize=10)
fig.savefig("/home/z/my-project/scripts/precheck_chart.png")
print("saved chart -> /home/z/my-project/scripts/precheck_chart.png")
