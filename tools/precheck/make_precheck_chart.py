#!/usr/bin/env python3
"""Horizontal bar chart: self-overlap 5-gram containment by source.

Data-driven: reads results/self_overlap_results.json (written by
self_overlap_scan.py), so the chart always matches the latest scan.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "self_overlap_results.json")
OUT = os.path.join(HERE, "precheck_chart.png")

LABELS = {
    "ICE2CT2026_unpublished_manuscript": "ICE2CT-2026 unpublished manuscript",
    "repo_README": "repo README",
    "repo_RESEARCH_COUNCIL": "repo RESEARCH_COUNCIL",
    "repo_REGULATORY_COMMENT_DRAFT": "repo REGULATORY_COMMENT_DRAFT",
    "repo_paper_COVER_LETTER": "paper COVER_LETTER",
    "repo_ONE_PAGER": "repo ONE_PAGER",
}

data = json.load(open(RESULTS))
data.sort(key=lambda r: r["pct_of_paper_5grams"])   # smallest bar at bottom
sources = [LABELS.get(r["source"], r["source"]) for r in data]
vals = [r["pct_of_paper_5grams"] for r in data]

fig, ax = plt.subplots(figsize=(10, 5.0), dpi=200, constrained_layout=True)
bars = ax.barh(sources, vals, color="#1B6B7A", height=0.62, edgecolor="white")
for b, v in zip(bars, vals):
    ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.2f}%",
            va="center", ha="left", fontsize=10, color="#1C2A3D")
ax.axvline(3.0, color="#B13D3D", linestyle="--", linewidth=1.2)
ax.text(3.06, len(vals) - 0.72, "3% single-source attention level", fontsize=9,
        color="#B13D3D", va="center")
ax.set_xlabel("Share of manuscript 5-grams also found in source (%)", fontsize=11)
ax.set_title("Self-overlap scan: journal manuscript vs prior and public text",
             fontsize=13, pad=12)
ax.set_xlim(0, 3.4)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3, color="#C8DDE2")
ax.tick_params(axis="y", labelsize=10)
fig.savefig(OUT)
print(f"saved chart -> {OUT}")
