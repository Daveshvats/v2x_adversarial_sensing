#!/usr/bin/env python3
"""run_margin_stratified.py — margin-stratified reporting + detection
baseline (council P1-c, reviewer 21-c item 5).

Preempts the referee objection "MEAP measures the weakest 20% of a
100%-accuracy CNN's windows": the paper's own R3 insight (real captures
expose decision-margin fragility) is applied to the SYNTHETIC headline.

Protocol (identical to attack_results_merged.json, urban untargeted,
mask setting): PGD-10 zero-init, alpha 0.25, psd_margin 2.0, band
[0,10] MHz, attack seed 7, 100 active windows/class, PSR grid
-45..+10 dB, budgets = clean received-signal window energy.

Adds:
  1. Clean-decision margin per eligible window (top1 - top2 logit gap,
     clean pass). Eligible windows split into tertiles (low/mid/high).
  2. cond-ASR per margin tertile per PSR + per-tertile MEAP(20%).
  3. Confidence-gate detection baseline: max-softmax on clean vs attacked
     windows per PSR -> AUROC + TPR at FPR=1% (the simplest deployable
     adversarial-input detector; a Zhao-style DDB would be stronger).

Output: results/w13_margin_stratified.json (incremental per cell).
"""
import os
import sys
import time
import json

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(2)

from waveforms import ATTACK_BANDS  # noqa: E402
from receiver import FrontEnd, DualStreamModel  # noqa: E402
from attack_mask import waveform_pgd, meap_curve  # noqa: E402
from run_attack import build_eval_set  # noqa: E402

OUT = os.path.join(ROOT, "results")
RES = os.path.join(OUT, "w13_margin_stratified.json")
PSR_GRID = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0,
            -5.0, 0.0, 5.0, 10.0]
SEED = 7
N_PER_CLASS = 100


def max_softmax(logits):
    s = torch.softmax(logits, dim=1)
    return s.max(dim=1).values


def auroc(scores_pos, scores_neg):
    """AUROC of score separating pos (attacked) from neg (clean)."""
    s = np.concatenate([scores_pos, scores_neg])
    lab = np.concatenate([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[lab == 1].sum()
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((r_pos - n1 * (n1 + 1) / 2) / (n1 * n0))


def tpr_at_fpr(scores_pos, scores_neg, fpr=0.01, lower=False):
    """TPR of threshold set so that FPR <= fpr on the clean scores.
    lower=True flags LOW scores as attacked (confidence-drop gate)."""
    if lower:
        thr = np.quantile(scores_neg, fpr)
        return float((scores_pos <= thr).mean())
    thr = np.quantile(scores_neg, 1 - fpr)
    return float((scores_pos >= thr).mean())


def main():
    ck = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                    map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])

    x_rx, y, h_a, p_clean = build_eval_set("urban", N_PER_CLASS, SEED)
    y_np = y.numpy()
    with torch.no_grad():
        logits = []
        for i in range(0, x_rx.size(0), 256):
            logits.append(model.forward_wave(x_rx[i:i + 256], frontend))
        logits = torch.cat(logits)
        clean_preds = logits.argmax(1)
    eligible = (clean_preds == y).numpy()
    sm_clean = max_softmax(logits).numpy()

    # decision margin: top1 - top2 logit gap on the CLEAN pass (numpy)
    top2 = logits.topk(2, dim=1).values
    margin = (top2[:, 0] - top2[:, 1]).numpy()

    # tertiles of the eligible margin distribution
    m_el = margin[eligible]
    q33, q66 = np.quantile(m_el, [1 / 3, 2 / 3])
    q33, q66 = float(q33), float(q66)
    stratum = np.full(len(margin), -1, dtype=int)
    stratum[eligible & (margin <= q33)] = 0        # low margin
    stratum[eligible & (margin > q33) & (margin <= q66)] = 1
    stratum[eligible & (margin > q66)] = 2         # high margin
    n_per_str = [int((stratum == s).sum()) for s in range(3)]
    assert n_per_str[0] > 0 and n_per_str[2] > 0, \
        f"tertile split degenerate: {n_per_str}"
    print(f"[margins] eligible {eligible.sum()}/{len(y_np)}  tertile cuts "
          f"{q33:.3f}/{q66:.3f}  per-tertile {n_per_str}", flush=True)

    results = {
        "experiment": "Margin-stratified cond-ASR + confidence-gate "
                      "detection (council P1-c item 5)",
        "config": {
            "protocol": "identical to attack_results_merged.json urban/"
                        "untargeted/cv2x_mask: PGD-10 zero-init, alpha 0.25, "
                        "psd_margin 2.0, band [0,10] MHz, seed 7, "
                        "n=100/class",
            "psr_db": PSR_GRID,
            "margin_def": "top1 - top2 logit gap on the clean pass "
                          "(eligible windows only)",
            "strata": "tertiles of the eligible margin distribution",
            "tertile_cuts": [round(float(q33), 4), round(float(q66), 4)],
            "n_eligible": int(eligible.sum()),
            "n_per_tertile": n_per_str,
            "detector": "max-softmax confidence gate, both directions "
                        "(high-conf and low-conf tails) at 1% FPR; "
                        "attacked = positive, clean = negative; "
                        "direction-free AUROC = max(A, 1-A)",
        },
        "clean_acc_active": round(float(eligible.mean()), 4),
        "per_psr": {},
    }
    if os.path.exists(RES):
        results = json.load(open(RES))

    band = ATTACK_BANDS["cv2x_attacker"]
    for psr in PSR_GRID:
        key = f"psr={psr:+.0f}dB"
        if key in results["per_psr"]:
            continue
        t0 = time.time()
        p_budget = p_clean * (10 ** (psr / 10.0))
        out = waveform_pgd(model, frontend, x_rx, h_a, y, p_budget,
                           steps=10, band=band, targeted=False,
                           alpha_frac=0.25, psd_margin=2.0,
                           return_delta=True)
        # waveform_pgd returns the ATTACKED logits (fresh no-grad forward)
        lg_adv = out["logits"]
        sm_adv = max_softmax(lg_adv).numpy()
        adv_preds = out["preds"].numpy()

        flipped = eligible & (adv_preds != y_np)
        cell = {
            "cond_asr_all": round(100 * flipped.sum() / eligible.sum(), 2),
        }
        for s, name in enumerate(("low", "mid", "high")):
            sel = stratum == s
            cell[f"cond_asr_{name}"] = round(
                100 * (flipped & sel).sum() / max(sel.sum(), 1), 2)
        # detector: max-softmax confidence gate, BOTH directions reported
        # (attacked windows can be MORE confident at high power — CE pushes
        # hard to another class — or LESS confident at low power; a deployed
        # gate is free to threshold either tail at 1% FPR)
        sp = sm_adv
        sn = sm_clean
        a_hi = auroc(sp, sn)
        a_lo = 1.0 - a_hi
        cell["detector_auroc_direction_free"] = round(max(a_hi, a_lo), 4)
        cell["detector_auroc_high_dir"] = round(a_hi, 4)
        cell["detector_tpr_at_fpr1pct_high_dir"] = round(
            tpr_at_fpr(sp, sn, 0.01, lower=False), 4)
        cell["detector_tpr_at_fpr1pct_low_dir"] = round(
            tpr_at_fpr(sp, sn, 0.01, lower=True), 4)
        cell["detector_tpr_at_fpr1pct_best"] = round(max(
            cell["detector_tpr_at_fpr1pct_high_dir"],
            cell["detector_tpr_at_fpr1pct_low_dir"]), 4)
        results["per_psr"][key] = cell
        with open(RES, "w") as f:
            json.dump(results, f, indent=2)
        print(f"{key}: ASR {cell['cond_asr_all']:5.2f}%  "
              f"low {cell['cond_asr_low']:5.2f}  mid {cell['cond_asr_mid']}"
              f":5.2f  high {cell['cond_asr_high']:5.2f}  AUROC* "
              f"{cell['detector_auroc_direction_free']:.3f}  TPR@1% "
              f"{cell['detector_tpr_at_fpr1pct_best']:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    # per-tertile MEAPs (20% threshold on each tertile curve)
    ps_sorted = sorted(PSR_GRID)
    def curve(name):
        return [results["per_psr"][f"psr={p:+.0f}dB"][name]
                for p in ps_sorted]
    summary = {}
    for name in ("cond_asr_all", "cond_asr_low", "cond_asr_mid",
                 "cond_asr_high"):
        meap, censor = meap_curve(ps_sorted, curve(name), 20.0)
        summary[name + "_meap_db"] = meap
        summary[name + "_censor"] = censor
    summary["margin_gap_low_minus_high_at_meap"] = None
    # at the all-MEAP PSR, the low-vs-high ASR gap:
    m_all = summary["cond_asr_all_meap_db"]
    if m_all is not None:
        # find bracketing cells
        import math
        lo = max([p for p in ps_sorted if p <= m_all], default=None)
        hi = min([p for p in ps_sorted if p >= m_all], default=None)
        if lo is not None and hi is not None and lo != hi:
            c_lo = results["per_psr"][f"psr={lo:+.0f}dB"]
            c_hi = results["per_psr"][f"psr={hi:+.0f}dB"]
            t = (m_all - lo) / (hi - lo)
            for name in ("cond_asr_low", "cond_asr_high"):
                v = c_lo[name] + t * (c_hi[name] - c_lo[name])
                summary[f"{name}_at_all_meap"] = round(v, 2)
            summary["margin_gap_low_minus_high_at_meap"] = round(
                summary["cond_asr_low_at_all_meap"] -
                summary["cond_asr_high_at_all_meap"], 2)
    results["summary"] = summary
    with open(RES, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(summary, indent=1))
    print("wrote", RES)
    print("DONE")


if __name__ == "__main__":
    main()
