#!/usr/bin/env python3
"""w14b_uncensor_genies.py — Wave 14 / 35-b-14 (council 34-d iv): un-censor
every 'le'-censored genie floor by EXTENDING the PSR grids to -55 dB with
protocol-identical PGD-10 cells.

Sites fixed (all stored, all censored at the -45 dB grid edge):
  1. results/attack_results_urban_untargeted_s123.json  (genie, seed 123)
  2. results/attack_results_urban_untargeted_s456.json  (genie, seed 456)
  3. results/real_wifi_attack.json frozen.genie
  4. results/real_wifi_attack.json frozen.genie_real_wifi_only
  5. results/real_wifi_attack.json frozen.cv2x_mask_real_wifi_only
  6. results/real_wifi_attack.json finetuned.genie

New cells: psr = -55, -50 dB, PGD-10, zero-init, alpha 0.25, psd_margin 2.0,
budgets = clean received-signal window energy, same eval sets (identical
seeds and generators as the stored runs). Deterministic given the seed, so
the stored protocol numbers above -45 dB are unchanged; only the censored
floor is resolved. Summaries (MEAP/censor/PoC) recomputed in place.

After this script, re-run:
  python3 scripts/w12_cis.py            (rebuilds w12_confidence_intervals.json)
  python3 scripts/w14_fix_poc_ci.py     (re-applies the grid55 merge, F5)

Idempotent: existing cells are skipped.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(2)

from waveforms import ATTACK_BANDS, NOISE_CLASS
from receiver import FrontEnd, DualStreamModel
from attack_mask import (cond_asr, meap_curve, price_of_compliance,
                         waveform_pgd)
from run_attack import build_eval_set
from run_real_wifi import build_mixed_eval_set
from real_wifi import build_real_wifi_pool

OUT = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data", "real_wifi")
NEW_PSRS = [-55.0, -50.0]


def pgd_cell(model, frontend, x_rx, h_a, y, p_clean, psr, band):
    """One protocol-identical PGD-10 cell -> (cond_asr_pct, n_elig, robust)."""
    p_budget = p_clean * (10 ** (psr / 10.0))
    preds = []
    for i in range(0, x_rx.size(0), 300):
        out = waveform_pgd(model, frontend, x_rx[i:i + 300],
                           h_a[i:i + 300], y[i:i + 300],
                           p_budget[i:i + 300], steps=10, band=band,
                           targeted=False, target_class=NOISE_CLASS)
        preds.append(out["preds"])
    adv = torch.cat(preds)
    with torch.no_grad():
        cp = []
        for i in range(0, x_rx.size(0), 256):
            cp.append(model.forward_wave(x_rx[i:i + 256], frontend).argmax(1))
        cp = torch.cat(cp)
    c, n = cond_asr(cp, adv, y, targeted=False, target=NOISE_CLASS)
    rob = (adv == y).float().mean().item()
    return round(100 * c, 2), int(n), round(100 * rob, 2)


def load_model(ck_name):
    ck = torch.load(os.path.join(OUT, ck_name), map_location="cpu",
                    weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])
    return model, frontend


def stored_psrs(cells):
    return sorted(float(k[4:-2]) for k in cells)


def extend_canonical_seed(seed, fname):
    """Extend the canonical PGD-10 seed run's genie grid to -55 dB."""
    path = os.path.join(OUT, fname)
    d = json.load(open(path))
    node = d["per_scenario"]["urban"]["runs"]["untargeted/genie"]
    todo = [p for p in NEW_PSRS if f"psr={p:+.0f}dB" not in node]
    if not todo:
        print(f"[s{seed}] genie grid already extended", flush=True)
    else:
        model, frontend = load_model("checkpoint_dual.pt")
        x_rx, y, h_a, p_clean = build_eval_set("urban", 100, seed)
        for psr in todo:
            c, n, r = pgd_cell(model, frontend, x_rx, h_a, y, p_clean, psr,
                               None)
            node[f"psr={psr:+.0f}dB"] = {"cond_asr": c, "n_eligible": n,
                                        "robust_acc": r}
            print(f"[s{seed}] genie {psr:+.0f}dB: cond-ASR {c}%  "
                  f"(n={n})", flush=True)
        json.dump(d, open(path, "w"), indent=2)

    # recompute the urban untargeted summary over stored curves
    runs = d["per_scenario"]["urban"]["runs"]
    pg, pm = stored_psrs(runs["untargeted/genie"]), \
        stored_psrs(runs["untargeted/cv2x_mask"])
    g = [runs["untargeted/genie"][f"psr={p:+.0f}dB"]["cond_asr"] for p in pg]
    m = [runs["untargeted/cv2x_mask"][f"psr={p:+.0f}dB"]["cond_asr"]
         for p in pm]
    mg, cg = meap_curve(pg, g, 20.0)
    mm, cm = meap_curve(pm, m, 20.0)
    poc, cnotes = price_of_compliance(pg, g, pm, m, 20.0)
    d["per_scenario"]["urban"]["summary"]["untargeted"] = {
        "meap_genie_db": mg, "meap_genie_censor": cg,
        "meap_cv2x_mask_db": mm, "meap_cv2x_mask_censor": cm,
        "price_of_compliance_db": poc, "poc_censor": cnotes,
        "genie_curve": g, "mask_curve": m,
    }
    json.dump(d, open(path, "w"), indent=2)
    print(f"[s{seed}] MEAP genie {mg} ({cg})  mask {mm} ({cm})  "
          f"PoC {poc} {cnotes or ''}", flush=True)


def recompute_curve_ci_like(curve_cells):
    """(psrs, asrs) from a real-wifi curve dict {'psr=XdB': cond_asr|dict}."""
    psrs, asrs = [], []
    for k in sorted(curve_cells, key=lambda s: float(s[4:-2])):
        v = curve_cells[k]
        asr = v["cond_asr"] if isinstance(v, dict) else v
        psrs.append(float(k[4:-2]))
        asrs.append(asr)
    return psrs, asrs


def extend_real_wifi():
    path = os.path.join(OUT, "real_wifi_attack.json")
    d = json.load(open(path))

    # rebuild the mixed eval set exactly as run_real_wifi.py did
    pool, metas, prov, win_refs = build_real_wifi_pool(DATA)
    eval_idx = [i for i, (fn, s) in enumerate(win_refs) if "_uz_" in fn]
    pool = pool[eval_idx]
    x_rx, y, h_a, p_clean, real_mask = build_mixed_eval_set(pool, 100, 7)
    rm = real_mask.bool()
    band = ATTACK_BANDS["cv2x_attacker"]

    # ---- frozen arms ----
    frozen_model, frozen_fe = load_model("checkpoint_dual.pt")
    targets = [
        ("genie", "frozen", frozen_model, frozen_fe, x_rx, h_a, y, p_clean,
         None),
        ("genie_real_wifi_only", "frozen", frozen_model, frozen_fe,
         x_rx[rm], h_a[rm], y[rm], p_clean[rm], None),
        ("cv2x_mask_real_wifi_only", "frozen", frozen_model, frozen_fe,
         x_rx[rm], h_a[rm], y[rm], p_clean[rm], band),
    ]
    for arm_name, which, model, fe, xr, ha, yy, pc, sband in targets:
        curve = d[which][arm_name]["curve"]
        # normalize cell shape to match the stored cells (dict vs float):
        # w12_cis.find_curves requires ALL leaves of a subtree to be dicts
        # with 'cond_asr' when the stored curve uses that shape.
        ref = next(iter(curve.values()), None)
        want_dict = isinstance(ref, dict)
        if want_dict:
            n_ref = next((c.get("n_eligible") for c in curve.values()
                          if isinstance(c, dict)), None)
            for k, v in list(curve.items()):
                if not isinstance(v, dict):
                    curve[k] = {"cond_asr": v, "n_eligible": n_ref,
                                "robust_acc": None}
        todo = [p for p in NEW_PSRS if f"psr={p:+.0f}dB" not in curve]
        if todo:
            for psr in todo:
                c, n, r = pgd_cell(model, fe, xr, ha, yy, pc, psr, sband)
                if want_dict:
                    curve[f"psr={psr:+.0f}dB"] = {"cond_asr": c,
                                                  "n_eligible": n,
                                                  "robust_acc": r}
                else:
                    curve[f"psr={psr:+.0f}dB"] = c
                print(f"[real-wifi {which}.{arm_name}] {psr:+.0f}dB: "
                      f"cond-ASR {c}%", flush=True)
        psrs, asrs = recompute_curve_ci_like(curve)
        mg, cg = meap_curve(psrs, asrs, 20.0)
        d[which][arm_name]["meap_db"] = mg
        d[which][arm_name]["censor"] = cg
        print(f"[real-wifi {which}.{arm_name}] MEAP {mg} ({cg})", flush=True)

    # ---- finetuned genie ----
    ft_model, ft_fe = load_model("checkpoint_dual_realft.pt")
    curve = d["finetuned"]["genie"]["curve"]
    ref = next(iter(curve.values()), None)
    want_dict = isinstance(ref, dict)
    if want_dict:
        n_ref = next((c.get("n_eligible") for c in curve.values()
                      if isinstance(c, dict)), None)
        for k, v in list(curve.items()):
            if not isinstance(v, dict):
                curve[k] = {"cond_asr": v, "n_eligible": n_ref,
                            "robust_acc": None}
    todo = [p for p in NEW_PSRS if f"psr={p:+.0f}dB" not in curve]
    if todo:
        for psr in todo:
            c, n, r = pgd_cell(ft_model, ft_fe, x_rx, h_a, y, p_clean, psr,
                               None)
            if want_dict:
                curve[f"psr={psr:+.0f}dB"] = {"cond_asr": c,
                                              "n_eligible": n,
                                              "robust_acc": r}
            else:
                curve[f"psr={psr:+.0f}dB"] = c
            print(f"[real-wifi finetuned.genie] {psr:+.0f}dB: "
                  f"cond-ASR {c}%", flush=True)
    psrs, asrs = recompute_curve_ci_like(curve)
    mg, cg = meap_curve(psrs, asrs, 20.0)
    d["finetuned"]["genie"]["meap_db"] = mg
    d["finetuned"]["genie"]["censor"] = cg
    print(f"[real-wifi finetuned.genie] MEAP {mg} ({cg})", flush=True)

    # ---- PoCs over full-set curves ----
    for which in ("frozen", "finetuned"):
        gp, ga = recompute_curve_ci_like(d[which]["genie"]["curve"])
        mp, ma = recompute_curve_ci_like(d[which]["cv2x_mask"]["curve"])
        poc, cn = price_of_compliance(gp, ga, mp, ma, 20.0)
        d[which]["price_of_compliance_db"] = poc
        print(f"[real-wifi {which}] PoC {poc} dB {cn or ''}", flush=True)

    d.setdefault("config", {})
    d["config"]["w14b_uncensor_update"] = {
        "what": "genie (and real-WiFi-only mask) floors extended to -55 dB "
                "with protocol-identical PGD-10 cells (35-b-14); stored "
                "cells above -45 dB unchanged; MEAPs/PoCs recomputed",
        "new_psrs_db": NEW_PSRS,
    }
    json.dump(d, open(path, "w"), indent=2)


def main():
    torch.manual_seed(7)
    extend_canonical_seed(123, "attack_results_urban_untargeted_s123.json")
    extend_canonical_seed(456, "attack_results_urban_untargeted_s456.json")
    extend_real_wifi()
    print("UNCENSOR COMPLETE — now re-run w12_cis.py + w14_fix_poc_ci.py",
          flush=True)


if __name__ == "__main__":
    main()
