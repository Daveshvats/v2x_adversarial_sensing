#!/usr/bin/env python3
"""run_etsi_mask.py — mask-SHAPE sensitivity: ETSI EN 302 571 Table-7-shaped
attacker vs the paper's idealized flat-cap attacker (council P0, 21-a/21-b:
"flat-cap mask not ETSI EN 302 571 template").

Protocol IDENTICAL to the canonical runs behind attack_results_merged.json
(urban, untargeted, PGD-10, attack seed 7, 100 active samples/class, alpha
0.25, psd_margin 2.0, PSR grid -45..0, budgets = clean received-signal window
energy, same eval sets via run_attack.build_eval_set) — the ONLY change is
the projection shape: ETSI Table 7 piecewise skirt (src/etsi_mask.py,
values verbatim from data/standards/en302571_tables.json) instead of the
hard OOB null + flat 2x in-band cap.

Output: results/etsi_mask_results.json with (a) the ETSI-shaped mask-attacker
curve, (b) the flat-cap reference copied verbatim from
attack_results_merged.json for the same protocol, (c) MEAP/PoC comparison
and the verdict on whether the Price-of-Compliance headline survives the
real mask shape.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(2)

from waveforms import ATTACK_BANDS, NOISE_CLASS
from receiver import FrontEnd, DualStreamModel
from attack_mask import cond_asr, meap_curve, waveform_pgd
from etsi_mask import project_etsi
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")
PSR_GRID = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0,
            -5.0, 0.0]
FC_MHZ = 5.0                     # attacker carrier: centre of [0,+10] MHz


def etsi_project_wrapper(delta, band, p_budget, psd_margin=2.0):
    """Adapt project_etsi(delta, fc, ...) to the waveform_pgd project_fn
    signature (delta, band, p_budget, psd_margin) — band is implicit in fc."""
    return project_etsi(delta, FC_MHZ, p_budget, psd_margin)


def main():
    t0 = time.time()
    torch.manual_seed(7)
    ck = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                    map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])

    x_rx, y, h_a, p_clean = build_eval_set("urban", 100, 7)
    with torch.no_grad():
        clean_preds = []
        for i in range(0, x_rx.size(0), 256):
            lg = model.forward_wave(x_rx[i:i + 256], frontend)
            clean_preds.append(lg.argmax(1))
        clean_preds = torch.cat(clean_preds)
    clean_acc = (clean_preds == y).float().mean().item()
    print(f"[etsi] clean acc on active tx: {clean_acc*100:.2f}%", flush=True)

    runs = {}
    for tag, pfn in [("etsi_table7", etsi_project_wrapper)]:
        runs[f"untargeted/{tag}"] = {}
        for psr in PSR_GRID:
            p_budget = p_clean * (10 ** (psr / 10.0))
            preds = []
            for i in range(0, x_rx.size(0), 300):
                out = waveform_pgd(model, frontend, x_rx[i:i + 300],
                                   h_a[i:i + 300], y[i:i + 300],
                                   p_budget[i:i + 300], steps=10,
                                   band=ATTACK_BANDS["cv2x_attacker"],
                                   targeted=False,
                                   psd_margin=2.0, project_fn=pfn)
                preds.append(out["preds"])
            adv = torch.cat(preds)
            c_asr, n_elig = cond_asr(clean_preds, adv, y, targeted=False)
            rob = (adv == y).float().mean().item()
            runs[f"untargeted/{tag}"][f"psr={psr:+.0f}dB"] = {
                "cond_asr": round(100 * c_asr, 2),
                "robust_acc": round(100 * rob, 2),
                "n_eligible": n_elig,
            }
            print(f"    untargeted/{tag} PSR {psr:+5.0f} dB: "
                  f"cond-ASR {100*c_asr:5.1f}%  robust {100*rob:5.1f}%",
                  flush=True)

    # ---- flat-cap reference, verbatim from the canonical merged JSON ----
    ref = json.load(open(os.path.join(OUT, "attack_results_merged.json")))
    flat = ref["per_scenario"]["urban"]["runs"]["untargeted/cv2x_mask"]
    genie = ref["per_scenario"]["urban"]["runs"]["untargeted/genie"]

    e = [runs["untargeted/etsi_table7"][f"psr={p:+.0f}dB"]["cond_asr"]
         for p in PSR_GRID]
    f_ = [flat[f"psr={p:+.0f}dB"]["cond_asr"]
          for p in PSR_GRID if f"psr={p:+.0f}dB" in flat]
    f_psr = [p for p in PSR_GRID if f"psr={p:+.0f}dB" in flat]
    g = [genie[f"psr={p:+.0f}dB"]["cond_asr"] for p in f_psr]

    me, ce = meap_curve(PSR_GRID, e, 20.0)
    mf, cf = meap_curve(f_psr, f_, 20.0)
    mg, cg = meap_curve(f_psr, g, 20.0)

    results = {
        "experiment": "Mask-shape sensitivity: ETSI EN 302 571 V1.2.1 "
                      "Table-7-shaped attacker vs idealized flat cap",
        "config": {
            "scenario": "urban", "mode": "untargeted", "steps": 10,
            "seed": 7, "n_eval_per_class": 100,
            "attack_pgd_alpha_frac": 0.25, "psd_cap_margin": 2.0,
            "psr_db": PSR_GRID,
            "attacker_carrier_fc_mhz_rel_5895": FC_MHZ,
            "etsi_template": "EN 302 571 V1.2.1 (2013-09) Table 7, "
                             "10 MHz channel; linear-in-dB interpolation; "
                             "-50 dB rel held beyond +-15 MHz",
            "etsi_knots_offset_mhz": [0.0, 4.5, 5.0, 5.5, 10.0, 15.0],
            "etsi_knots_db_rel": [0.0, 0.0, -26.0, -32.0, -40.0, -50.0],
            "standard_source": "data/standards/en302571_tables.json "
                               "(official ETSI PDF, sha256 667a939...)",
            "normalization": "in-channel reference cap = 2x uniform budget "
                             "sharing (same as flat cap); total window-energy "
                             "budget identical",
        },
        "clean_acc_active": round(clean_acc, 4),
        "runs": runs,
        "summary": {
            "meap_etsi_table7_db": me, "meap_etsi_censor": ce,
            "meap_flat_cap_db": mf, "meap_flat_censor": cf,
            "meap_genie_db": mg, "meap_genie_censor": cg,
            "poc_flat_minus_genie_db": (None if (mf is None or mg is None)
                                        else round(mf - mg, 2)),
            "poc_etsi_minus_genie_db": (None if (me is None or mg is None)
                                        else round(me - mg, 2)),
            "shape_delta_meap_db": (None if (me is None or mf is None)
                                    else round(me - mf, 2)),
            "note": "positive shape_delta = ETSI-shaped mask WEAKER for the "
                    "attacker (higher MEAP) than the idealized flat cap; "
                    "negative = stronger",
        },
        "flat_cap_reference": {
            "source": "results/attack_results_merged.json (urban, "
                      "untargeted, cv2x_mask — same protocol)",
            "curve": {k: v["cond_asr"] for k, v in flat.items()},
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "etsi_mask_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[etsi] MEAP ETSI {me} ({ce})   flat {mf} ({cf})   "
          f"genie {mg} ({cg})", flush=True)
    print(f"[etsi] shape delta (ETSI - flat): "
          f"{results['summary']['shape_delta_meap_db']} dB", flush=True)
    print(f"[etsi] PoC flat {results['summary']['poc_flat_minus_genie_db']} "
          f"dB vs PoC ETSI {results['summary']['poc_etsi_minus_genie_db']} dB",
          flush=True)


if __name__ == "__main__":
    main()
