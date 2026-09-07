#!/usr/bin/env python3
"""run_conformance_us.py — Wave 14 / 35-c-2 (council F3): run the 47 CFR
95.3205-SHAPED conformance projection, closing F3 with data instead of
wording ("archived + projection queued" -> projection RUN).

Protocol IDENTICAL to the canonical runs behind attack_results_merged.json /
run_conformance.py (urban, untargeted, PGD-10, attack seed 7, n=100/class,
alpha 0.25, psd_margin 2.0, PSR grid -45..0, budgets = clean received-signal
window energy, same eval set via run_attack.build_eval_set). The ONLY change
is the projection domain: src/conformance_us.py project_conform_us — the US
C-V2X OBU unwanted-emissions template (47 CFR 95.3205; rows archived in
data/standards/en302571_tables.json) enforced as RBW/mean-power constraints
with the rule's own measurement structure (100-kHz RBW within 1 MHz of the
channel edge, 1-MHz RBW further out, flat-per-region limits).

Arms compared in the summary (all stored, same protocol):
  * conformance_us (NEW, this run)
  * conformance_rbw (ETSI Table-7 in the RBW domain, results/conformance_results.json, C41)
  * etsi_table7 per-bin (measurement-INEQUIVALENT reference, etsi_mask_results.json)
  * flat_cap (attack_results_merged.json) and genie

Post-PA check: projected delta pushed through the Wave-11 Rapp PA at IBO 13
and MEASURED with measure_us ("asserted by projection, verified by
measurement").

Idempotent/resumable: cells already present in the output JSON are skipped.
Output: results/conformance_us_results.json
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
from conformance_us import project_conform_us, measure_us
from run_attack import build_eval_set
from run_papr_pa import rapp_pa

OUT = os.path.join(ROOT, "results")
PSR_GRID = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0, -10.0,
            -5.0, 0.0]
FC_MHZ = 5.0
IBO_DB = 13.0
RAPP_P = 3.0
RESULT_JSON = os.path.join(OUT, "conformance_us_results.json")


def us_project_wrapper(delta, band, p_budget, psd_margin=2.0):
    return project_conform_us(delta, FC_MHZ, p_budget, psd_margin)


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

    report = None
    if os.path.exists(RESULT_JSON):
        report = json.load(open(RESULT_JSON))
    if report is None or "runs" not in report:
        report = {
            "experiment": "US 47 CFR 95.3205-shaped conformance-domain "
                          "enforcement: C-V2X OBU unwanted-emissions limits "
                          "(89 FR 100855) as RBW/mean-power constraints "
                          "(100-kHz RBW within 1 MHz of the channel edge, "
                          "1-MHz RBW further out, flat-per-region limits)",
            "config": {
                "scenario": "urban", "mode": "untargeted", "steps": 10,
                "seed": 7, "n_eval_per_class": 100,
                "attack_pgd_alpha_frac": 0.25, "psd_cap_margin": 2.0,
                "psr_db": list(PSR_GRID),
                "attacker_carrier_fc_mhz_rel_5895": 5.0,
                "measurement_domain": "mean power per band; RBW 100 kHz for "
                                       "centres within 1 MHz of the attacker "
                                       "channel edge (step 0.05 MHz), RBW "
                                       "1 MHz elsewhere (step 0.5 MHz, "
                                       "overlapping)",
                "standard_source": "data/standards/en302571_tables.json "
                                   "fcc_47cfr_95_3205 rows (eCFR, 89 FR "
                                   "100855; corrected 90 FR 5724)",
                "interpretation": [
                    "limits keyed to the attacker's 10-MHz CHANNEL edge "
                    "(archived rows say offset_from_band_edge; for an "
                    "adjacent-channel in-band attacker the channel edge is "
                    "the binding reference)",
                    "-16 dBm/100 kHz = -6 dBm/MHz equivalent = -29 dB rel "
                    "the 23 dBm/MHz in-channel anchor; -13 dBm/MHz -> -36 "
                    "dB rel; -16 dBm/MHz -> -39 dB rel",
                    "RBW and limit keyed to the measurement centre "
                    "(measurement-receiver convention)",
                    "flat limit per offset region (step boundaries at 1 and "
                    "5 MHz, literal reading; no interpolation knots in the "
                    "US rule)",
                    "in-channel unconstrained by the OOB table (0 dB rel, "
                    "same in-channel treatment as the ETSI/flat arms)",
                    "PSR-relative budget convention identical to "
                    "project_etsi/project_conform",
                ],
                "pa_check": f"Rapp p={RAPP_P}, IBO {IBO_DB} dB (C30 "
                            f"compliant backoff), post-PA measured with "
                            f"measure_us (same band structure)",
                "projection": "per-band proportional water-filling, 8 iters, "
                              "residual disclosed per cell",
            },
            "runs": {},
            "elapsed_s": 0.0,
        }

    runs = report["runs"]
    key = "untargeted/conformance_us"
    if key not in runs:
        runs[key] = {}

    x_rx, y, h_a, p_clean = build_eval_set("urban", 100, 7)
    with torch.no_grad():
        clean_preds = []
        for i in range(0, x_rx.size(0), 256):
            lg = model.forward_wave(x_rx[i:i + 256], frontend)
            clean_preds.append(lg.argmax(1))
        clean_preds = torch.cat(clean_preds)
    clean_acc = (clean_preds == y).float().mean().item()
    report["clean_acc_active"] = clean_acc
    print(f"[conform-us] clean acc on active tx: {clean_acc*100:.2f}%",
          flush=True)

    f = torch.fft.fftfreq(x_rx.size(1), d=1.0 / 20e6) / 1e6
    inband = (f >= FC_MHZ - 5.0) & (f <= FC_MHZ + 5.0)
    n_in = float(inband.float().sum())

    for psr in PSR_GRID:
        cell = f"psr={psr:+.0f}dB"
        if cell in runs[key]:
            print(f"[conform-us] {cell}: cached "
                  f"(asr {runs[key][cell]['cond_asr']})", flush=True)
            continue
        p_budget = p_clean * (10 ** (psr / 10.0))
        p_ref = 2.0 * x_rx.size(1) * p_budget / n_in

        preds, proj_stats, worst_postpa = [], [], []
        for i in range(0, x_rx.size(0), 300):
            out = waveform_pgd(model, frontend, x_rx[i:i + 300],
                               h_a[i:i + 300], y[i:i + 300],
                               p_budget[i:i + 300], steps=10,
                               band=ATTACK_BANDS["cv2x_attacker"],
                               targeted=False, psd_margin=2.0,
                               return_delta=True,
                               project_fn=us_project_wrapper)
            preds.append(out["preds"])
            d = out["delta"]
            w_pre = measure_us(d, FC_MHZ, p_ref[i:i + 300])
            proj_stats.append(float(w_pre.max()))
            rms = d.abs().pow(2).sum(dim=1).sqrt()
            asat = rms * 10 ** (IBO_DB / 20.0)
            y_pa = rapp_pa(d, asat, RAPP_P)
            w_post = measure_us(y_pa, FC_MHZ, p_ref[i:i + 300])
            worst_postpa.append(float(w_post.max()))

        adv = torch.cat(preds)
        c_asr, n_elig = cond_asr(clean_preds, adv, y, targeted=False)
        runs[key][cell] = {
            "cond_asr": round(100 * c_asr, 2),
            "robust_acc": round(100 * (adv == y).float().mean().item(), 2),
            "n_eligible": n_elig,
            "proj_residual_dbr_worst": round(max(proj_stats), 3),
            "post_pa_worst_excess_dbr": round(max(worst_postpa), 3),
        }
        report["elapsed_s"] = round(report.get("elapsed_s", 0.0) +
                                    (time.time() - t0), 1)
        json.dump(report, open(RESULT_JSON, "w"), indent=1)
        print(f"[conform-us] {cell}: cond-ASR {100*c_asr:5.1f}%  "
              f"pre-PA worst {max(proj_stats):.2f} dBr  "
              f"post-PA worst {max(worst_postpa):.2f} dBr", flush=True)
        t0 = time.time()

    # ---- summary vs all stored arms ----
    psrs, asrs = [], []
    for psr in PSR_GRID:
        c = runs[key][f"psr={psr:+.0f}dB"]
        psrs.append(psr)
        asrs.append(c["cond_asr"])
    meap_us, cens = meap_curve(psrs, asrs)

    etsi_rbw = json.load(open(os.path.join(OUT, "conformance_results.json")))
    meap_etsi_rbw = etsi_rbw["summary"]["meap_conformance_rbw_db"]
    flat = json.load(open(os.path.join(OUT, "etsi_mask_results.json")))
    meap_etsi = flat["summary"]["meap_etsi_table7_db"]
    meap_flat = flat["summary"]["meap_flat_cap_db"]
    meap_genie = flat["summary"]["meap_genie_db"]

    if meap_us is None:
        print("MEAP unresolved (grid incomplete) — re-run to complete",
              flush=True)
        return
    post_pa = [runs[key][f"psr={p:+.0f}dB"]["post_pa_worst_excess_dbr"]
               for p in PSR_GRID]
    report["summary"] = {
        "meap_conformance_us_db": meap_us,
        "meap_conformance_us_censor": cens,
        "meap_etsi_rbw_db": meap_etsi_rbw,
        "meap_etsi_perbin_db": meap_etsi,
        "meap_flat_cap_db": meap_flat,
        "meap_genie_db": meap_genie,
        "delta_us_minus_etsi_rbw_db": round(meap_us - meap_etsi_rbw, 3),
        "delta_us_minus_flat_db": round(meap_us - meap_flat, 3),
        "poc_conformance_us_db": round(meap_us - meap_genie, 2),
        "post_pa_worst_excess_dbr_max": round(max(post_pa), 3),
        "post_pa_worst_excess_dbr_mean": round(float(np.mean(post_pa)), 3),
        "verdict_note": "positive delta = the US 95.3205-shaped enforcement "
                        "is WEAKER for the attacker than the named "
                        "reference (attack needs MORE power); all three "
                        "enforcement domains compared under the identical "
                        "protocol",
    }
    report["elapsed_s"] = round(report.get("elapsed_s", 0.0) +
                                (time.time() - t0), 1)
    json.dump(report, open(RESULT_JSON, "w"), indent=1)
    print(json.dumps(report["summary"], indent=1))


if __name__ == "__main__":
    main()
