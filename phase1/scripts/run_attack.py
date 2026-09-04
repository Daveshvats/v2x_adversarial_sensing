#!/usr/bin/env python3
"""run_attack.py — the Compliant-Attacker experiments (Phase-1 core results).

For each scenario in {urban, highway, rural} and each mode in
{untargeted, targeted_noise} (selectable via --modes; run one mode at a time
to stay inside short process limits):
  1. build a fixed eval set of ACTIVE transmissions (classes PC5/11p/WiFi —
     attacking a noise sample is meaningless),
  2. clean predictions (conditional-ASR denominator),
  3. attack settings (SAME window-energy budget in both):
       genie      : power-only budget, full 20 MHz, NO PSD cap (upper bound)
       cv2x_mask  : budget + emissions restricted to the attacker's ITS-band
                    allocation [0, +10] MHz (the compliant attacker), OOB
                    nulled + in-band flat-PSD cap (psd_cap_margin)
  4. PSR sweep. PSR (dB) = 10 log10(E_attack / E_clean_rx) with E_* the
     2048-sample window energy — equal-length windows, so PSR is also the
     mean-power ratio. (NOTE: a 2026-09-05 audit found an earlier version
     used a mean-power budget against an energy constraint, shifting all
     absolute PSR labels by 10 log10(2048) = 33.11 dB; this file is the
     corrected version. The Price-of-Compliance difference metric was
     invariant to that error; absolute MEAP values were not.)
  5. MEAP (Minimum Effective Attack Power) with explicit censoring flags.

Outputs: results/attack_results_{scenario}_{mode}.json per invocation;
scripts/run_plot.py merges them into attack_results_merged.json + the figure.
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch

from waveforms import gen_signal, CLASS_NAMES, NOISE_CLASS, ATTACK_BANDS
import channels as CH
from receiver import FrontEnd, DualStreamModel
from attack_mask import (cconv, waveform_pgd, cond_asr, meap_curve,
                         price_of_compliance, project)

OUT = os.path.join(ROOT, "results")

DEFAULT_PSR = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
               -10.0, -5.0, 0.0, 5.0, 10.0]


def build_eval_set(scenario, n_per_class, seed):
    """Returns x_rx (B,L) signal+noise, y, h_a, p_clean (B,) window energy of
    the pre-noise received signal — the physical PSR reference."""
    rng = np.random.default_rng(seed)
    X, y, H, P = [], [], [], []
    L = 2048
    for cls in (0, 1, 2):                    # active transmissions only
        for _ in range(n_per_class):
            s = gen_signal(cls, rng)
            h_v = CH.draw_channel(scenario, rng)          # victim link
            r = np.convolve(s, h_v)[(len(h_v) - 1) // 2:
                                     (len(h_v) - 1) // 2 + L]
            p = np.mean(np.abs(r) ** 2)
            p_clean = np.sum(np.abs(r) ** 2)               # window ENERGY
            snr = rng.uniform(5, 25)
            ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
            r = r + ns * (rng.standard_normal(L) +
                          1j * rng.standard_normal(L))
            h_a = CH.draw_channel(scenario, rng)          # attacker link
            X.append(r.astype(np.complex64))
            y.append(cls)
            H.append(h_a)
            P.append(p_clean)
    return (torch.from_numpy(np.array(X)),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(np.array(H)),
            torch.tensor(np.array(P), dtype=torch.float32))


def run_setting(model, frontend, x_rx, h_a, y, p_budget, band, targeted,
                steps, psd_margin):
    """Attack in eval-sized chunks (memory safety)."""
    preds = []
    B = 300
    for i in range(0, x_rx.size(0), B):
        out = waveform_pgd(model, frontend, x_rx[i:i + B], h_a[i:i + B],
                           y[i:i + B], p_budget[i:i + B], steps=steps,
                           band=band, targeted=targeted,
                           target_class=NOISE_CLASS, psd_margin=psd_margin)
        preds.append(out["preds"])
    return torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=100,
                    help="samples per active class (300 total by default)")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--psr", nargs="+", type=float, default=DEFAULT_PSR)
    ap.add_argument("--scenarios", nargs="+", default=["urban", "highway"])
    ap.add_argument("--modes", nargs="+",
                    default=["untargeted", "targeted_noise"],
                    choices=["untargeted", "targeted_noise"],
                    help="run a subset to stay inside short process limits")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--psd-margin", type=float, default=2.0,
                    help="in-band flat-PSD cap margin (mask attacker only);"
                         " sweep {2, 10} for the idealization sensitivity")
    ap.add_argument("--alpha", type=float, default=0.25)
    args = ap.parse_args()

    t0 = time.time()
    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    for scen in args.scenarios:
        print(f"\n=== scenario: {scen} ===", flush=True)
        x_rx, y, h_a, p_clean = build_eval_set(scen, args.n_eval, args.seed)

        with torch.no_grad():
            clean_preds = []
            for i in range(0, x_rx.size(0), 256):
                lg = model.forward_wave(x_rx[i:i + 256], frontend)
                clean_preds.append(lg.argmax(1))
            clean_preds = torch.cat(clean_preds)
        clean_acc = (clean_preds == y).float().mean().item()
        print(f"  clean acc on active tx: {clean_acc*100:.2f}%", flush=True)

        p_rx = p_clean          # window energy of the clean received SIGNAL
        band = ATTACK_BANDS["cv2x_attacker"]
        scen_file = os.path.join(OUT, f"attack_results_{scen}_{{}}.json")

        for mode in args.modes:
            targeted = (mode == "targeted_noise")
            fname = scen_file.format(mode)
            runs = {}
            for setting, sband in [("genie", None),
                                   ("cv2x_mask", band)]:
                key = f"{mode}/{setting}"
                runs[key] = {}
                for psr in args.psr:
                    p_budget = p_rx * (10 ** (psr / 10.0))
                    adv_preds = run_setting(model, frontend, x_rx, h_a, y,
                                            p_budget, sband, targeted,
                                            args.steps, args.psd_margin)
                    c_asr, n_elig = cond_asr(clean_preds, adv_preds, y,
                                             targeted=targeted,
                                             target=NOISE_CLASS)
                    rob = (adv_preds == y).float().mean().item()
                    runs[key][f"psr={psr:+.0f}dB"] = {
                        "cond_asr": round(100 * c_asr, 2),
                        "robust_acc": round(100 * rob, 2),
                        "n_eligible": n_elig,
                    }
                    print(f"    {key:26s} PSR {psr:+5.0f} dB: "
                          f"cond-ASR {100*c_asr:5.1f}%  "
                          f"robust {100*rob:5.1f}%", flush=True)

            # per-mode summary with censoring-aware MEAP
            g = [runs[f"{mode}/genie"][f"psr={p:+.0f}dB"]["cond_asr"]
                 for p in args.psr]
            m = [runs[f"{mode}/cv2x_mask"][f"psr={p:+.0f}dB"]["cond_asr"]
                 for p in args.psr]
            mg, cg = meap_curve(args.psr, g, 20.0)
            mm, cm = meap_curve(args.psr, m, 20.0)
            poc, cnotes = price_of_compliance(args.psr, g, args.psr, m, 20.0)
            summary = {
                "meap_genie_db": mg, "meap_genie_censor": cg,
                "meap_cv2x_mask_db": mm, "meap_cv2x_mask_censor": cm,
                "price_of_compliance_db": poc, "poc_censor": cnotes,
                "genie_curve": g, "mask_curve": m,
            }
            results = {
                "experiment": "Compliant-Attacker (mask-constrained "
                              "waveform PGD, corrected PSR axis)",
                "config": {
                    "steps": args.steps, "psr_db": args.psr,
                    "n_eval_per_class": args.n_eval, "seed": args.seed,
                    "attack_pgd_alpha_frac": args.alpha,
                    "psd_cap_margin": args.psd_margin,
                    "psr_reference": "clean received-signal window energy "
                                     "(sum |r|^2, pre-noise)",
                    "mask_band_mhz": band,
                    "genie": "power-only, no PSD cap (corrected 2026-09-05)",
                },
                "per_scenario": {
                    scen: {
                        "clean_acc_active": round(clean_acc, 4),
                        "runs": runs,
                        "summary": {mode: summary},
                    }
                },
                "elapsed_s": round(time.time() - t0, 1),
            }
            with open(fname, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  wrote {fname}", flush=True)
            print(f"    MEAP genie {mg} dB ({cg})  mask {mm} dB ({cm})  "
                  f"PoC {poc} dB {('' if not cnotes else cnotes)}",
                  flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
