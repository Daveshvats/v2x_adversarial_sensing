#!/usr/bin/env python3
"""w3a_run_attack.py — chunkable/resumable variant of run_attack.py (W3-A, Task 7-a).

Same experiment, same eval sets, same attack code as scripts/run_attack.py
(build_eval_set is IMPORTED from it, so the sample sets are bit-identical for a
given scenario/seed), but:

  * the results JSON is (re)written after EVERY completed PSR point, so a
    sandbox kill loses at most one condition;
  * --resume skips conditions already present in the output file (config must
    match), so a killed sweep is continued, not restarted;
  * output goes to a file named by --out (default: the canonical
    attack_results_{scenario}_{mode}.json), so cited baseline JSONs are never
    touched unless the default name is explicitly requested.

Output format mirrors run_attack.py exactly (per_scenario -> runs ->
"{mode}/{setting}" -> "psr=XdB" -> {cond_asr, robust_acc, n_eligible}; summary
recomputed from the points completed so far — exact on the full grid).
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

from run_attack import build_eval_set, DEFAULT_PSR, OUT
from waveforms import NOISE_CLASS, ATTACK_BANDS
from receiver import FrontEnd, DualStreamModel
from attack_mask import waveform_pgd, cond_asr, meap_curve, price_of_compliance


def run_setting(model, frontend, x_rx, h_a, y, p_budget, band, targeted,
                steps, psd_margin):
    """Attack in eval-sized chunks (memory safety) — identical to run_attack."""
    preds = []
    B = 300
    for i in range(0, x_rx.size(0), B):
        out = waveform_pgd(model, frontend, x_rx[i:i + B], h_a[i:i + B],
                           y[i:i + B], p_budget[i:i + B], steps=steps,
                           band=band, targeted=targeted,
                           target_class=NOISE_CLASS, psd_margin=psd_margin)
        preds.append(out["preds"])
    return torch.cat(preds)


def summarize(runs, mode, psr_grid):
    """Recompute MEAP/PoC from whatever PSR points are complete (both settings)."""
    key_g, key_m = f"{mode}/genie", f"{mode}/cv2x_mask"
    if key_g not in runs or key_m not in runs:
        return None
    done = [p for p in psr_grid if f"psr={p:+.0f}dB" in runs[key_g]
            and f"psr={p:+.0f}dB" in runs[key_m]]
    if len(done) < 2:
        return None
    g = [runs[key_g][f"psr={p:+.0f}dB"]["cond_asr"] for p in done]
    m = [runs[key_m][f"psr={p:+.0f}dB"]["cond_asr"] for p in done]
    mg, cg = meap_curve(done, g, 20.0)
    mm, cm = meap_curve(done, m, 20.0)
    poc, cnotes = price_of_compliance(done, g, done, m, 20.0)
    return {
        "meap_genie_db": mg, "meap_genie_censor": cg,
        "meap_cv2x_mask_db": mm, "meap_cv2x_mask_censor": cm,
        "price_of_compliance_db": poc, "poc_censor": cnotes,
        "genie_curve": g, "mask_curve": m,
        "psr_completed_db": done,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--psr", nargs="+", type=float, default=DEFAULT_PSR,
                    help="target grid recorded in the JSON + used for "
                         "MEAP/PoC summaries")
    ap.add_argument("--psr-now", nargs="+", type=float, default=None,
                    help="run only these points THIS invocation (subset of "
                         "--psr; lets one sweep span several short "
                         "invocations without config mismatch)")
    ap.add_argument("--scenarios", nargs="+", default=["urban", "highway"])
    ap.add_argument("--modes", nargs="+", default=["untargeted"],
                    choices=["untargeted", "targeted_noise"])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--psd-margin", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--out", default=None,
                    help="output filename (inside results/); default "
                         "attack_results_{scenario}_{mode}.json")
    ap.add_argument("--resume", action="store_true",
                    help="continue an existing output file (config must match)")
    args = ap.parse_args()

    t0 = time.time()
    psr_grid = sorted(args.psr)
    psr_now = sorted(args.psr_now) if args.psr_now else psr_grid
    bad = [p for p in psr_now if p not in psr_grid]
    if bad:
        sys.exit(f"--psr-now must be a subset of --psr; offending: {bad}")

    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    band = ATTACK_BANDS["cv2x_attacker"]

    for scen in args.scenarios:
        for mode in args.modes:
            out_name = args.out or f"attack_results_{scen}_{mode}.json"
            fname = os.path.join(OUT, out_name)
            cfg = {
                "steps": args.steps, "psr_db": psr_grid,
                "n_eval_per_class": args.n_eval, "seed": args.seed,
                "attack_pgd_alpha_frac": args.alpha,
                "psd_cap_margin": args.psd_margin,
                "scenario": scen, "mode": mode,
                "psr_reference": "clean received-signal window energy "
                                 "(sum |r|^2, pre-noise)",
                "mask_band_mhz": band,
                "genie": "power-only, no PSD cap (corrected 2026-09-05)",
                "runner": "w3a_run_attack.py (incremental/resumable; "
                          "same eval sets + attack code as run_attack.py)",
            }

            if args.resume and os.path.exists(fname):
                with open(fname) as f:
                    results = json.load(f)
                if results.get("config") != cfg:
                    sys.exit(f"config mismatch on resume for {fname}; "
                             f"use a new --out")
                print(f"resuming {fname} "
                      f"({sum(1 for k in results['per_scenario'][scen]['runs'][f'{mode}/genie'] if k.startswith('psr='))} "
                      f"genie points present)", flush=True)
            else:
                results = {
                    "experiment": "Compliant-Attacker (mask-constrained "
                                  "waveform PGD, corrected PSR axis)",
                    "config": cfg,
                    "per_scenario": {scen: {"clean_acc_active": None,
                                             "runs": {}, "summary": {}}},
                    "elapsed_s": 0.0,
                }

            scen_block = results["per_scenario"][scen]
            runs = scen_block["runs"]
            targeted = (mode == "targeted_noise")

            if scen_block["clean_acc_active"] is None:
                print(f"\n=== scenario: {scen} mode: {mode} ===", flush=True)
                x_rx, y, h_a, p_clean = build_eval_set(scen, args.n_eval,
                                                       args.seed)
                with torch.no_grad():
                    clean_preds = []
                    for i in range(0, x_rx.size(0), 256):
                        lg = model.forward_wave(x_rx[i:i + 256], frontend)
                        clean_preds.append(lg.argmax(1))
                    clean_preds = torch.cat(clean_preds)
                scen_block["clean_acc_active"] = round(
                    (clean_preds == y).float().mean().item(), 4)
                print(f"  clean acc on active tx: "
                      f"{scen_block['clean_acc_active']*100:.2f}%", flush=True)
            else:
                # rebuild the (deterministic) eval set for a resumed sweep
                x_rx, y, h_a, p_clean = build_eval_set(scen, args.n_eval,
                                                       args.seed)
                with torch.no_grad():
                    clean_preds = torch.cat([
                        model.forward_wave(x_rx[i:i + 256], frontend).argmax(1)
                        for i in range(0, x_rx.size(0), 256)])

            p_rx = p_clean

            # PSR-outer loop: every incremental save has PAIRED genie/mask
            # points, so MEAP/PoC in a partially-complete file are meaningful.
            for psr in psr_now:
                pk = f"psr={psr:+.0f}dB"
                for setting, sband in [("genie", None), ("cv2x_mask", band)]:
                    key = f"{mode}/{setting}"
                    runs.setdefault(key, {})
                    if pk in runs[key]:
                        print(f"    [skip] {key:26s} {pk} (done)", flush=True)
                        continue
                    p_budget = p_rx * (10 ** (psr / 10.0))
                    adv_preds = run_setting(model, frontend, x_rx, h_a, y,
                                            p_budget, sband, targeted,
                                            args.steps, args.psd_margin)
                    c_asr, n_elig = cond_asr(clean_preds, adv_preds, y,
                                             targeted=targeted,
                                             target=NOISE_CLASS)
                    rob = (adv_preds == y).float().mean().item()
                    runs[key][pk] = {
                        "cond_asr": round(100 * c_asr, 2),
                        "robust_acc": round(100 * rob, 2),
                        "n_eligible": n_elig,
                    }
                    print(f"    {key:26s} PSR {psr:+5.0f} dB: "
                          f"cond-ASR {100*c_asr:5.1f}%  "
                          f"robust {100*rob:5.1f}%", flush=True)
                    # ---- incremental write after EVERY condition ----
                    s = summarize(runs, mode, psr_grid)
                    if s is not None:
                        scen_block["summary"][mode] = s
                    results["elapsed_s"] = round(time.time() - t0, 1)
                    tmp = fname + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(results, f, indent=2)
                    os.replace(tmp, fname)

            s = summarize(runs, mode, psr_grid)
            if s is not None:
                scen_block["summary"][mode] = s
                results["elapsed_s"] = round(time.time() - t0, 1)
                with open(fname + ".tmp", "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(fname + ".tmp", fname)
            print(f"  wrote {fname}", flush=True)
            if s is not None:
                print(f"    MEAP genie {s['meap_genie_db']} dB "
                      f"({s['meap_genie_censor']})  mask "
                      f"{s['meap_cv2x_mask_db']} dB "
                      f"({s['meap_cv2x_mask_censor']})  "
                      f"PoC {s['price_of_compliance_db']} dB "
                      f"{('' if not s['poc_censor'] else s['poc_censor'])}",
                      flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
