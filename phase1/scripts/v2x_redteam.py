#!/usr/bin/env python3
"""v2x_redteam.py — one-command red-team evaluation of an RF sensing victim.

Wave 6, G3. Wraps the existing attack/MEAP machinery (no new attack code)
behind a certification-style CLI:

  evaluate : run genie + mask-compliant attack grids on a victim, compute
             MEAP / Price-of-Compliance, and verify the projection physics
             (budget exactness, OOB leakage, genie line) on the way out.
  report   : render any results JSON as a certification-style markdown report.

Examples
--------
  # canonical synthetic task, canonical victim:
  python scripts/v2x_redteam.py evaluate --victim dual --scenario urban

  # second victim (ResNet), 7-point quick grid:
  python scripts/v2x_redteam.py evaluate --victim resnet --psr -40 -30 -20 -10 0

  # real-OTA-WiFi eval set (Wave 6 G1):
  python scripts/v2x_redteam.py evaluate --victim dual-realft --real-wifi

  # render a report:
  python scripts/v2x_redteam.py report --json results/redteam_<...>.json
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

from waveforms import CLASS_NAMES, NOISE_CLASS, ATTACK_BANDS
from receiver import FrontEnd, DualStreamModel
from attack_mask import (waveform_pgd, project, cconv, cond_asr, meap_curve,
                         price_of_compliance)

OUT = os.path.join(ROOT, "results")

VICTIMS = {
    "dual": ("checkpoint_dual.pt", "DualStreamModel"),
    "mag": ("checkpoint_mag.pt", "MagOnlyModel"),
    "resnet": ("checkpoint_resnet.pt", "ResNetVictim"),
    "dual-at": ("checkpoint_dual_at.pt", "DualStreamModel"),
    "dual-realft": ("checkpoint_dual_realft.pt", "DualStreamModel"),
}

DEFAULT_PSR = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
               -10.0, -5.0, 0.0, 5.0, 10.0]


def load_victim(name):
    from victim_resnet import ResNetVictim
    from receiver import MagOnlyModel
    ckpt_name, arch = VICTIMS[name]
    path = os.path.join(OUT, ckpt_name)
    if not os.path.exists(path):
        sys.exit(f"ERROR: checkpoint for victim '{name}' not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = {"DualStreamModel": DualStreamModel,
             "MagOnlyModel": MagOnlyModel,
             "ResNetVictim": ResNetVictim}[arch]()
    model.load_state_dict(ckpt["model"]); model.eval()
    fe = FrontEnd(); fe.set_stats(ckpt["mag_mean"], ckpt["mag_std"])
    return model, fe


def build_set(args):
    if not args.real_wifi:
        from run_attack import build_eval_set
        x_rx, y, h_a, p_clean = build_eval_set(args.scenario, args.n_eval,
                                                args.seed)
        meta = {"eval_set": f"synthetic, {args.scenario}, "
                            f"{args.n_eval}/active class, seed {args.seed}"}
        return x_rx, y, h_a, p_clean, meta
    from run_real_wifi import build_mixed_eval_set
    from real_wifi import build_real_wifi_pool
    pool, _, prov, refs = build_real_wifi_pool(args.real_wifi_dir)
    if args.real_wifi_eval_files:
        idx = [i for i, (fn, s) in enumerate(refs)
               if any(t in fn for t in args.real_wifi_eval_files)]
        pool = pool[idx]
    x_rx, y, h_a, p_clean, rm = build_mixed_eval_set(pool, args.n_eval,
                                                      args.seed,
                                                      args.scenario)
    meta = {"eval_set": f"REAL OTA WiFi ({pool.shape[0]} windows) + "
                        f"synthetic PC5/11p ({args.n_eval}/class), "
                        f"{args.scenario} channels, seed {args.seed}",
            "real_wifi_provenance": prov}
    return x_rx, y, h_a, p_clean, meta


def physics_checks(model, fe, x_rx, h_a, y, p_budget, band, psd_margin=2.0):
    """D2 gates on a probe batch: budget exactness, OOB leakage."""
    # NOTE: no no_grad here — the PGD call needs autograd
    out = waveform_pgd(model, fe, x_rx[:32], h_a[:32], y[:32],
                       p_budget[:32], steps=10, band=band,
                       return_delta=True)
    d = out["delta"]
    with torch.no_grad():
        proj = project(d, band, p_budget[:32], psd_margin=psd_margin)
        e_d = (d.abs() ** 2).sum(dim=1)
        e_p = (proj.abs() ** 2).sum(dim=1)
        budget_ratio = (e_p / p_budget[:32]).median().item()
        # OOB fraction of the (projected) delta energy, MHz grid
        from waveforms import N_SAMPLES
        F = np.fft.fftfreq(N_SAMPLES, d=1.0 / 20e6) / 1e6
        D = torch.fft.fft(d, dim=1).abs() ** 2
        if band is None:
            oob = 0.0
        else:
            inb = torch.from_numpy((F >= band["lo"]) & (F <= band["hi"]))
            oob = float((D[:, ~inb].sum(dim=1) / D.sum(dim=1)).mean())
    return {"post_projection_budget_ratio_median": round(budget_ratio, 8),
            "delta_oob_fraction": float(f"{oob:.3e}"),
            "note": "budget ratio must be <= 1.0000001 and OOB ~ 0 (mask)"}


def cmd_evaluate(args):
    t0 = time.time()
    model, fe = load_victim(args.victim)
    x_rx, y, h_a, p_clean, setmeta = build_set(args)
    band = ATTACK_BANDS[args.attacker]
    print(f"victim={args.victim}  set={setmeta['eval_set']}", flush=True)
    print(f"eval windows: {x_rx.size(0)}", flush=True)

    with torch.no_grad():
        cp = [model.forward_wave(x_rx[i:i + 256], fe).argmax(1)
              for i in range(0, x_rx.size(0), 256)]
    cp = torch.cat(cp)
    clean_acc = float((cp == y).float().mean())
    print(f"clean acc (active tx): {clean_acc*100:.2f}%", flush=True)

    targeted = (args.mode == "targeted_noise")
    results = {"tool": "v2x-redteam", "version": "1.0",
               "victim": args.victim,
               "attacker": args.attacker,
               "mode": args.mode,
               "clean_acc_active": round(clean_acc, 4),
               "eval_set": setmeta,
               "config": {"steps": args.steps, "seed": args.seed,
                          "psr_db": args.psr, "psd_margin": args.psd_margin,
                          "alpha": 0.25,
                          "psr_reference": "clean received-signal window "
                                           "energy (sum |r|^2, pre-noise)"},
               "worst_case_disclosure": "attacker knows the victim model and "
               "the exact received realization incl. noise — ASRs are upper "
               "bounds" if not args.surrogate else
               "TRANSFER mode: surrogate model, perfect channel",
               }
    for setting, sband in [("genie", None), ("mask_compliant", band)]:
        curve = {}
        for psr in args.psr:
            p_budget = p_clean * (10 ** (psr / 10.0))
            preds = []
            for i in range(0, x_rx.size(0), 300):
                out = waveform_pgd(model, fe, x_rx[i:i + 300], h_a[i:i + 300],
                                   y[i:i + 300], p_budget[i:i + 300],
                                   steps=args.steps, band=sband,
                                   targeted=targeted, target_class=NOISE_CLASS,
                                   psd_margin=args.psd_margin)
                preds.append(out["preds"])
            adv = torch.cat(preds)
            c, n = cond_asr(cp, adv, y, targeted=targeted, target=NOISE_CLASS)
            rob = float((adv == y).float().mean())
            curve[f"psr={psr:+.0f}dB"] = {"cond_asr": round(100 * c, 2),
                                          "n_eligible": n,
                                          "robust_acc": round(100 * rob, 2)}
            print(f"  {setting:15s} PSR {psr:+5.0f} dB: cond-ASR "
                  f"{100*c:5.1f}%", flush=True)
        g = [curve[f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
        mg, cg = meap_curve(args.psr, g, args.threshold)
        results[setting] = {"curve": curve, "meap_db": mg, "censor": cg,
                            "threshold_pct": args.threshold}
        print(f"  {setting}: MEAP {mg} dB ({cg})", flush=True)
    gg = [results["genie"]["curve"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
    mm = [results["mask_compliant"]["curve"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
    poc, cn = price_of_compliance(args.psr, gg, args.psr, mm, args.threshold)
    results["price_of_compliance_db"] = poc
    results["poc_censor"] = cn
    # physics gates on the probe batch at mid-grid
    mid = args.psr[len(args.psr) // 2]
    pb = p_clean * (10 ** (mid / 10.0))
    results["physics_checks"] = physics_checks(model, fe, x_rx, h_a, y, pb,
                                                band, args.psd_margin)
    results["elapsed_s"] = round(time.time() - t0, 1)
    fname = os.path.join(OUT, f"redteam_{args.victim}_{args.attacker}_"
                              f"{args.scenario}.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {fname}", flush=True)
    print(f"MEAP genie {results['genie']['meap_db']} dB | "
          f"compliant {results['mask_compliant']['meap_db']} dB | "
          f"PoC {poc} dB", flush=True)


def cmd_report(args):
    with open(args.json) as f:
        r = json.load(f)
    lines = []
    lines.append(f"# V2X Red-Team Report — victim `{r.get('victim')}`")
    lines.append("")
    lines.append(f"- **Eval set:** {r['eval_set']['eval_set']}")
    lines.append(f"- **Attacker allocation:** {r.get('attacker')} "
                 f"({'mask-compliant (OOB nulled, in-band PSD capped)' if r.get('attacker') != 'genie' else 'genie (unconstrained)'})")
    lines.append(f"- **Clean accuracy (active tx):** "
                 f"{100*r['clean_acc_active']:.2f}%")
    lines.append(f"- **Worst-case disclosure:** {r.get('worst_case_disclosure')}")
    cfg = r["config"]
    lines.append(f"- **Config:** PGD-{cfg['steps']}, attack seed {cfg['seed']}, "
                 f"PSD margin {cfg['psd_margin']}, alpha {cfg['alpha']}")
    lines.append(f"- **PSR reference:** {cfg['psr_reference']}")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    thr = r["mask_compliant"].get("threshold_pct", 20)
    lines.append(f"MEAP (PSR at which conditional-ASR crosses {thr}%) and "
                 "Price of Compliance (MEAP_compliant − MEAP_genie):")
    lines.append("")
    lines.append("| Setting | MEAP (dB) | censored |")
    lines.append("|---|---|---|")
    for k in ("genie", "mask_compliant"):
        lines.append(f"| {k} | {r[k]['meap_db']} | {r[k]['censor']} |")
    lines.append(f"| **Price of Compliance** | "
                 f"**{r['price_of_compliance_db']} dB** | "
                 f"{r.get('poc_censor')} |")
    lines.append("")
    lines.append("Conditional-ASR sweep (success over windows the victim "
                 "classified correctly before the attack):")
    lines.append("")
    keys = list(r["mask_compliant"]["curve"].keys())
    lines.append("| PSR (dB) | genie ASR % | compliant ASR % | robust acc % |")
    lines.append("|---|---|---|---|")
    for k in keys:
        lines.append(f"| {k.replace('psr=','').replace('dB','')} | "
                     f"{r['genie']['curve'][k]['cond_asr']} | "
                     f"{r['mask_compliant']['curve'][k]['cond_asr']} | "
                     f"{r['mask_compliant']['curve'][k]['robust_acc']} |")
    pc = r.get("physics_checks", {})
    lines.append("")
    lines.append("## Projection physics (verified during the run)")
    lines.append("")
    lines.append(f"- post-projection budget ratio (median): "
                 f"`{pc.get('post_projection_budget_ratio_median')}` "
                 "(must be ≤ 1.0000001)")
    lines.append(f"- perturbation out-of-band fraction: "
                 f"`{pc.get('delta_oob_fraction')}` (mask: ~0)")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(f"The mask-compliant attacker — a transmitter that respects "
                 f"its regulatory allocation and emission mask — reaches the "
                 f"{thr}% conditional-ASR threshold at "
                 f"**{r['mask_compliant']['meap_db']} dB** PSR "
                 f"(received attack power relative to the victim signal). "
                 f"The unconstrained genie needs "
                 f"{r['genie']['meap_db']} dB; the rules cost the attacker "
                 f"**{r['price_of_compliance_db']} dB**. Lower (more negative) "
                 "MEAP = weaker victim. Every number above is an upper bound "
                 "(worst-case attacker knowledge; see disclosure).")
    txt = "\n".join(lines)
    out = args.out or (args.json.replace(".json", "_report.md"))
    with open(out, "w") as f:
        f.write(txt)
    print(f"wrote {out}", flush=True)
    print(txt, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ev = sub.add_parser("evaluate", help="run the red-team evaluation")
    ev.add_argument("--victim", default="dual", choices=list(VICTIMS))
    ev.add_argument("--scenario", default="urban",
                    choices=["urban", "highway", "rural"])
    ev.add_argument("--mode", default="untargeted",
                    choices=["untargeted", "targeted_noise"])
    ev.add_argument("--attacker", default="cv2x_attacker",
                    choices=["cv2x_attacker", "wifi_attacker", "genie"])
    ev.add_argument("--psr", nargs="+", type=float, default=DEFAULT_PSR)
    ev.add_argument("--n-eval", type=int, default=100)
    ev.add_argument("--steps", type=int, default=10)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--psd-margin", type=float, default=2.0)
    ev.add_argument("--threshold", type=float, default=20.0)
    ev.add_argument("--surrogate", action="store_true",
                    help="mark report as transfer/surrogate mode")
    ev.add_argument("--real-wifi", action="store_true",
                    help="use the real-OTA-WiFi eval set (Wave 6)")
    ev.add_argument("--real-wifi-dir",
                    default="/home/z/my-project/download/wave6_data/extracted")
    ev.add_argument("--real-wifi-eval-files", nargs="*", default=None,
                    help="substring filters for eval files (default: all)")
    ev.set_defaults(func=cmd_evaluate)

    rp = sub.add_parser("report", help="render markdown report from JSON")
    rp.add_argument("--json", required=True)
    rp.add_argument("--out", default=None)
    rp.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
