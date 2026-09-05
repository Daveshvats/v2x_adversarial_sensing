#!/usr/bin/env python3
"""run_papr_pa.py — Wave 11-A: power-amplifier reality check for the Compliant Attacker.

THE QUESTION
  The paper's threat model claims a "regulator-verifiable" adversary: the
  attack waveform is projected onto the emission mask at the transmitter
  port, in floating point. A real transmitter has a power amplifier (PA):
  AM/AM nonlinearity causes SPECTRAL REGROWTH — energy re-appears out of
  band after the mask projection has already happened. If the optimized
  adversarial waveform has high PAPR, a real PA breaks its compliance and
  distorts it in-band. This experiment measures: (a) the PAPR the optimizer
  actually produces, (b) the backoff (IBO) a compliant attacker needs so the
  post-PA spectrum still meets the mask, and (c) what that backoff costs the
  attack (MEAP shift) — plus a PA-aware re-optimization that puts the Rapp
  model inside the attack chain.

PRE-REGISTERED HYPOTHESES (D5, written before running)
  H1  Optimized mask-constrained deltas have HIGHER PAPR than every benign
      class waveform (PGD exploits coherent time-domain peaks within the
      PSD cap). Prediction: mask-delta PAPR > max(PC5 6.1, 11p 8.8, WiFi
      8.6 dB), roughly 9-14 dB.
  H2  Post-PA OOB regrowth at low IBO violates the strict mask gate
      (-40 dBr peak OOB PSD); the minimal compliant backoff IBO* scales
      with PAPR, so the compliant attacker pays an EXTRA power penalty
      beyond the floating-point MEAP. Prediction: MEAP shift +0.5..+4 dB
      at IBO*, PA-aware re-optimization recovers <= half of it.
  H3  (KILL) If no IBO in [0,16] dB restores strict-gate compliance for
      95% of windows, or the MEAP shift at IBO* exceeds +6 dB even after
      PA-aware re-optimization, then the "regulator-verifiable in power
      units" claim is overstated by the floating-point model and the paper
      must be softened accordingly. Both outcomes are publishable; neither
      is hidden.

PA MODELS
  Rapp AM/AM (memoryless, smooth, differentiable):
      y = x * [1 + (|x|/As)^(2p)]^(-1/(2p)),   p in {2, 3}
  As = saturation amplitude set per window from the window's own input RMS
  and the requested IBO: IBO_dB = 10 log10(As^2 / P_in). Output energy is
  re-normalized to the attack budget (a real transmitter power-controls to
  its licensed output limit, so compression is compensated; this keeps PSR
  labels physical). OBO is reported. The soft limiter is p -> infinity and
  is approximated numerically for reference only (non-differentiable).

COMPLIANCE GATES (post-PA, transmit port, 2048-pt rectangular FFT of the
PA output — the same per-bin grid the projection itself uses)
  strict  : peak OOB PSD <= -40 dBr w.r.t. peak in-band PSD (802.11-style
            far-out mask level; also the paper's window-spread reference)
  lenient : peak OOB PSD <= -28 dBr (802.11 +/-11 MHz shoulder level)
  A window "passes" a gate if its post-PA spectrum meets it. IBO* = minimal
  IBO (fine 1 dB grid) where >= 95% of windows pass.

  NOTE (disclose): dBr regrowth of the Rapp output is scale-invariant given
  per-window IBO (Rapp is equivariant under As scaling), so IBO* does not
  depend on absolute PSR; it is a property of (waveform shape, p, IBO).

EVALUATION
  Canonical setup replicated exactly: urban scenario, n_eval=100/class
  (300 active windows), attack seed 7, PGD-10, alpha 0.25, psd_margin 2.0,
  untargeted, checkpoint_dual.pt (seed 42). PSR grid densified near the
  canonical MEAP (-37.1 dB) and extended down to -50 dB (de-censoring).
  The control (no PA) arm re-derives the canonical curve in-process
  (sanity: must match attack_results_urban_untargeted.json within
  interpolation resolution). Genie MEAP is taken as the canonical -44.22 dB
  (the genie is an idealization and is not PA-realistic by construction;
  disclosed).

CHUNKED / RESUMABLE EXECUTION
  --arms papr,regrowth,control,pa:6:3,aware:13:3
  Each invocation merges its results into the same JSON (never clobbers
  previously computed arms), so the study can run in foreground chunks.

OUTPUTS
  results/papr_pa_results.json — all curves, PAPR stats, IBO*, MEAP shifts,
  PA-aware arm, full config + seeds embedded (house style).
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch

from waveforms import (gen_signal, CLASS_NAMES, NOISE_CLASS, ATTACK_BANDS,
                       FS, N_SAMPLES)
import channels as CH
from receiver import FrontEnd, DualStreamModel
from attack_mask import (cconv, waveform_pgd, cond_asr, meap_curve, project)
from run_attack import build_eval_set   # canonical eval-set builder

OUT = os.path.join(ROOT, "results")

DEFAULT_PSR = [-50.0, -47.0, -44.0, -41.0, -38.0, -36.0, -34.0, -32.0,
               -30.0, -27.0, -24.0, -21.0, -18.0, -15.0, -12.0, -9.0,
               -6.0, -3.0, 0.0]

FREQS_MHZ = torch.fft.fftfreq(N_SAMPLES, d=1.0 / FS) / 1e6   # (N,)
STRICT_DBRR, LENIENT_DBRR = -40.0, -28.0
PASS_FRAC = 0.95


# ---------------------------------------------------------------------------
# PA models
# ---------------------------------------------------------------------------
def rapp_pa(x: torch.Tensor, asat: torch.Tensor, p: float) -> torch.Tensor:
    """Rapp AM/AM. x: (B, N) complex; asat: (B,) saturation amplitude.
    Differentiable (smooth) — usable inside the PGD chain."""
    r = x.abs()
    g = (1.0 + (r / asat.unsqueeze(1)) ** (2.0 * p)) ** (-1.0 / (2.0 * p))
    return x * g


def soft_limiter(x: torch.Tensor, asat: torch.Tensor) -> torch.Tensor:
    """Memoryless soft limiter (reference; non-differentiable a.e.)."""
    r = x.abs()
    scale = torch.clamp(asat.unsqueeze(1) / (r + 1e-12), max=1.0)
    return x * scale


def apply_pa_budget(x: torch.Tensor, ibo_db: float, p: float,
                    limiter: bool = False):
    """Apply PA at per-window IBO, then renormalize output energy to the
    input's (power control compensates compression; keeps PSR physical).

    Returns (y, obo_db) where obo_db = 10log10(As^2 / P_out) per window.
    """
    p_in = (x.abs() ** 2).mean(dim=1)                       # (B,)
    asat = torch.sqrt(p_in * (10.0 ** (ibo_db / 10.0)) + 1e-20)
    y = soft_limiter(x, asat) if limiter else rapp_pa(x, asat, p)
    p_out = (y.abs() ** 2).mean(dim=1)
    # renormalize output window energy to the input's (power control)
    e_in = (x.abs() ** 2).sum(dim=1)
    e_out = (y.abs() ** 2).sum(dim=1)
    y = y * torch.sqrt(e_in / (e_out + 1e-20)).unsqueeze(1)
    obo = 10.0 * torch.log10(asat ** 2 / (p_out + 1e-20))
    return y, obo


# ---------------------------------------------------------------------------
# PAPR + post-PA spectrum metrics
# ---------------------------------------------------------------------------
def papr_db(x: torch.Tensor) -> torch.Tensor:
    """(B,) per-window PAPR in dB (uniform sampling, no oversampling —
    disclose; oversampled PAPR would be >= this)."""
    p_peak = (x.abs() ** 2).max(dim=1).values
    p_mean = (x.abs() ** 2).mean(dim=1)
    return 10.0 * torch.log10(p_peak / (p_mean + 1e-20))


def mask_excess_dbr(y: torch.Tensor, band: dict) -> torch.Tensor:
    """(B,) post-PA peak OOB PSD relative to peak in-band PSD (dBr), using
    the same per-bin FFT grid the projection uses. Negative = suppressed."""
    Y = torch.fft.fft(y, dim=1)
    f = FREQS_MHZ.to(y.device).unsqueeze(0)
    inb = (f >= band["lo"]) & (f <= band["hi"])
    pbin = Y.abs() ** 2                                    # (B, N)
    peak_in = pbin.masked_fill(~inb, 0).max(dim=1).values
    peak_out = pbin.masked_fill(inb, 0).max(dim=1).values
    return 10.0 * torch.log10(peak_out / (peak_in + 1e-20))


def oob_fraction_db(y: torch.Tensor, band: dict) -> torch.Tensor:
    """(B,) 10log10(P_oob / P_total) of the PA output."""
    Y = torch.fft.fft(y, dim=1)
    f = FREQS_MHZ.to(y.device).unsqueeze(0)
    inb = (f >= band["lo"]) & (f <= band["hi"])
    p = (Y.abs() ** 2)
    p_oob = p.masked_fill(inb, 0).sum(dim=1)
    p_tot = p.sum(dim=1)
    return 10.0 * torch.log10(p_oob / (p_tot + 1e-20))


def pa_aware_pgd(model, frontend, x_rx, h_a, y, p_budget, band, steps,
                 ibo_db, rapp_p, alpha_frac=0.25, psd_margin=2.0,
                 targeted=False):
    """PGD with the Rapp PA INSIDE the differentiable chain:
        delta -> project(mask) -> Rapp(IBO) -> renorm to budget -> channel
    The optimizer sees the PA. Projection stays at the transmit port
    (pre-PA) — post-PA compliance is then MEASURED, not enforced (honest).
    """
    B = x_rx.size(0)
    delta = torch.zeros_like(x_rx, requires_grad=True)
    h = torch.as_tensor(h_a, dtype=x_rx.dtype)

    def chain(d):
        d = project(d, band, p_budget, psd_margin=psd_margin)
        # PA saturation is sized to the INTENDED transmit MEAN POWER
        # (budget / N per sample — the same per-sample IBO definition as
        # apply_pa_budget; sizing to window energy would add +33.11 dB of
        # phantom backoff, the same unit class as the Wave-3 PSR bug) and
        # stays nonzero at d=0 (the 2026-09-05 probe found that sizing
        # asat to the iterate's own power made the renorm term 0/0 at the
        # zero initialization, poisoning the first gradient with NaN ->
        # argmax-of-NaN garbage that mimicked a 13 dB "improvement").
        asat = torch.sqrt(p_budget / N_SAMPLES *
                          (10.0 ** (ibo_db / 10.0)) + 1e-20)
        yv = rapp_pa(d, asat, rapp_p)
        # power control to the licensed limit: renormalize to BUDGET
        # (eps relative to budget keeps the gradient finite at d=0)
        e_out = (yv.abs() ** 2).sum(dim=1)
        eps = 1e-12 * p_budget
        yv = yv * torch.sqrt(p_budget / (e_out + eps)).unsqueeze(1)
        return yv

    for _ in range(steps):
        yv = chain(delta)
        r = x_rx + cconv(yv, h)
        logits = model.forward_wave(r, frontend)
        if targeted:
            loss = torch.nn.functional.cross_entropy(
                logits, torch.full_like(y, NOISE_CLASS))
        else:
            loss = -torch.nn.functional.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, delta)
        gn = g / (g.abs().norm(dim=1, keepdim=True) + 1e-12)
        step = alpha_frac * p_budget.sqrt().unsqueeze(1) * gn
        with torch.no_grad():
            delta = delta - step
            delta = project(delta, band, p_budget, psd_margin=psd_margin)
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        yv = chain(delta)
        r = x_rx + cconv(yv, h)
        logits = model.forward_wave(r, frontend)
        preds = logits.argmax(1)
    return {"preds": preds, "delta": delta.detach(), "y_tx": yv.detach()}


# ---------------------------------------------------------------------------
def run_arm(model, frontend, x_rx, h_a, y, p_budget, band, clean_preds,
            psr_list, targeted, steps, psd_margin, pa=None, pa_aware=False):
    """One arm over the PSR grid. pa=None -> no-PA control; else
    (ibo_db, rapp_p, limiter). Returns (curve, tx_ref, stats)."""
    preds_all, tx_ref = [], None
    B = 300
    ref_psr = -36.0 if -36.0 in psr_list else psr_list[len(psr_list) // 2]
    for psr in psr_list:
        pb = p_budget * (10.0 ** (psr / 10.0))
        preds = []
        for i in range(0, x_rx.size(0), B):
            xb, hb, yb, pbb = x_rx[i:i+B], h_a[i:i+B], y[i:i+B], pb[i:i+B]
            if pa_aware:
                out = pa_aware_pgd(model, frontend, xb, hb, yb, pbb, band,
                                   steps, pa[0], pa[1])
                preds.append(out["preds"])
                if abs(psr - ref_psr) < 1e-9:
                    tx_ref = out["y_tx"]     # post-PA transmit waveform
            else:
                out = waveform_pgd(model, frontend, xb, hb, yb, pbb,
                                   steps=steps, band=band, targeted=targeted,
                                   target_class=NOISE_CLASS,
                                   psd_margin=psd_margin, return_delta=True)
                d = out["delta"]
                if pa is not None:
                    ibo, rp, lim = pa
                    d, _ = apply_pa_budget(d, ibo, rp, limiter=lim)
                    r = xb + cconv(d, hb)
                    lg = model.forward_wave(r, frontend)
                    preds.append(lg.argmax(1))
                else:
                    preds.append(out["preds"])
                if abs(psr - ref_psr) < 1e-9:
                    tx_ref = d
        preds_all.append(torch.cat(preds))
    curve = []
    for psr, pr in zip(psr_list, preds_all):
        c_asr, n_elig = cond_asr(clean_preds, pr, y, targeted=targeted,
                                 target=NOISE_CLASS)
        curve.append(round(100 * c_asr, 2))
    stats = {"ref_psr_db": ref_psr}
    if tx_ref is not None:
        me = mask_excess_dbr(tx_ref, band)
        stats["post_tx_mask_excess_p95_dbr"] = round(
            float(me.quantile(0.95)), 2)
        stats["post_tx_pass_strict"] = round(float(
            (me <= STRICT_DBRR).float().mean()), 4)
        stats["post_tx_papr_mean_db"] = round(
            float(papr_db(tx_ref).mean()), 2)
        stats["post_tx_papr_p95_db"] = round(
            float(papr_db(tx_ref).quantile(0.95)), 2)
    return curve, tx_ref, stats


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--psr", nargs="+", type=float, default=DEFAULT_PSR)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--psd-margin", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--scenario", default="urban")
    ap.add_argument("--checkpoint", default="checkpoint_dual.pt")
    ap.add_argument("--rapp-p", nargs="+", type=float, default=[2.0, 3.0])
    ap.add_argument("--ibo-sweep", nargs="+", type=float,
                    default=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    ap.add_argument("--arms", default="all",
                    help="comma list from: papr,regrowth,control,pa:IBO[:p],"
                         "aware:IBO[:p],all — chunked/resumable execution;"
                         " results merge into the same JSON (no clobber)")
    ap.add_argument("--quick", action="store_true",
                    help="smoke: 30/class, short PSR grid, p=3 only")
    ap.add_argument("--tag-out", default="")
    args = ap.parse_args()
    if args.quick:
        args.n_eval, args.rapp_p = 30, [3.0]
        args.ibo_sweep = [0.0, 6.0, 12.0]
        args.psr = [-40.0, -36.0, -32.0, -28.0, -24.0, -20.0]
        args.steps = 5

    torch.set_num_threads(2)     # sandbox protocol: 2 threads, bounded RSS
    t0 = time.time()

    ckpt = torch.load(os.path.join(OUT, args.checkpoint),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    band = ATTACK_BANDS["cv2x_attacker"]
    x_rx, y, h_a, p_clean = build_eval_set(args.scenario, args.n_eval,
                                           args.seed)
    with torch.no_grad():
        clean_preds = []
        for i in range(0, x_rx.size(0), 256):
            lg = model.forward_wave(x_rx[i:i+256], frontend)
            clean_preds.append(lg.argmax(1))
        clean_preds = torch.cat(clean_preds)
    clean_acc = (clean_preds == y).float().mean().item()
    print(f"clean acc on active tx: {clean_acc*100:.2f}%  "
          f"(n={x_rx.size(0)})", flush=True)

    res = {
        "experiment": "Wave 11-A: PAPR + PA-regrowth reality check for the "
                      "Compliant Attacker (Rapp model, IBO sweep, PA-aware "
                      "re-optimization)",
        "preregistered_hypotheses": ["H1 papr_adv > benign classes",
                                     "H2 compliant backoff costs 0.5-4 dB "
                                     "MEAP; PA-aware recovers <= half",
                                     "H3 KILL if IBO* > 16 dB unreachable "
                                     "or MEAP shift > 6 dB post-reopt"],
        "config": {
            "scenario": args.scenario, "n_eval_per_class": args.n_eval,
            "attack_seed": args.seed, "steps": args.steps,
            "alpha_frac": args.alpha, "psd_cap_margin": args.psd_margin,
            "psr_db": args.psr, "rapp_p": args.rapp_p,
            "ibo_sweep_db": args.ibo_sweep,
            "mask_band_mhz": band,
            "gates_dbr": {"strict": STRICT_DBRR, "lenient": LENIENT_DBRR},
            "pass_fraction_required": PASS_FRAC,
            "papr_note": "no oversampling (uniform samples); oversampled "
                         "PAPR would be higher — disclosed",
            "genie_reference": "canonical -44.22 dB (urban, untargeted, "
                               "seed 7) from attack_results_urban_untargeted"
                               ".json; genie is an idealization, not "
                               "PA-realistic (disclosed)",
            "power_control": "post-PA window energy renormalized to the "
                             "attack budget (models transmitter power "
                             "control; compression compensated; OBO "
                             "reported)",
        },
        "clean_acc_active": round(clean_acc, 4),
    }

    fname = os.path.join(OUT, f"papr_pa_results{args.tag_out}.json")
    prev_state = {}
    if os.path.exists(fname):
        try:
            prev_state = json.load(open(fname))
        except Exception:
            prev_state = {}

    def flush():
        """merge-on-write: never clobber previously computed arms.
        NOTE: prev curves must be read from prev_state BEFORE
        merged.update(res) — res['curves'] holds only this invocation's
        arms, and updating first would silently drop all earlier arms
        (caught after the 2026-09-05 run; deterministic seeds make the
        lost arms re-computable)."""
        merged = dict(prev_state)
        merged.update(res)
        pc = dict(prev_state.get("curves", {}))
        pc.update(res.get("curves", {}))
        merged["curves"] = pc
        merged["elapsed_s"] = round(
            float(merged.get("elapsed_s", 0.0)) + (time.time() - t0), 1)
        with open(fname, "w") as f:
            json.dump(merged, f, indent=2)

    armset = {a.strip() for a in args.arms.split(",") if a.strip()}
    want_all = ("all" in armset)

    def parse_arm(tok):
        """pa:IBO[:p] / aware:IBO[:p] -> (kind, ibo, p)"""
        parts = tok.split(":")
        kind = parts[0]
        ibo = float(parts[1]) if len(parts) > 1 else 8.0
        p = float(parts[2]) if len(parts) > 2 else 3.0
        return kind, ibo, p

    pa_arms = [parse_arm(a) for a in armset if a.startswith("pa:")]
    aware_arms = [parse_arm(a) for a in armset if a.startswith("aware:")]

    do_papr = want_all or ("papr" in armset)
    do_reg = want_all or ("regrowth" in armset)
    do_ctrl = want_all or ("control" in armset)

    # ------------------------------------------------------------------
    # 1) PAPR of benign classes vs optimized deltas (H1)
    # ------------------------------------------------------------------
    benign_waves, benign_papr = {}, {}
    if do_papr or do_reg:
        print("\n=== PAPR of benign class waveforms (transmit-side) ===",
              flush=True)
        rng = np.random.default_rng(123)
        for cls in range(4):
            w = [torch.from_numpy(gen_signal(cls, rng).astype(np.complex64))
                 for _ in range(200)]
            w = torch.stack(w)
            benign_waves[CLASS_NAMES[cls]] = w
            pr = papr_db(w)
            benign_papr[CLASS_NAMES[cls]] = {
                "mean_db": round(float(pr.mean()), 2),
                "p95_db": round(float(pr.quantile(0.95)), 2),
            }
            print(f"  {CLASS_NAMES[cls]:14s} PAPR mean {pr.mean():5.2f} dB  "
                  f"p95 {pr.quantile(0.95):5.2f} dB", flush=True)
        res["papr_benign_db"] = benign_papr
        flush()

    delta_store = {}
    if do_papr or do_reg:
        print("\n=== optimizing mask + genie deltas (PAPR + regrowth source)"
              " ===", flush=True)
        psr_papr = [-36.0, -25.0] if do_papr else [-36.0]
        names = [("mask", band), ("genie", None)] if do_papr \
            else [("mask", band)]
        papr_opt = {"mask": {}, "genie": {}}
        for name, b in names:
            for psr in psr_papr:
                pb = p_clean * (10.0 ** (psr / 10.0))
                ds = []
                for i in range(0, x_rx.size(0), 300):
                    out = waveform_pgd(model, frontend, x_rx[i:i+300],
                                       h_a[i:i+300], y[i:i+300],
                                       pb[i:i+300], steps=args.steps,
                                       band=b, psd_margin=args.psd_margin,
                                       return_delta=True)
                    ds.append(out["delta"])
                d = torch.cat(ds)
                pr = papr_db(d)
                papr_opt[name][f"psr={psr:+.0f}dB"] = {
                    "mean_db": round(float(pr.mean()), 2),
                    "p95_db": round(float(pr.quantile(0.95)), 2),
                    "std_db": round(float(pr.std()), 2),
                }
                delta_store[(name, psr)] = d
                print(f"  {name:5s} PSR {psr:+5.0f} dB: PAPR mean "
                      f"{pr.mean():5.2f} dB  p95 {pr.quantile(0.95):5.2f} dB",
                      flush=True)
        if do_papr:
            res["papr_optimized_db"] = papr_opt
            h1_mask = papr_opt["mask"]["psr=-36dB"]["mean_db"]
            h1_benign = max(v["mean_db"] for v in benign_papr.values())
            res["H1"] = {"adv_papr_db": h1_mask, "benign_max_db": h1_benign,
                         "confirmed": bool(h1_mask > h1_benign)}
        flush()

    # ------------------------------------------------------------------
    # 2) regrowth vs IBO (scale-invariant pre-computation, H2 part 1)
    # ------------------------------------------------------------------
    if do_reg:
        print("\n=== post-PA regrowth vs IBO (mask deltas @ -36 dB PSR) ===",
              flush=True)
        d_ref = delta_store[("mask", -36.0)]
        regrowth = {}
        for p in args.rapp_p:
            for ibo in args.ibo_sweep:
                y_pa, obo = apply_pa_budget(d_ref, ibo, p)
                me = mask_excess_dbr(y_pa, band)
                of = oob_fraction_db(y_pa, band)
                frac_strict = float((me <= STRICT_DBRR).float().mean())
                frac_len = float((me <= LENIENT_DBRR).float().mean())
                regrowth[f"p={p:g}/ibo={ibo:g}"] = {
                    "mask_excess_p50_dbr": round(float(me.median()), 2),
                    "mask_excess_p95_dbr": round(float(me.quantile(0.95)), 2),
                    "oob_frac_p50_db": round(float(of.median()), 2),
                    "pass_strict": round(frac_strict, 4),
                    "pass_lenient": round(frac_len, 4),
                    "obo_mean_db": round(float(obo.mean()), 2),
                    "papr_out_mean_db": round(float(papr_db(y_pa).mean()), 2),
                }
                print(f"  p={p:g} IBO {ibo:4.0f} dB: excess p95 "
                      f"{me.quantile(0.95):6.1f} dBr  strict-pass "
                      f"{frac_strict*100:5.1f}%  OBO {obo.mean():4.1f} dB",
                      flush=True)
        # soft limiter reference at 6 dB IBO
        yl, obol = apply_pa_budget(d_ref, 6.0, 3.0, limiter=True)
        mel = mask_excess_dbr(yl, band)
        regrowth["limiter/ibo=6"] = {
            "mask_excess_p95_dbr": round(float(mel.quantile(0.95)), 2),
            "pass_strict": round(
                float((mel <= STRICT_DBRR).float().mean()), 4),
            "obo_mean_db": round(float(obol.mean()), 2),
        }
        res["regrowth"] = regrowth

        # fine IBO* search per p (1 dB grid extended to 16 dB)
        ibo_star = {}
        for p in args.rapp_p:
            fine = np.arange(0.0, 16.1, 1.0)
            star = None
            for ibo in fine:
                y_pa, _ = apply_pa_budget(d_ref, float(ibo), p)
                me = mask_excess_dbr(y_pa, band)
                if float((me <= STRICT_DBRR).float().mean()) >= PASS_FRAC:
                    star = float(ibo)
                    break
            ibo_star[f"p={p:g}"] = star
            print(f"  IBO* (strict gate, 95% windows) p={p:g}: "
                  f"{star if star is not None else 'NOT REACHED <= 16 dB'}",
                  flush=True)
        res["ibo_star_strict_db"] = ibo_star

        # benign regrowth INCREMENT comparison — the honest like-for-like:
        # benign generators carry rectangular-window skirts of -30..-40 dBr
        # pre-PA (a known synthetic limitation), so TOTAL OOB gates are not
        # comparable; the PA's incremental damage is. The mask-projected
        # attack's pre-PA OOB is numerically zero, so its post-PA pedestal
        # is pure regrowth and IS comparable to the benign skirt level.
        benign_band = {"C-V2X-PC5": band, "802.11p": band,
                       "WiFi-U-NII4": {"lo": -10.0, "hi": 0.0}}
        benign_regrowth = {}
        for cname, wv in benign_waves.items():
            if cname not in benign_band:
                continue
            b = benign_band[cname]
            me_pre = mask_excess_dbr(wv, b)
            row = {"pre_p95_dbr": round(float(me_pre.quantile(0.95)), 2)}
            for ibo in (0.0, 6.0, 12.0):
                y_pa, _ = apply_pa_budget(wv, ibo, 3.0)
                me_post = mask_excess_dbr(y_pa, b)
                row[f"post_ibo{ibo:g}_p95_dbr"] = round(
                    float(me_post.quantile(0.95)), 2)
                row[f"increment_ibo{ibo:g}_p95_db"] = round(float(
                    me_post.quantile(0.95) - me_pre.quantile(0.95)), 2)
            benign_regrowth[cname] = row
            print(f"  benign {cname:14s}: pre skirt p95 "
                  f"{row['pre_p95_dbr']:6.1f} dBr; increment @IBO6 "
                  f"{row['increment_ibo6_p95_db']:+5.1f} dB", flush=True)
        res["benign_regrowth_increment"] = benign_regrowth
        flush()

    # ------------------------------------------------------------------
    # 3) attack effectiveness through the PA (MEAP shift), H2 part 2
    # ------------------------------------------------------------------
    curves = {}
    m_ctrl = None
    if "control_meap_db" in prev_state:
        m_ctrl = prev_state["control_meap_db"]

    if do_ctrl:
        print("\n=== cond-ASR curves (urban, untargeted) ===", flush=True)
        t = time.time()
        ctrl_curve, _, ctrl_stats = run_arm(model, frontend, x_rx, h_a, y,
                                            p_clean, band, clean_preds,
                                            args.psr, False, args.steps,
                                            args.psd_margin, pa=None)
        m_ctrl, c_ctrl = meap_curve(args.psr, ctrl_curve, 20.0)
        curves["control_no_pa"] = ctrl_curve
        res["control_meap_db"] = m_ctrl
        res["control_censor"] = c_ctrl
        print(f"  control (no PA): MEAP {m_ctrl} dB ({c_ctrl})  "
              f"[{time.time()-t:.0f}s]", flush=True)

    def run_pa_arm(ibo, p, aware):
        key = (f"pa_aware_p{p:g}_ibo{ibo:g}" if aware
               else f"pa_p{p:g}_ibo{ibo:g}")
        t = time.time()
        cur, _, st = run_arm(model, frontend, x_rx, h_a, y, p_clean, band,
                             clean_preds, args.psr, False, args.steps,
                             args.psd_margin, pa=(ibo, p, False),
                             pa_aware=aware)
        m, c = meap_curve(args.psr, cur, 20.0)
        entry = {"meap_db": m, "censor": c, **st}
        if m is not None and m_ctrl is not None:
            entry["shift_vs_control_db"] = round(m - m_ctrl, 2)
        if aware:
            entry["note"] = ("Rapp inside the PGD chain; projection pre-PA; "
                             "post-PA compliance reported in stats")
        curves[key] = cur
        curves[key + "__meap"] = entry
        sh = entry.get("shift_vs_control_db")
        print(f"  {key:18s}: MEAP {m} dB ({c})  shift "
              f"{('%+.2f dB' % sh) if sh is not None else 'n/a'}"
              f"  [{time.time()-t:.0f}s]", flush=True)
        res["curves"] = curves
        flush()

    if want_all:
        # full sweep: default behavior (all IBOs + star + aware per p)
        ibo_star = res.get("ibo_star_strict_db",
                           prev_state.get("ibo_star_strict_db", {}))
        for p in args.rapp_p:
            star = ibo_star.get(f"p={p:g}")
            ibos = sorted(set(list(args.ibo_sweep) +
                              ([star] if star is not None else [])))
            for ibo in ibos:
                if ibo < 0:
                    continue
                run_pa_arm(ibo, p, aware=False)
        for p in args.rapp_p:
            star = ibo_star.get(f"p={p:g}")
            ibo = star if star is not None else 8.0
            run_pa_arm(ibo, p, aware=True)
    else:
        for kind, ibo, p in pa_arms:
            run_pa_arm(ibo, p, aware=False)
        for kind, ibo, p in aware_arms:
            run_pa_arm(ibo, p, aware=True)

    res["curves"] = curves
    res["psr_grid"] = args.psr
    flush()
    print(f"\nwrote {fname} (merged)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
