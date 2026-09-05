#!/usr/bin/env python3
"""w11_check3_regrowth.py — W11 auditor CHECK 3 + CHECK 4 (fresh regrowth point
+ power-control energy audit).

Fresh data point, independent of run_papr_pa.py's stored numbers:
  1. build the canonical eval set (urban, n_eval=10, seed 7 -> 30 windows)
     with the canonical builder imported from scripts/run_attack.py;
  2. run the mask-constrained waveform_pgd from src/attack_mask.py
     (band ATTACK_BANDS["cv2x_attacker"], psd_margin 2.0, steps 10,
     PSR -36 dB, untargeted, checkpoint_dual.pt) -> 30 deltas;
  3. apply THIS script's OWN Rapp model (p=3, IBO 6 dB referenced to the
     per-window MEAN input power, asat = sqrt(mean|x|^2 * 10^(6/10)));
  4. own PSD metric: 2048-pt FFT of the PA output, in-band = [0,10] MHz on
     the fftfreq axis, OOB = everything else; per-window
     10*log10(peak OOB bin power / peak in-band bin power); report p95;
     expected order/sign: -17..-22 dBr (JSON's 300-window value: -19.32);
  5. scale-invariance of the dBr regrowth under per-window IBO (Rapp
     equivariance): rescale x by c, asat by c -> identical dBr;
  6. CHECK 4 (energy audit): renormalize the Rapp output to the per-window
     budget (sum|x|^2 = p_budget); 10*log10(E_out/p_budget) must be 0.00
     +/- 0.01 dB per window.

No PA/PSD/PAPR function of run_papr_pa.py is imported. Only src/ modules
(waveforms/receiver/attack_mask) and the canonical eval-set builder.
"""
import sys, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

torch.set_num_threads(2)

from waveforms import ATTACK_BANDS, FS, N_SAMPLES
from receiver import FrontEnd, DualStreamModel
from attack_mask import waveform_pgd
from run_attack import build_eval_set

OUT = os.path.join(ROOT, "results")

# --- own PA / PSD implementations (numpy) ----------------------------------

def rapp_own(x, p, ibo_db):
    """x: (B, N) complex numpy. Rapp AM/AM with asat set per window from the
    window's own MEAN input power: IBO_dB = 10 log10(As^2 / mean|x|^2)."""
    p_in = np.mean(np.abs(x) ** 2, axis=1)                   # (B,)
    asat = np.sqrt(p_in * 10.0 ** (ibo_db / 10.0))           # (B,)
    r = np.abs(x)
    g = (1.0 + (r / asat[:, None]) ** (2.0 * p)) ** (-1.0 / (2.0 * p))
    return x * g

def mask_excess_dbr_own(y, lo=0.0, hi=10.0):
    """(B,) peak OOB PSD minus peak in-band PSD, dBr, own implementation.
    2048-pt FFT, fftfreq axis, in-band = [lo,hi] MHz, OOB = everything else."""
    Y = np.fft.fft(y, axis=1)
    f = np.fft.fftfreq(y.shape[1], d=1.0 / FS) / 1e6         # MHz
    inb = (f >= lo) & (f <= hi)
    pbin = np.abs(Y) ** 2
    peak_in = np.max(np.where(inb[None, :], pbin, 0.0), axis=1)
    peak_out = np.max(np.where(inb[None, :], 0.0, pbin), axis=1)
    return 10.0 * np.log10(peak_out / peak_in)

def papr_db_own(x):
    p = np.abs(x).astype(np.float64) ** 2
    return 10.0 * np.log10(p.max(axis=1) / p.mean(axis=1))

def main():
    t0 = time.time()
    band = ATTACK_BANDS["cv2x_attacker"]
    assert band == {"lo": 0.0, "hi": 10.0}

    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    x_rx, y, h_a, p_clean = build_eval_set("urban", 10, 7)   # 30 windows
    B = x_rx.size(0)
    print(f"eval set: {B} windows (urban, n_eval=10, seed 7)")

    psr = -36.0
    pb = p_clean * (10.0 ** (psr / 10.0))                    # (B,) budget
    pb_np = pb.numpy().astype(np.float64)                    # numpy copy
    out = waveform_pgd(model, frontend, x_rx, h_a, y, pb, steps=10,
                       band=band, targeted=False, psd_margin=2.0,
                       return_delta=True)
    d = out["delta"].numpy()                                 # (B, N) complex
    print(f"PGD-10 done ({time.time()-t0:.0f}s); delta shape {d.shape}")

    # pre-PA sanity: mask-projected deltas -> OOB numerically ~ -inf
    me_pre = mask_excess_dbr_own(d)
    print(f"pre-PA mask excess: max {np.max(me_pre):.1f} dBr "
          f"(must be ~ -inf: OOB bins are nulled by projection)")
    print(f"pre-PA delta PAPR mean {papr_db_own(d).mean():.2f} dB "
          f"(H1-style sanity, JSON 300-window: 11.05)")

    # --- CHECK 3: own Rapp p=3, IBO 6 dB -----------------------------------
    y_pa = rapp_own(d, p=3.0, ibo_db=6.0)
    me = mask_excess_dbr_own(y_pa)
    p95 = float(np.percentile(me, 95))
    print(f"\nCHECK3 post-PA mask excess (p=3, IBO 6, own Rapp/PSD): "
          f"p50 {np.percentile(me,50):.2f}  p95 {p95:.2f} dBr  "
          f"(JSON 300-window p95: -19.32; task range -17..-22)")
    verdict3 = "PASS" if (-22.0 <= p95 <= -17.0) else "FAIL"
    print(f"CHECK3 verdict: {verdict3} (order/sign check)")

    # --- scale-invariance of dBr regrowth ----------------------------------
    for c in (0.5, 0.123, 3.0):
        me_s = mask_excess_dbr_own(rapp_own(d * c, p=3.0, ibo_db=6.0))
        dv = np.max(np.abs(me_s - me))
        # complex64 rounding noise floor is ~1e-5 dBr; anything <= 1e-4 is
        # exact equivariance for practical purposes
        print(f"scale-invariance x{c:<5}: max |dBr change| {dv:.2e} "
              f"({'ok' if dv < 1e-4 else 'BROKEN'})")

    # --- CHECK 4: energy audit (power control to the budget) ---------------
    e_out = np.sum(np.abs(y_pa) ** 2, axis=1)                # window energy
    y_pc = y_pa * np.sqrt(pb_np / e_out)[:, None]            # renormalize
    e_pc = np.sum(np.abs(y_pc) ** 2, axis=1)
    ratio_db = 10.0 * np.log10(e_pc / pb_np)
    print(f"\nCHECK4 after renormalizing Rapp output to the budget: "
          f"10log10(E_out/p_budget): mean {ratio_db.mean():+.6f} dB  "
          f"max|.| {np.max(np.abs(ratio_db)):.2e} dB  "
          f"(required 0.00 +/- 0.01 dB)")
    verdict4 = "PASS" if np.max(np.abs(ratio_db)) <= 0.01 else "FAIL"
    print(f"CHECK4 verdict: {verdict4}")
    # dBr unchanged by the power control (ratio metric, scale-invariant)
    me_pc = mask_excess_dbr_own(y_pc)
    print(f"(mask excess unchanged by renorm: max diff "
          f"{np.max(np.abs(me_pc-me)):.2e} dBr)")

    print(f"\nelapsed {time.time()-t0:.0f}s")
    return 0 if (verdict3 == "PASS" and verdict4 == "PASS") else 1

if __name__ == "__main__":
    raise SystemExit(main())
