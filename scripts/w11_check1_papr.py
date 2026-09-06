#!/usr/bin/env python3
"""w11_check1_papr.py — W11 auditor CHECK 1 (independent PAPR re-derivation).

Re-generates 200 benign waveforms per class exactly the way run_papr_pa.py
did (single np.random.default_rng(123), cls 0..3 in order, 200 windows each),
but computes PAPR with THIS script's own numpy code:

    PAPR_dB = 10*log10( max_n |x_n|^2 / mean_n |x_n|^2 )

No torch, no import of any PA/PAPR helper from run_papr_pa.py.
Compares the means against results/papr_pa_results.json::papr_benign_db
(expected PC5 6.26, 11p 8.77, WiFi 8.69, Noise 9.02; task tolerance 0.15 dB
on PC5 and WiFi).
"""
import sys, os, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
from waveforms import gen_signal, CLASS_NAMES          # generator only

torch_free = "numpy-only PAPR implementation"

def papr_db_own(w):
    """w: (B, N) complex64 -> (B,) PAPR in dB, own implementation."""
    p = np.abs(w).astype(np.float64) ** 2              # per-sample power
    p_peak = p.max(axis=1)
    p_mean = p.mean(axis=1)
    return 10.0 * np.log10(p_peak / p_mean)

def main():
    rng = np.random.default_rng(123)                   # mirror the script
    waves = {}
    for cls in range(4):
        w = np.stack([gen_signal(cls, rng) for _ in range(200)])
        waves[CLASS_NAMES[cls]] = w

    res = json.load(open(os.path.join(ROOT, "results", "papr_pa_results.json")))
    stored = res["papr_benign_db"]

    print(f"own implementation: {torch_free}")
    print(f"{'class':14s} {'own mean':>9s} {'stored':>7s} {'diff':>7s} "
          f"{'own p95':>8s} {'stored p95':>10s}")
    ok = True
    for cls in range(4):
        name = CLASS_NAMES[cls]
        pr = papr_db_own(waves[name])
        m_own, p95_own = float(pr.mean()), float(np.percentile(pr, 95))
        m_st = stored[name]["mean_db"]
        p95_st = stored[name]["p95_db"]
        print(f"{name:14s} {m_own:9.3f} {m_st:7.2f} {m_own-m_st:+7.3f} "
              f"{p95_own:8.2f} {p95_st:10.2f}")
        if abs(m_own - m_st) > 0.15:
            ok = False
            print(f"  !! {name}: |diff| {abs(m_own-m_st):.3f} dB > 0.15 dB")
    # explicit task tolerances
    pc5 = abs(float(papr_db_own(waves['C-V2X-PC5']).mean()) - 6.26)
    wifi = abs(float(papr_db_own(waves['WiFi-U-NII4']).mean()) - 8.69)
    print(f"\nCHECK1 PC5  mean vs 6.26 : diff {pc5:.3f} dB  -> "
          f"{'PASS' if pc5 <= 0.15 else 'FAIL'}")
    print(f"CHECK1 WiFi mean vs 8.69 : diff {wifi:.3f} dB  -> "
          f"{'PASS' if wifi <= 0.15 else 'FAIL'}")
    print(f"overall: {'PASS' if ok else 'FAIL'}")

if __name__ == "__main__":
    main()
