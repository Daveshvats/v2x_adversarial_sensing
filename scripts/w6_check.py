#!/usr/bin/env python3
"""w6_check.py — Wave-6 D2 physics gates for the real-WiFi loader.

Runs BEFORE any real-wifi number is quoted anywhere:
  1. format sanity (float32 I/Q, sizes)
  2. DC removal effectiveness (pre/post DC bin power)
  3. occupancy in [-10, 0] MHz after placement (>= 90% gate)
  4. burst-gate statistics per file (kept fraction, floor, threshold)
  5. PAPR of real segments vs synthetic WiFi class (OFDM envelope sanity)
  6. native SNR estimate of the captures (signal floor vs idle floor)
"""
import sys, os, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # repo root
sys.path.insert(0, os.path.join(ROOT, "src"))

DATA = "/home/z/my-project/download/wave6_data/extracted"
from real_wifi import (load_capture, active_segments, build_real_wifi_pool,
                       occupancy_check, noise_floor_est)
from waveforms import gen_signal, FS

out = {}

print("=== 1-2. load + DC gate ===")
fp = os.path.join(DATA, "wf10Msps_g76_rabot_f5240MHz_r1.bin")
x = load_capture(fp)
# recompute pre-DC version for comparison
raw = np.fromfile(fp, dtype=np.float32)
iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
def dc_frac(sig):
    N = 2048
    n = (sig.size // N) * N
    S = np.abs(np.fft.fft(sig[:n].reshape(-1, N), axis=1)) ** 2
    f = np.fft.fftfreq(N, d=1.0 / 10e6) / 1e6
    dc = np.argmin(np.abs(f))
    return float((S[:, dc] / S.sum(axis=1)).mean())
print(f"  DC-bin energy fraction pre-DC: {dc_frac(iq):.4f}  "
      f"post-DC: {dc_frac(x[:iq.size]):.4f}")
out["dc_pre"] = dc_frac(iq); out["dc_post"] = dc_frac(x[:iq.size])

print("=== 3-4. burst gating + occupancy per file ===")
pool, metas, prov, win_refs = build_real_wifi_pool(DATA, verbose=True)
occ_mean, occ_min, occ_max = occupancy_check(pool)
print(f"  pool: {pool.shape[0]} windows x 2048")
print(f"  occupancy in [-10,0] MHz: mean {occ_mean*100:.1f}%  "
      f"min {occ_min*100:.1f}%  max {occ_max*100:.1f}%  "
      f"GATE {'PASS' if occ_mean >= 0.90 else 'FAIL'}")
out["occupancy"] = {"mean": occ_mean, "min": occ_min, "max": occ_max}
out["pool_size"] = int(pool.shape[0])
out["per_file"] = [{k: v for k, v in m.items()} for m in metas]

print("=== 5. PAPR: real vs synthetic WiFi ===")
def papr_db(segs):
    p_inst = np.abs(segs) ** 2
    return float(10 * np.log10((p_inst.max(axis=1) / p_inst.mean(axis=1)).mean()))
rng = np.random.default_rng(0)
syn = np.stack([gen_signal(2, rng) for _ in range(200)])
pr, ps = papr_db(pool[:200]), papr_db(syn)
print(f"  real WiFi PAPR {pr:.2f} dB | synthetic WiFi PAPR {ps:.2f} dB "
      f"(same OFDM family => expect within ~2 dB)")
out["papr_real_db"] = pr; out["papr_synthetic_db"] = ps

print("=== 6. native capture SNR estimate (active vs idle floor) ===")
for m in metas[:4]:
    if m["n_kept"] > 0 and not m.get("dense", False):
        snr = 10 * np.log10(m["p_kept_median"] / max(m["floor"], 1e-30))
        print(f"  {m['file'][:44]:46s} active/floor ~ {snr:.1f} dB")
    elif m["n_kept"] > 0:
        print(f"  {m['file'][:44]:46s} dense capture (no idle floor; occupancy-gated)")
out["verdict"] = "PASS" if (occ_mean >= 0.90 and pool.shape[0] >= 90) else "CHECK"

with open(os.path.join(ROOT, "results", "real_wifi_checks.json"), "w") as f:
    json.dump({"experiment": "Wave-6 G1 real-WiFi physics gates",
               "checks": out, "provenance": prov}, f, indent=2)
print(f"\nverdict: {out['verdict']}  ->  results/real_wifi_checks.json")
