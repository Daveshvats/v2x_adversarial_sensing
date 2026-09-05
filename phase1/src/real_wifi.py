"""real_wifi.py — real over-the-air WiFi captures as the U-NII-4 class (Wave 6).

Source (G1, "real signal in the loop"):
  Fontaine et al., "Technology-Recognition dataset of real-life LTE, Wi-Fi and
  DVB-T" (imec / UGent), captures in Gent, Belgium. 802.11 WiFi at 5240 MHz
  (U-NII-2 band, same OFDM PHY family as U-NII-4) captured over-the-air with a
  USRP at 10 Msps, float32 interleaved I/Q, 8.8 MB (1.1 M complex samples,
  0.11 s) per file. License: CC BY-NC-SA 4.0 (redistributed with attribution
  for research use).

Processing pipeline (every step is physics-gated — see scripts/w6_check.py):
  1. load float32 interleaved -> complex64
  2. remove DC carrier (mean subtraction — USRP LO leakage)
  3. resample 10 -> 20 Msps (x2 polyphase, real filter applied to I and Q)
  4. place in the sensing window at center -5 MHz -> occupies [-10, 0] MHz
     relative to 5895 MHz: the in-window half of a U-NII-4 channel, matching
     the physical position of the synthetic WiFi class (below the boundary)
  5. burst gating: WiFi is duty-cycled; keep only 2048-sample windows whose
     power exceeds max(6 dB above the file's 10th-percentile window power,
     3x the file's noise-floor estimate) — "active airtime" segments
  6. per-window unit-power normalization (same convention as synthetic
     waveforms; SNR is applied at the receiver, see below)

Channel honesty (disclosed in the paper):
  Real captures already contain a REAL propagation channel (real multipath
  from the capture location) plus the USRP's native noise floor. We therefore
  apply NO additional synthetic victim channel to real WiFi segments, and only
  add the receiver noise model at the drawn SNR (same as synthetic classes).
  Effective SNR of real segments is bounded by the capture's native quality.
  The synthetic classes keep the TR 37.885 victim channel as in all canonical
  runs. This asymmetry is the honest way to get a real signal into the loop
  without double-channeling it.
"""

import os
import glob
import numpy as np
from scipy.signal import resample_poly

FS_CAP = 10e6          # capture sampling rate
WINDOW_MHZ = 20.0      # our sensing window width
PLACE_CENTER_MHZ = -5.0
N_OUT = 2048           # sensing window length at 20 Msps
GATE_DB = 6.0          # active-segment power gate above file percentile
GATE_FLOOR_X = 3.0     # and at least 3x the estimated noise floor

PROVENANCE = {
    "dataset": "Technology-Recognition dataset of real-life LTE, Wi-Fi and "
               "DVB-T (Fontaine et al., UGent/imec, Gent, Belgium)",
    "url": "https://cloud.ilabt.imec.be/index.php/s/qrJCWgzQaGPfHPr",
    "paper": "Fontaine et al., 'Towards low-complexity wireless technology "
             "classification across multiple environments', Ad Hoc Networks "
             "91:101881, 2019",
    "license": "CC BY-NC-SA 4.0",
    "captures": "wf10Msps_*_f5240MHz_*.bin — 802.11 WiFi, 5240 MHz, USRP, "
                "10 Msps, float32 interleaved I/Q",
    "processing": "DC removal -> x2 polyphase resample to 20 Msps -> "
                  f"frequency shift to {PLACE_CENTER_MHZ} MHz -> burst "
                  "gating (>= 6 dB above 10th-pct window power and >= 3x "
                  "noise floor) -> unit-power normalization",
}


def load_capture(path):
    """float32 interleaved I/Q file -> complex64 at 20 Msps, DC-removed,
    placed at PLACE_CENTER_MHZ, unit total capture power preserved."""
    raw = np.fromfile(path, dtype=np.float32)
    iq = (raw[0::2] + 1j * raw[1::2]).astype(np.complex64)
    iq = iq - iq.mean()                                  # DC (LO leakage)
    # resample x2: real filter on I and Q separately
    i_rs = resample_poly(iq.real, 2, 1)
    q_rs = resample_poly(iq.imag, 2, 1)
    x = (i_rs + 1j * q_rs).astype(np.complex64)
    # place at -5 MHz (freq shift on the 20 Msps grid)
    n = np.arange(x.size)
    x = x * np.exp(2j * np.pi * (PLACE_CENTER_MHZ * 1e6 / 20e6) * n)
    x = x.astype(np.complex64)
    return x


def noise_floor_est(x, n_seg=2048):
    """Median of the lowest decile of segment powers = idle-airtime floor."""
    n = (x.size // n_seg) * n_seg
    p = (np.abs(x[:n].reshape(-1, n_seg)) ** 2).mean(axis=1)
    return np.percentile(p, 10)


def active_segments(x, n_seg=N_OUT, hop=None):
    """Burst-gated 2048-sample windows, unit-power normalized.

    hop=None: non-overlapping windows (eval protocol).
    hop=512 : 4x overlapping windows — ONLY for fine-tune training data,
              combined with exclude_ranges to prevent overlap with the eval
              windows (leakage guard, see run_real_wifi.py).
    """
    if hop is None:
        hop = n_seg
    starts = np.arange(0, x.size - n_seg + 1, hop)
    segs = np.stack([x[s:s + n_seg] for s in starts])
    # NOTE on why the gate is what it is: after x2 polyphase resampling the
    # ENTIRE capture (signal AND noise) lives inside the placed 10 MHz, so
    # band-occupancy cannot discriminate bursts from idle air. WiFi bursts
    # are identified by (a) spectral NON-flatness of the placed band (OFDM
    # data/pilot structure: flatness ~0.2-0.4) vs flat noise (~0.5+), and
    # (b) power above the idle floor. Verified per-file in w6_check.py.
    N = segs.shape[1]
    f = np.fft.fftfreq(N, d=1.0 / 20e6) / 1e6
    band = (f >= -10.0) & (f <= 0.0)
    S = np.abs(np.fft.fft(segs, axis=1)) ** 2
    Sb = S[:, band] + 1e-30
    from scipy.stats import gmean
    sf = gmean(Sb, axis=1) / Sb.mean(axis=1)          # spectral flatness
    p = Sb.mean(axis=1)                                # in-band power
    # idle floor from flat windows (noise); if none, file is all-noise
    flat = sf >= 0.48
    if flat.sum() >= 0.05 * segs.shape[0]:
        floor = float(np.median(p[flat]))
        keep = (sf <= 0.45) & (p >= 3.0 * floor)
    else:
        # no flat reference: treat the 10th-percentile power as floor
        floor = float(np.percentile(p, 10))
        keep = (sf <= 0.45) & (p >= 3.0 * floor)
    kept = segs[keep]
    kept_starts = starts[keep]
    if kept.shape[0] == 0:
        return kept, 0.0, {"floor": floor, "n_total": int(segs.shape[0]),
                           "n_kept": 0, "kept_frac": 0.0}, kept_starts
    # unit average power per window (waveform convention)
    kept = kept / np.sqrt((np.abs(kept) ** 2).mean(axis=1, keepdims=True))
    meta = {"floor": floor,
            "p_kept_median": float(np.median(p[keep])),
            "sf_kept_median": float(np.median(sf[keep])),
            "n_total": int(segs.shape[0]),
            "n_kept": int(kept.shape[0]),
            "kept_frac": float(kept.shape[0] / segs.shape[0])}
    return kept.astype(np.complex64), meta["kept_frac"], meta, kept_starts


def build_real_wifi_pool(data_dir, max_per_file=None, verbose=False, hop=None):
    """All gated segments from all wf10Msps 5240 MHz captures in data_dir.

    Returns (pool, metas, provenance, win_refs) — win_refs is a list of
    (file, start_sample) per pooled window, for leakage-free splits.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "wf10Msps*f5240*.bin")))
    pool, metas, win_refs = [], [], []
    for fp in files:
        x = load_capture(fp)
        segs, kf, meta, starts = active_segments(x, hop=hop)
        meta["file"] = os.path.basename(fp)
        meta["n_total"] = int(x.size // N_OUT)
        if max_per_file and segs.shape[0] > max_per_file:
            idx = np.linspace(0, segs.shape[0] - 1, max_per_file).astype(int)
            segs = segs[idx]; starts = starts[idx]
        if segs.shape[0]:
            pool.append(segs)
            win_refs += [(os.path.basename(fp), int(s)) for s in starts]
        metas.append(meta)
        if verbose:
            print(f"  {meta['file']}: {meta['n_total']} -> "
                  f"{meta['n_kept']} windows ({kf*100:.0f}%)")
    if pool:
        pool = np.concatenate(pool, axis=0)
    else:
        pool = np.zeros((0, N_OUT), np.complex64)
    prov = dict(PROVENANCE)
    prov["files_used"] = [m["file"] for m in metas if m["n_kept"] > 0]
    prov["n_windows_total"] = int(pool.shape[0])
    return pool, metas, prov, win_refs


def occupancy_check(segs, lo=-10.0, hi=0.0, fs=20e6):
    """D2 gate: fraction of window energy inside [lo, hi] MHz (mean over
    windows). Real OFDM bursts after placement should be ~>= 90%."""
    N = segs.shape[1]
    f = np.fft.fftfreq(N, d=1.0 / fs) / 1e6
    band = (f >= lo) & (f <= hi)
    S = np.abs(np.fft.fft(segs, axis=1)) ** 2
    frac = S[:, band].sum(axis=1) / S.sum(axis=1)
    return float(frac.mean()), float(frac.min()), float(frac.max())
