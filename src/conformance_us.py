"""conformance_us.py — 47 CFR 95.3205-SHAPED measurement-domain feasible set
(Wave 14 / 35-c-2; closes council round-2 F3 with data instead of wording).

Companion to src/conformance.py (the EN 302 571 Table-7 RBW/mean-power arm,
C41). This module enforces the US C-V2X OBU unwanted-emissions limits of
47 CFR 95.3205 (89 FR 100855, Dec 13 2024; corrected 90 FR 5724, Jan 17 2025;
rows archived verbatim in data/standards/en302571_tables.json under
fcc_47cfr_95_3205) in the SAME measurement-domain formalism:

  * within +/-1 MHz of the channel edge : -16 dBm/100 kHz  -> RBW 100 kHz
  * +/-1 MHz to +/-5 MHz                : -13 dBm/MHz     -> RBW 1 MHz
  * +/-5 MHz to +/-30 MHz               : -16 dBm/MHz     -> RBW 1 MHz
  * beyond 30 MHz                       : -28 dBm/MHz     -> RBW 1 MHz

Interpretation choices (all disclosed in the output JSON and the paper):
  1. OFFSET REFERENCE: limits are keyed to the attacker's authorized 10-MHz
     CHANNEL edge (the archived eCFR rows say "offset_from_band_edge"; for an
     adjacent-channel attacker inside the ITS band the channel edge is the
     binding reference — the band-edge reading would only relax the far skirt,
     and the near skirt into the victim channel is what the attack needs).
  2. PER-MHz EQUIVALENCE OF THE 100-kHz ROW: -16 dBm/100 kHz = -6 dBm/MHz
     equivalent (one decade of bandwidth). Relative to the 23 dBm/MHz
     in-channel anchor this is -29 dB rel; -13 dBm/MHz -> -36 dB rel;
     -16 dBm/MHz -> -39 dB rel; -28 dBm/MHz -> -51 dB rel (out of window).
  3. RBW KEYED TO CENTRE: a measurement centre within 1 MHz of an edge is
     measured with 100-kHz RBW and the first-MHz limit; centres further out
     use 1-MHz RBW and their region limit (standard measurement-receiver
     convention — the limit and RBW are functions of the measurement
     frequency, not of the emission's energy distribution).
  4. STEP-FUNCTION LIMITS: unlike ETSI Table-7 (explicit knots with
     linear-in-dB interpolation), 95.3205 specifies one flat limit per offset
     region; the step boundaries at 1 and 5 MHz are implemented literally.
  5. IN-CHANNEL: unconstrained by the OOB table (0 dB rel, identical
     in-channel treatment to the ETSI/flat arms — the arms differ ONLY in
     the out-of-channel skirt, preserving the experimental control).
  6. PSR-RELATIVE BUDGET convention identical to project_etsi/project_conform
     (same normalization; absolute 23 dBm/MHz anchor lives in run_harm_chain).

Projection: per-band proportional water-filling shrink (8 iters, overlapping
bands couple), then the total window-energy rescale — identical algorithm to
project_conform, generalized to per-band (centre, RBW, cap) tuples.
"""
import numpy as np
import torch

from waveforms import FS, N_SAMPLES

FREQS_MHZ = torch.fft.fftfreq(N_SAMPLES, d=1.0 / FS) / 1e6   # rel. window ctr

STEP_NEAR_MHZ = 0.05   # 100-kHz-RBW centres: dense (RBW/2) conservative sweep
STEP_FAR_MHZ = 0.5     # 1-MHz-RBW centres: dense (RBW/2) conservative sweep
RBW_NEAR_MHZ = 0.1
RBW_FAR_MHZ = 1.0
N_ITERS = 8
TOL_REL = 0.01

# 47 CFR 95.3205 rows (offset from channel edge, dB rel 23 dBm/MHz in-ch)
# regions: [0,1] -> -29 ; (1,5] -> -36 ; (5,30] -> -39 ; (30,inf) -> -51
US_REGIONS = [
    (0.0, 1.0, -29.0),
    (1.0, 5.0, -36.0),
    (5.0, 30.0, -39.0),
    (30.0, 1e9, -51.0),
]


def us_cap_db(offset_from_edge):
    """Flat-per-region cap (dB rel in-channel) for offsets OUTSIDE the channel."""
    off = np.maximum(np.asarray(offset_from_edge, dtype=float), 0.0)
    cap = np.full_like(off, US_REGIONS[-1][2], dtype=float)
    for lo, hi, c in US_REGIONS:
        m = (off > lo) & (off <= hi)
        cap[m] = c
    # offsets == 0 (centre exactly at the edge) belong to the first-MHz region
    m = off == 0.0
    cap[m] = US_REGIONS[0][2]
    return cap


def _channel_edges(fc_mhz, ch_bw_mhz=10.0):
    return (fc_mhz - ch_bw_mhz / 2.0, fc_mhz + ch_bw_mhz / 2.0)


def _us_bands(fc_mhz):
    """(centers, rBWs, cap_dBs, bin-index arrays) for the whole FFT window.

    Centre-keyed: centres within 1 MHz OUTSIDE an edge get 100-kHz RBW + the
    first-MHz cap; all other centres get 1-MHz RBW + the cap of their offset
    region (in-channel centres: cap 0 dB rel, RBW 1 MHz).
    """
    e_lo, e_hi = _channel_edges(fc_mhz)
    fmin, fmax = float(FREQS_MHZ.min()), float(FREQS_MHZ.max())
    centers, rbws, caps = [], [], []

    # --- near-edge 100-kHz-RBW centres (step 0.05) on both sides of each edge
    for e in (e_lo, e_hi):
        lo = e - 1.0 + RBW_NEAR_MHZ / 2.0
        hi = e + 1.0 - RBW_NEAR_MHZ / 2.0
        lo = max(lo, fmin + RBW_NEAR_MHZ / 2.0)
        hi = min(hi, fmax - RBW_NEAR_MHZ / 2.0)
        if hi < lo:
            continue
        c = np.arange(np.ceil(lo / STEP_NEAR_MHZ) * STEP_NEAR_MHZ,
                      hi + 1e-9, STEP_NEAR_MHZ)
        for x in c:
            if e_lo <= x <= e_hi:      # centre inside the channel
                continue               # (in-channel, 1-MHz treatment below)
            centers.append(x)
            rbws.append(RBW_NEAR_MHZ)
            caps.append(float(us_cap_db(abs(x - e))))

    # --- far 1-MHz-RBW centres (step 0.5) over the whole window
    lo = np.ceil((fmin + RBW_FAR_MHZ / 2.0) / STEP_FAR_MHZ) * STEP_FAR_MHZ
    hi = np.floor((fmax - RBW_FAR_MHZ / 2.0) / STEP_FAR_MHZ) * STEP_FAR_MHZ
    for x in np.arange(lo, hi + 1e-9, STEP_FAR_MHZ):
        if any(abs(x - c) < 1e-9 for c in centers):
            continue                    # already covered as a near-edge centre
        centers.append(x)
        rbws.append(RBW_FAR_MHZ)
        if e_lo <= x <= e_hi:           # in-channel
            caps.append(0.0)
        else:
            off = min(abs(x - e_lo), abs(x - e_hi))
            caps.append(float(us_cap_db(off)))

    order = np.argsort(centers)
    centers = np.array(centers)[order]
    rbws = np.array(rbws)[order]
    caps = np.array(caps)[order]

    f = FREQS_MHZ.cpu().numpy()
    bins = [np.where((f >= c - r / 2.0) & (f <= c + r / 2.0))[0]
            for c, r in zip(centers, rbws)]
    return centers, rbws, caps, bins


_US_CACHE = {}


def _bands(fc_mhz):
    if fc_mhz not in _US_CACHE:
        centers, rbws, caps, bins = _us_bands(fc_mhz)
        _US_CACHE[fc_mhz] = (centers, rbws, caps,
                             [torch.as_tensor(i, dtype=torch.long)
                              for i in bins])
    return _US_CACHE[fc_mhz]


def project_conform_us(delta: torch.Tensor, fc_mhz: float,
                       p_budget: torch.Tensor, psd_margin: float = 2.0,
                       return_stats: bool = False):
    """Project perturbations onto the 95.3205-shaped RBW/mean-power set.

    Signature mirrors project_etsi / project_conform (PSR-relative budget
    convention) so it plugs into waveform_pgd's project_fn hook.
    """
    B, N = delta.shape
    D = torch.fft.fft(delta, dim=1)

    f = FREQS_MHZ.to(delta.device).unsqueeze(0)
    e_lo, e_hi = _channel_edges(fc_mhz)
    inband = (f >= e_lo) & (f <= e_hi)
    n_in = inband.float().sum()
    ref = psd_margin * N * p_budget / n_in                   # (B,) per-bin ref

    centers, rbws, caps, bins_t = _bands(fc_mhz)
    caps_t = torch.tensor(10.0 ** (caps / 10.0), device=delta.device,
                          dtype=torch.float32)               # (nb,)

    P = (D.abs() ** 2)                                       # (B, N)
    factor = torch.ones_like(P)
    max_viol_rel = 0.0
    for _ in range(N_ITERS):
        max_viol = 0.0
        for bi, idx_t in enumerate(bins_t):
            if len(idx_t) == 0:
                continue
            # mean per-bin power over the band's bins (RBW-mean detector)
            m = P[:, idx_t].mean(dim=1)                      # (B,)
            cap = ref * caps_t[bi]                           # (B,)
            viol = m - cap
            mv = float((viol.clamp(min=0) / (cap + 1e-12)).max())
            max_viol = max(max_viol, mv)
            s = torch.where(viol > 0,
                            cap / (m + 1e-12),
                            torch.ones_like(m))              # (B,)
            P[:, idx_t] = P[:, idx_t] * s.unsqueeze(1)
            factor[:, idx_t] = factor[:, idx_t] * s.unsqueeze(1)
        max_viol_rel = max(max_viol_rel, max_viol)
        if max_viol < TOL_REL:
            break

    D = D * factor.clamp(max=1.0).sqrt()
    # total window-energy rescale (identical to project_etsi/project_conform)
    p = (D.abs() ** 2).sum(dim=1) / N
    over = p > p_budget
    if over.any():
        g = torch.where(over, (p_budget / (p + 1e-12)).sqrt(),
                        torch.ones_like(p))
        D = D * g.unsqueeze(1)
    out = torch.fft.ifft(D, dim=1)
    if return_stats:
        return out, {"iters_run": N_ITERS,
                     "residual_viol_rel": max_viol_rel,
                     "n_bands": int(len(centers)),
                     "n_bands_100khz": int((rbws == RBW_NEAR_MHZ).sum()),
                     "n_bands_1mhz": int((rbws == RBW_FAR_MHZ).sum())}
    return out


def measure_us(x: torch.Tensor, fc_mhz: float, p_ref: torch.Tensor):
    """95.3205-shaped MEASUREMENT of a waveform (post-PA check).

    Returns per-window worst excess in dB relative to the cap (>= 0 = fail).
    """
    B, N = x.shape
    D = torch.fft.fft(x, dim=1)
    P = (D.abs() ** 2)
    centers, rbws, caps, bins_t = _bands(fc_mhz)
    caps_t = 10.0 ** (torch.tensor(caps, device=x.device,
                                    dtype=torch.float32) / 10.0)
    worst_db = torch.full((B,), -1e9, device=x.device)
    for bi, idx_t in enumerate(bins_t):
        if len(idx_t) == 0:
            continue
        m = P[:, idx_t].mean(dim=1)
        cap = p_ref * caps_t[bi]
        excess_db = 10.0 * torch.log10(m / (cap + 1e-30))
        worst_db = torch.maximum(worst_db, excess_db)
    return worst_db                                        # (B,) dBr
