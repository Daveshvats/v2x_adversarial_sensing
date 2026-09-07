"""conformance.py — EN 302 571 clause 6.4.2-STYLE measurement-domain feasible
set (Wave 14, council round-2 34-b blocker a: "the code enforces a relative
per-bin cap pre-PA, not the e.i.r.p./RBW/mean-detector procedure").

Difference vs src/etsi_mask.py (per-FFT-bin caps at 9.77-kHz resolution):
the Table-7 unwanted-emissions limits are specified as MEAN POWER SPECTRAL
DENSITY in dBm/MHz, measured with a measurement receiver whose resolution
bandwidth (RBW) is the 1 MHz of the table's units and a mean-power (average)
detector. This module therefore constrains, for every measurement centre c
(step 0.5 MHz, overlapping 1-MHz RBW windows — a conservative dense sweep of
the continuous measurement frequency):

    mean per-bin FFT power over the bins within [c-0.5, c+0.5] MHz
        <=  psd_margin * (N * p_budget / n_in) * 10^(table7_cap_db(|c-fc|)/10)

i.e. the SAME PSR-relative budget/normalization convention as the flat and
per-bin-ETSI variants (so the three arms differ ONLY in the measurement
domain of the constraint), with the absolute layer (23 dBm/MHz in-channel,
33 dBm class EIRP) anchored in scripts/run_harm_chain.py as before.

Consequences of the domain change (why this is not cosmetic):
  * in-band: per-9.77-kHz-bin caps forbid spectral concentration inside the
    channel; an RBW/mean constraint ALLOWS it (a real transmitter may
    concentrate power in PRBs) — the conformance set is WEAKER in-band;
  * at the Table-7 knots: a 1-MHz RBW measurement centred at a steep
    transition averages across the cliff, so edge freedom is curtailed —
    the conformance set is STRONGER at the in-band/skirt boundary;
  * the net effect on the attack is an empirical question — that is what
    scripts/run_conformance.py measures.

The projection is a per-band proportional water-filling shrink, iterated
(overlapping bands couple), followed by the same total window-energy rescale
as project_etsi. Residual violation statistics are returned for disclosure.

Also provides measure_rbw(): the conformance MEASUREMENT (mean power per
1-MHz band, dB relative to the in-band reference) for post-PA checks —
"compliance is asserted by projection, verified by measurement".
"""
import numpy as np
import torch

from waveforms import FS, N_SAMPLES
from etsi_mask import table7_cap_db, TABLE7_KNOTS

FREQS_MHZ = torch.fft.fftfreq(N_SAMPLES, d=1.0 / FS) / 1e6   # rel. window ctr

BIN_MHZ = FS / 1e6 / N_SAMPLES                               # ~0.00977 MHz
RBW_MHZ = 1.0                                                # Table 7 units
STEP_MHZ = 0.5                                               # dense overlap (conservative)
N_ITERS = 8
TOL_REL = 0.01


def _band_centers(fc_mhz: float, step: float = STEP_MHZ):
    """Measurement centres covering the whole FFT window (rel. window ctr)."""
    fmin, fmax = float(FREQS_MHZ.min()), float(FREQS_MHZ.max())
    lo = np.ceil((fmin + RBW_MHZ / 2) / step) * step
    hi = np.floor((fmax - RBW_MHZ / 2) / step) * step
    return np.arange(lo, hi + 1e-9, step)


def _band_bins(centers, freqs_mhz):
    """List of index arrays: bins within [c-0.5, c+0.5] MHz for each centre."""
    f = np.asarray(freqs_mhz, dtype=float)
    out = []
    for c in centers:
        idx = np.where((f >= c - RBW_MHZ / 2) & (f <= c + RBW_MHZ / 2))[0]
        out.append(idx)
    return out


_CACHE = {}


def _bands():
    """Cached (centers, bins_per_band, torch index tensors) — the centre grid
    spans the whole FFT window and does not depend on fc."""
    if "v" not in _CACHE:
        centers = _band_centers(0.0)
        freqs = FREQS_MHZ.cpu().numpy()
        bpb = _band_bins(centers, freqs)
        _CACHE["v"] = (centers, bpb,
                       [torch.as_tensor(i, dtype=torch.long) for i in bpb])
    return _CACHE["v"]


def project_conform(delta: torch.Tensor, fc_mhz: float,
                    p_budget: torch.Tensor, psd_margin: float = 2.0,
                    return_stats: bool = False):
    """Project perturbations onto the RBW/mean-power Table-7 feasible set.

    Signature mirrors project_etsi (PSR-relative budget convention), so it
    plugs directly into waveform_pgd's project_fn hook.
    """
    B, N = delta.shape
    D = torch.fft.fft(delta, dim=1)

    f = FREQS_MHZ.to(delta.device).unsqueeze(0)              # (1, N)
    inband = (f >= fc_mhz - 5.0) & (f <= fc_mhz + 5.0)
    n_in = inband.float().sum()
    ref = psd_margin * N * p_budget / n_in                   # (B,) per-bin ref

    centers, bins_per_band, bins_t = _bands()
    # limit (dB rel) at each centre: interpolate Table 7 in |c - fc|, hold
    # the -50 dB floor beyond +-15 MHz (same convention as etsi_mask)
    off = np.abs(centers - fc_mhz)
    caps_db = table7_cap_db(off)
    caps = torch.tensor(10.0 ** (caps_db / 10.0),
                        device=delta.device, dtype=torch.float32)  # (nb,)

    P = (D.abs() ** 2)                                       # (B, N)
    factor = torch.ones_like(P)
    max_viol_rel = 0.0
    for _ in range(N_ITERS):
        max_viol = 0.0
        for bi, idx_t in enumerate(bins_t):
            if len(idx_t) == 0:
                continue
            m = P[:, idx_t].mean(dim=1)                      # (B,)
            cap = ref * caps[bi]                             # (B,)
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
    # total window-energy rescale (identical to project_etsi)
    p = (D.abs() ** 2).sum(dim=1) / N
    over = p > p_budget
    if over.any():
        g = torch.where(over, (p_budget / (p + 1e-12)).sqrt(),
                        torch.ones_like(p))
        D = D * g.unsqueeze(1)
    out = torch.fft.ifft(D, dim=1)
    if return_stats:
        return out, {"iters_run": N_ITERS, "residual_viol_rel": max_viol_rel,
                     "n_bands": len(centers), "step_mhz": STEP_MHZ,
                     "rbw_mhz": RBW_MHZ}
    return out


def measure_rbw(x: torch.Tensor, fc_mhz: float, p_ref: torch.Tensor):
    """Conformance MEASUREMENT of a waveform (post-PA check).

    Returns per-window, per-band mean power in dB relative to the in-band
    reference (p_ref: (B,) linear per-bin reference = N*p_budget/n_in or the
    equivalent absolute anchor), plus the worst excess in dBr.
    """
    B, N = x.shape
    D = torch.fft.fft(x, dim=1)
    P = (D.abs() ** 2)
    centers, bins_per_band, bins_t = _bands()
    off = np.abs(centers - fc_mhz)
    caps_db = table7_cap_db(off)
    worst_db = torch.full((B,), -1e9)
    for bi, idx_t in enumerate(bins_t):
        if len(idx_t) == 0:
            continue
        m = P[:, idx_t].mean(dim=1)
        cap = p_ref * (10.0 ** (caps_db[bi] / 10.0))
        excess_db = 10.0 * torch.log10(m / (cap + 1e-30))
        worst_db = torch.maximum(worst_db, excess_db)
    return worst_db                                         # (B,) dBr (>=0 = fail)
