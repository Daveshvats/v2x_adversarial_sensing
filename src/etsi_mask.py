"""etsi_mask.py — ETSI EN 302 571 V1.2.1 (2013-09) Table-7-shaped feasible-set
projection (Wave 12 part 2; council P0 item: "flat-cap mask not ETSI EN 302 571
template").

The original threat model used an IDEALIZED in-band mask: hard out-of-band
null + flat in-band PSD cap (2x uniform sharing). This module replaces the
SHAPE with the REAL European ITS unwanted-emissions template (Table 7,
clause 6.4.2), as a per-FFT-bin cap that decays with offset from the
attacker's carrier:

    offset |f - fc| (MHz):  0     4.5    5.0    5.5    10     15
    cap rel. in-ch (dB):     0      0    -26    -32    -40    -50

with linear-in-dB interpolation between knots and the -50 dB value held
beyond +-15 MHz (conservative: the standard's table stops there for the
10 MHz channel template). All values verbatim from
data/standards/en302571_tables.json (extracted from the official ETSI PDF,
sha256 667a939...; the older 'en302571_v211.pdf' in the W1-C corpus was a
WAF block page, not the standard).

Normalization convention (matches src.attack_mask.project):
  * The in-channel reference cap is psd_margin * (uniform sharing of the
    window-energy budget across the 10 MHz channel), exactly like the flat
    cap — so the ETSI variant differs from the flat variant ONLY in the
    SHAPE of the out-of-channel skirt (it ALLOWS skirted energy the flat
    mask nulls) while the total window-energy budget p_budget is unchanged.
  * Parseval (torch FFT, norm='backward'): per-bin FFT-domain caps are
    scaled by N relative to time-domain energy.
  * Absolute calibration is deliberately NOT enforced here (PSR-relative
    budgets); Table 3's 33 dBm / 23 dBm/MHz anchors the absolute layer in
    scripts/run_harm_chain.py. Disclosed in the paper's Limitations.
"""
import numpy as np
import torch

from waveforms import FS, N_SAMPLES

FREQS_MHZ = torch.fft.fftfreq(N_SAMPLES, d=1.0 / FS) / 1e6   # rel. window ctr

# ETSI EN 302 571 V1.2.1 Table 7 (10 MHz channel): knots (offset MHz, dB rel)
TABLE7_KNOTS = np.array([
    [0.0,   0.0],
    [4.5,   0.0],
    [5.0,  -26.0],
    [5.5,  -32.0],
    [10.0, -40.0],
    [15.0, -50.0],
])


def table7_cap_db(offset_mhz: np.ndarray) -> np.ndarray:
    """Piecewise-linear-in-dB cap (relative to in-channel PSD) for |f-fc|."""
    off = np.maximum(np.asarray(offset_mhz, dtype=float), 0.0)
    cap = np.interp(off, TABLE7_KNOTS[:, 0], TABLE7_KNOTS[:, 1])
    return np.where(off > TABLE7_KNOTS[-1, 0], TABLE7_KNOTS[-1, 1], cap)


def project_etsi(delta: torch.Tensor, fc_mhz: float, p_budget: torch.Tensor,
                 psd_margin: float = 2.0) -> torch.Tensor:
    """Project waveform-domain perturbations onto the ETSI-Table-7-shaped set.

    delta    : (B, N) complex time-domain perturbation
    fc_mhz   : attacker carrier centre (MHz, window-relative; +5 for the
               [0,+10] MHz ITS allocation)
    p_budget : (B,) per-sample window-energy budget (linear)
    psd_margin : multiplier on the in-channel reference cap (same role as the
               flat-cap margin; 2.0 = canonical)

    Shape: per-bin cap = psd_margin * (N * p_budget / n_in) * 10^(cap_db/10),
    with n_in = bins in [fc-5, fc+5]. Total energy then rescaled to p_budget
    (the skirt makes the cap non-uniform, so the total-power step binds).
    """
    B, N = delta.shape
    D = torch.fft.fft(delta, dim=1)

    f = FREQS_MHZ.to(delta.device).unsqueeze(0)              # (1, N)
    inband = (f >= fc_mhz - 5.0) & (f <= fc_mhz + 5.0)
    n_in = inband.float().sum()
    ref = psd_margin * N * p_budget / n_in                   # (B,) per-bin

    off = (f - fc_mhz).abs().cpu().numpy()                   # (1, N)
    shape = torch.from_numpy(
        10.0 ** (table7_cap_db(off) / 10.0)).to(
            device=delta.device, dtype=torch.float32)        # match delta
    if shape.dim() == 1:
        shape = shape.unsqueeze(0)                           # ensure (1, N)
    cap = ref.unsqueeze(1) * shape                           # (B, N)

    mag = D.abs() + 1e-12
    scale = torch.minimum(torch.ones_like(mag), cap.sqrt() / mag)
    D = D * scale

    p = (D.abs() ** 2).sum(dim=1) / N                        # = sum|d|^2
    over = p > p_budget
    if over.any():
        g = torch.where(over, (p_budget / (p + 1e-12)).sqrt(),
                        torch.ones_like(p))
        D = D * g.unsqueeze(1)
    return torch.fft.ifft(D, dim=1)
