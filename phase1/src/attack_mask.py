"""attack_mask.py — THE CORE INNOVATION: emission-mask-constrained, waveform-domain
adversarial attacks on DL spectrum sensing ("the Compliant Attacker").

Threat model
  * The adversary is a SPECTRUM-RULE-ABIDING device: it may only transmit within its
    regulatory allocation (e.g., the ITS-band 10 MHz channel [0, +10] MHz relative
    to the 5895 MHz boundary) with an idealized emission mask: out-of-band energy
    nulled, in-band PSD capped.
  * The adversary knows the victim model and the attacker->victim channel. WORST-CASE
    DISCLOSURE: the optimization below also uses the exact received realization
    (victim signal + receiver noise). A real attacker cannot know the receiver's
    noise draw, so every ASR here is an upper bound (favorable direction for the
    attack, pessimistic for the defense). A CSI/noise-mismatch variant is provided
    for realistic evaluation (see run scripts).
  * Budget is PHYSICAL and stated as WINDOW ENERGY (sum of squared samples) of the
    adversarial transmit waveform: PSR (dB) = 10 log10(E_attack / E_clean_rx),
    where E_clean_rx is the window energy of the clean received signal. Window
    lengths are equal, so PSR equals the mean-power ratio — a '0 dB' attacker
    transmits at the same power the victim's signal is received with.
  * Compliance is defined AT THE ATTACKER'S TRANSMIT PORT (projection happens before
    the channel). The finite 102.4 us rectangular sensing window then spreads the
    received adversarial component to about -33.5 dB out-of-band fraction — that is
    a receiver/window artifact, not a transmit violation. Practical RF impairments
    (PA ACLR, phase noise, LO leakage) are not modeled.

Attack chain (fully differentiable in torch):
    delta -> complex FIR channel h_a -> + victim signal + noise ->
    torch.stft front-end -> CNN -> loss
    PGD-K on delta with per-step projection onto the feasible set:
      1. FFT(delta), zero out-of-mask bins,
      2. cap per-bin PSD at the mask level,
      3. rescale so total power <= P_budget,
      4. iFFT back.

Baselines and metrics
  * "genie" attacker: same power budget, NO spectral restriction (any waveform in the
    full 20 MHz window) — the classical unrealistic upper bound.
  * Price of Compliance (PoC, dB): extra power a compliant attacker needs to reach the
    same conditional ASR as the genie attacker (interpolated at a target ASR).
  * MEAP (dB): Minimum Effective Attack Power — PSR at which conditional ASR crosses
    a threshold (default 20%).
  * conditional ASR: attack success measured ONLY over samples the model classified
    correctly before the attack (see Phase-0 AUDIT.md for why raw ASR misleads).
"""

import numpy as np
import torch
import torch.nn.functional as F

from waveforms import FS, N_SAMPLES, ATTACK_BANDS, NOISE_CLASS
from receiver import FrontEnd

FREQS = torch.fft.fftfreq(N_SAMPLES, d=1.0 / FS) / 1e6   # MHz, (N,)


# ---------------------------------------------------------------------------
# differentiable complex FIR convolution (channel) in torch
# ---------------------------------------------------------------------------
def cconv(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """x: (B, L) complex; h: (Lh,) complex OR (B, Lh) complex (per-sample channel)
    -> 'same' centered convolution (B, L). Uses grouped conv for batched kernels.

    F.conv1d computes cross-CORRELATION; we flip the taps so that cconv(x, h)
    matches np.convolve(x, h)[(Lh-1)//2 : (Lh-1)//2+L] exactly (verified in
    scripts/debug_attack.py) — so the torch attack chain and the numpy dataset
    chain implement the same physical operation and cross-verification replays
    produce identical waveforms."""
    if h.dim() == 1:
        h = h.unsqueeze(0).expand(x.size(0), -1)          # shared kernel
    h = torch.flip(h, dims=[-1])                          # correlation -> convolution
    B, Lh = h.shape
    # grouped conv: treat batch as channels -> each sample gets its own kernel
    xr = x.real.unsqueeze(0)                              # (1, B, L)
    xi = x.imag.unsqueeze(0)
    o1 = F.conv1d(xr, h.real.unsqueeze(1), padding=Lh // 2, groups=B)
    o2 = F.conv1d(xi, h.imag.unsqueeze(1), padding=Lh // 2, groups=B)
    o3 = F.conv1d(xr, h.imag.unsqueeze(1), padding=Lh // 2, groups=B)
    o4 = F.conv1d(xi, h.real.unsqueeze(1), padding=Lh // 2, groups=B)
    out = torch.complex(o1 - o2, o3 + o4).squeeze(0)     # (B, L')
    return out[:, : x.size(1)]


# ---------------------------------------------------------------------------
# feasible-set projection
# ---------------------------------------------------------------------------
def project(delta: torch.Tensor, band: dict, p_budget: torch.Tensor,
            psd_margin: float = 2.0) -> torch.Tensor:
    """Project waveform-domain perturbations onto the feasible set.

    Parseval (torch FFT, norm='backward'): sum_k |D_k|^2 = N * sum_n |delta_n|^2,
    so per-bin FFT-domain power caps must be scaled by N relative to the
    time-domain (window-energy) budget.

    band: {"lo", "hi"} in MHz. band=None => GENIE attacker: the ONLY constraint
          is the total window-energy budget — no spectral restriction and NO
          per-bin PSD cap (any waveform in the full 20 MHz window is allowed).
    p_budget: (B,) per-sample window-energy budget (linear units).
    psd_margin: in-band per-bin PSD cap = psd_margin * (uniform sharing of the
          budget across in-band bins), applied ONLY to the constrained attacker.
          NOTE: a flat cap is an IDEALIZATION that is stricter than real
          regulations (which cap channel power and OOB emission, not in-band
          flatness — a real C-V2X device may concentrate its power in a few
          PRBs). The binding physics is the out-of-band null; the cap models
          an adversarial device that must also look like a normal wideband
          transmission in-band. Sensitivity to this idealization is swept in
          the experiments (psd_margin in {2, 10}); disclose in Limitations.
    """
    B, N = delta.shape
    D = torch.fft.fft(delta, dim=1)                        # (B, N) complex
    if band is None:
        # genie: power-only constraint
        p = (D.abs() ** 2).sum(dim=1) / N                  # = sum|delta|^2
        over = p > p_budget
        if over.any():
            g = torch.where(over, (p_budget / (p + 1e-12)).sqrt(),
                            torch.ones_like(p))
            D = D * g.unsqueeze(1)
        return torch.fft.ifft(D, dim=1)

    f = FREQS.to(delta.device).unsqueeze(0)                # (1, N) MHz
    inband = (f >= band["lo"]) & (f <= band["hi"])
    D = D * inband                                         # OOB nulled
    n_in = inband.float().sum()
    # per-bin FFT-domain power cap: psd_margin * N * P_a / n_in
    cap = psd_margin * N * p_budget / n_in                 # (B,)
    mag = D.abs() + 1e-12
    scale = torch.minimum(torch.ones_like(mag),
                          (cap.sqrt().unsqueeze(1) / mag))
    D = D * scale
    # total window energy: sum|D|^2 / N  <=  p_budget
    p = (D.abs() ** 2).sum(dim=1) / N                      # (B,)
    over = p > p_budget
    if over.any():
        g = torch.where(over, (p_budget / (p + 1e-12)).sqrt(),
                        torch.ones_like(p))
        D = D * g.unsqueeze(1)
    return torch.fft.ifft(D, dim=1)


# ---------------------------------------------------------------------------
# the attack
# ---------------------------------------------------------------------------
def waveform_pgd(model, frontend, x_rx, h_a, y, p_budget, steps=10,
                 band=None, targeted=False, target_class=NOISE_CLASS,
                 alpha_frac=0.25, psd_margin=2.0, return_delta=False):
    """PGD-K attack on the waveform delta.

    model/frontend : victim (differentiable chain)
    x_rx           : (B, L) complex — victim-received clean signal incl. noise
    h_a            : (Lh,) complex — attacker->victim channel (fixed during opt)
    y              : (B,) true labels
    p_budget       : (B,) linear power budgets per sample
    band           : {"lo","hi"} MHz for mask compliance; None => genie
    targeted       : minimize CE to target_class (e.g., cloak as Noise)
    """
    B = x_rx.size(0)
    delta = torch.zeros_like(x_rx, requires_grad=True)
    h = torch.as_tensor(h_a, dtype=x_rx.dtype)

    for _ in range(steps):
        r = x_rx + cconv(delta, h)
        logits = model.forward_wave(r, frontend)
        if targeted:
            loss = F.cross_entropy(logits,
                                   torch.full_like(y, target_class))
        else:
            loss = -F.cross_entropy(logits, y)
        g, = torch.autograd.grad(loss, delta)
        gn = g / (g.abs().norm(dim=1, keepdim=True) + 1e-12)
        step = alpha_frac * p_budget.sqrt().unsqueeze(1) * gn
        with torch.no_grad():
            # torch complex autograd convention: z - a*g DESCENDS the loss;
            # verified empirically in scripts/debug_attack.py
            delta = delta - step
            delta = project(delta, band, p_budget, psd_margin=psd_margin)
        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        r_adv = x_rx + cconv(delta, h)
        logits = model.forward_wave(r_adv, frontend)
        preds = logits.argmax(1)
    out = {"preds": preds, "logits": logits}
    if return_delta:
        out["delta"] = delta.detach()
    return out


# ---------------------------------------------------------------------------
# evaluation + physical metrics
# ---------------------------------------------------------------------------
def cond_asr(clean_preds, adv_preds, y, targeted=False, target=NOISE_CLASS):
    elig = clean_preds == y
    n = int(elig.sum())
    if n == 0:
        return float("nan"), 0
    if targeted:
        ok = elig & (adv_preds == target)
    else:
        ok = elig & (adv_preds != y)
    return float(ok.sum()) / n, n


def meap_curve(psr_db_list, asr_list, threshold=20.0):
    """Minimum Effective Attack Power: linear interpolation of PSR where ASR
    first crosses `threshold` from below. Threshold is in the SAME UNITS as
    asr_list (percent).

    Returns (meap_db, censor):
      censor = None : MEAP interpolated inside the grid (exact).
      censor = "le" : ASR already >= threshold at the LOWEST grid PSR — the
                      true MEAP is <= meap_db (meap_db is the grid floor;
                      extend the grid downward to resolve).
      censor = "ge" : ASR never reaches threshold inside the grid — true MEAP
                      is > the highest grid PSR (meap_db is None).
    """
    psr = np.asarray(psr_db_list, dtype=float)
    a = np.asarray(asr_list, dtype=float)
    if len(a) == 0:
        return None, "ge"
    if a[0] >= threshold:
        return float(psr[0]), "le"
    for i in range(len(a) - 1):
        if a[i] < threshold <= a[i + 1]:
            t = (threshold - a[i]) / (a[i + 1] - a[i] + 1e-12)
            return float(psr[i] + t * (psr[i + 1] - psr[i])), None
    return None, "ge"


def price_of_compliance(psr_db_genie, asr_genie, psr_db_mask, asr_mask,
                        threshold=20.0):
    """Returns (poc_db, censor_notes): poc = MEAP_mask - MEAP_genie.
    censor_notes is None when both MEAPs are exact, else a dict describing
    which side was censored and in which direction the true PoC is biased."""
    m_g, c_g = meap_curve(psr_db_genie, asr_genie, threshold)
    m_m, c_m = meap_curve(psr_db_mask, asr_mask, threshold)
    if m_g is None or m_m is None:
        return None, {"genie": c_g, "mask": c_m}
    notes = {k: v for k, v in (("genie", c_g), ("mask", c_m)) if v}
    if c_g == "le":
        notes["poc_bias"] = "lower bound (genie MEAP censored at grid floor)"
    if c_m == "le":
        notes["poc_bias"] = "upper bound (mask MEAP censored at grid floor)"
    return m_m - m_g, (notes if notes else None)
