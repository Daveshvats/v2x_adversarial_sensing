#!/usr/bin/env python3
"""feature_space_equiv.py — C14: what does the legacy feature-space threat
model silently assume in physical (transmit/receive) power?

CORRECTED VERSION (Wave-4 audit fix): the earlier draft attacked the
UNSHIFTED STFT layout while the model was trained on the FrontEnd's
fftSHIFTED layout, and multiplied that perturbation onto the shifted
spectrogram — synthesizing a waveform that realized the edit mirrored to
the wrong half of the band (PSR_eq off by ~14 dB). This version attacks the
correct (shifted) layout and synthesizes from the same layout.

Method
  1. Feature-space PGD (the LEGACY threat model): perturb the receiver's
     log-magnitude input (computed exactly as FrontEnd does: fftshifted,
     z-scored) with an Linf ball in z-units, eps in {0.1, 0.3, 1.0},
     10 steps, untargeted CE ascent. 'Intended' cond-ASR = model(mag+delta).
  2. Physical translation: a log-mag change d on bin k is expressed by the
     complex STFT-domain addition Delta_k = Z_k * (10^d - 1) (magnitude
     scaling, phase preserved, SAME shifted layout); OLA-synthesis gives the
     implied received waveform perturbation delta_rx. PSR_eq =
     10 log10( sum|delta_rx|^2 / sum|x_clean|^2 ).
     FIDELITY DISCLOSURE: the hann-window round trip realizes only ~0.48-0.5
     of the intended per-bin log-change (dense-pattern slope; single-bin
     ~0.5, analytic w^2/OLA prediction). We therefore also measure the
     'realized' attack: feed x_rx + delta_rx through the true FrontEnd and
     report that cond-ASR (what a waveform of power PSR_eq arriving directly
     at the receiver would achieve). The intended-edit ASR overstates what
     that exact waveform realizes.
  3. Equal-power comparison: waveform-domain PGD attacks (genie and
     mask-compliant, through the attacker channel h_a) at PSR = round(median
     PSR_eq) on the same samples.

Received-side power note: PSR_eq is the power of the implied RECEIVED
perturbation; a transmitter must supply at least this after its channel
(unit-power in expectation).
"""
import sys, os, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
import torch.nn.functional as F

from run_attack import build_eval_set
from waveforms import ATTACK_BANDS
from receiver import FrontEnd, DualStreamModel
from attack_mask import waveform_pgd, cond_asr

OUT = os.path.join(ROOT, "results")
EPS_LIST = [0.1, 0.3, 1.0]
STEPS = 10

torch.manual_seed(7)
np.random.seed(7)


def stft_cf(x, window):
    """center=False, two-sided STFT exactly as FrontEnd computes it."""
    return torch.stft(x, n_fft=256, hop_length=128, window=window,
                      return_complex=True, center=False, onesided=False)


def istft_manual(D, window, length=2048, hop=128):
    """OLA synthesis with envelope guard (hann zero-envelope endpoints)."""
    B, N, T = D.shape
    frames = torch.fft.ifft(D, dim=1)
    out = torch.zeros(B, length, dtype=D.dtype)
    env = torch.zeros(length, dtype=window.dtype)
    for t in range(T):
        s = t * hop
        e = s + N
        if e > length:
            break
        out[:, s:e] += frames[:, :, t] * window
        env[s:e] += window * window
    guard = (env > 0.01)
    env_safe = torch.where(guard, env, torch.ones_like(env))
    out = torch.complex(out.real / env_safe, out.imag / env_safe)
    out = out * guard.to(out.dtype)
    return out


ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                  map_location="cpu", weights_only=False)
model = DualStreamModel(); model.load_state_dict(ckpt["model"]); model.eval()
frontend = FrontEnd(); frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])
mag_std = float(frontend.mag_std)

x_rx, y, h_a, p_clean = build_eval_set("urban", 100, seed=7)

# ---------- clean predictions + canonical (SHIFTED-layout) inputs ----------
with torch.no_grad():
    clean_preds, ifr_fixed, mags, Zs_all = [], [], [], []
    for i in range(0, x_rx.size(0), 256):
        xb = x_rx[i:i + 256]
        clean_preds.append(model.forward_wave(xb, frontend).argmax(1))
        _, ifr_b = frontend(xb)
        ifr_fixed.append(ifr_b)
        Z = torch.fft.fftshift(stft_cf(xb, frontend.window), dim=1)
        Zs_all.append(Z)
        mag = torch.log10(Z.abs() + 1e-10)
        mags.append(((mag - frontend.mag_mean) / frontend.mag_std).unsqueeze(1))
    clean_preds = torch.cat(clean_preds)
    ifr_fixed = torch.cat(ifr_fixed)
    mag = torch.cat(mags)                     # (B,1,256,15) SHIFTED layout
    Zs = torch.cat(Zs_all)                    # (B,256,15)   SHIFTED layout
print(f"clean acc: {(clean_preds == y).float().mean()*100:.1f}%")

# istft invertibility (interior)
with torch.no_grad():
    Z0 = stft_cf(x_rx[:8], frontend.window)
    x0r = istft_manual(Z0.clone(), frontend.window)
    w = frontend.window
    env0 = torch.zeros(2048)
    for t in range(15):
        env0[t * 128:t * 128 + 256] += w * w
    interior = env0 > 0.01
    inv_err = (x0r[:, interior] - x_rx[:8, interior]).abs().max().item() \
        / x_rx[:8].abs().max().item()
print(f"istft interior rel err: {inv_err:.2e}")

results = {"experiment": "feature-space threat-model physical equivalence "
                         "(C14, corrected shifted-layout translation)",
           "invertibility_interior_rel_err": inv_err,
           "method": "Linf (z-units) PGD on the canonical (fftshifted) mag "
                     "input -> complex STFT-domain magnitude scaling on the "
                     "same layout -> OLA synthesis -> implied received power; "
                     "realized (physically-arrived) attack vs intended edit; "
                     "equal-power transmitted-attack comparison",
           "fidelity_note": "hann window round trip realizes ~0.5 of the "
                            "intended per-bin log-change; realized-ASR is "
                            "what the implied waveform actually achieves",
           "per_epsilon": {}}

for eps in EPS_LIST:
    t0 = time.time()
    # ---------- 1. feature-space PGD on the canonical mag input ----------
    delta = torch.zeros_like(mag, requires_grad=True)
    for _ in range(STEPS):
        logits = model(mag + delta, ifr_fixed)
        loss = -F.cross_entropy(logits, y)   # descend this = ascend CE
        g, = torch.autograd.grad(loss, delta)
        gn = g / (g.norm(dim=(2, 3), keepdim=True) + 1e-12)
        with torch.no_grad():
            delta = delta - 0.25 * eps * gn
            delta = torch.clamp(delta, -eps, eps)
        delta = delta.detach().requires_grad_(True)
    with torch.no_grad():
        intended_preds = model(mag + delta, ifr_fixed).argmax(1)
    c_asr_intended, n_elig = cond_asr(clean_preds, intended_preds, y)

    # ---------- 2. physical translation (SAME shifted layout) ----------
    d_log10 = delta.detach().squeeze(1) * mag_std      # (B,256,15)
    scale = (10.0 ** d_log10) - 1.0
    Delta = torch.fft.ifftshift(Zs * scale, dim=1)     # back to FFT grid
    delta_rx = istft_manual(Delta, frontend.window)
    e_delta = (delta_rx.abs() ** 2).sum(dim=1)
    psr_eq = 10 * torch.log10(e_delta / p_clean + 1e-20)
    med = float(psr_eq.median())
    q1, q3 = [float(v) for v in
              torch.quantile(psr_eq, torch.tensor([0.25, 0.75]))]

    # ---------- 2b. REALIZED attack: the waveform through the true front-end
    with torch.no_grad():
        realized_preds = []
        for i in range(0, x_rx.size(0), 256):
            lg = model.forward_wave(x_rx[i:i + 256] + delta_rx[i:i + 256],
                                    frontend)
            realized_preds.append(lg.argmax(1))
        realized_preds = torch.cat(realized_preds)
    c_asr_realized, _ = cond_asr(clean_preds, realized_preds, y)

    # ---------- 3. equal-power TRANSMITTED attacks (through h_a) ----------
    psr_pow = int(np.round(med))
    p_budget = p_clean * (10 ** (psr_pow / 10.0))
    preds = []
    for i in range(0, x_rx.size(0), 300):
        out = waveform_pgd(model, frontend, x_rx[i:i + 300], h_a[i:i + 300],
                           y[i:i + 300], p_budget[i:i + 300], steps=STEPS,
                           band=None, targeted=False)
        preds.append(out["preds"])
    c_asr_genie, _ = cond_asr(clean_preds, torch.cat(preds), y)
    preds = []
    for i in range(0, x_rx.size(0), 300):
        out = waveform_pgd(model, frontend, x_rx[i:i + 300], h_a[i:i + 300],
                           y[i:i + 300], p_budget[i:i + 300], steps=STEPS,
                           band=ATTACK_BANDS["cv2x_attacker"],
                           targeted=False)
        preds.append(out["preds"])
    c_asr_mask, _ = cond_asr(clean_preds, torch.cat(preds), y)

    results["per_epsilon"][f"eps={eps}"] = {
        "cond_asr_feature_space_intended": round(100 * c_asr_intended, 2),
        "cond_asr_feature_space_realized": round(100 * c_asr_realized, 2),
        "median_psr_eq_db": round(med, 2),
        "psr_eq_iqr_db": [round(q1, 2), round(q3, 2)],
        "equal_power_psr_db": psr_pow,
        "cond_asr_waveform_genie": round(100 * c_asr_genie, 2),
        "cond_asr_waveform_mask": round(100 * c_asr_mask, 2),
        "n_eligible": n_elig,
    }
    print(f"eps={eps}: intended {100*c_asr_intended:5.1f}%  realized "
          f"{100*c_asr_realized:5.1f}%  PSR_eq {med:+6.2f} dB "
          f"[{q1:+.1f},{q3:+.1f}]  @equal power: genie "
          f"{100*c_asr_genie:5.1f}%  mask {100*c_asr_mask:5.1f}%  "
          f"({time.time()-t0:.0f}s)", flush=True)

with open(os.path.join(OUT, "feature_space_equiv.json"), "w") as f:
    json.dump(results, f, indent=2)
print("wrote results/feature_space_equiv.json")
print("DONE")
