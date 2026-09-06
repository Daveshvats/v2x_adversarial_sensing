#!/usr/bin/env python3
"""debug_attack.py — surgical check of the attack machinery before any sweep.
Run this after ANY change to src/ (waveforms, channels, receiver, attack_mask).

1. cconv == np.convolve equivalence (torch chain matches the numpy dataset
   chain — required for cross-verification replays).
2. gradient direction: does one PGD step (+/-) change CE as expected?
3. projection: window-energy budget exact; OOB nulled; genie has NO PSD cap.
4. band-plan occupancy: each class's FFT energy sits inside its documented
   band; PC5/11p co-channel in ITS [0,10]; WiFi in [-10,0].
5. end-to-end: PGD-20 at PSR 0 dB (budget-corrected axis) -> flips?
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import torch
import torch.nn.functional as F

from run_attack import build_eval_set
from receiver import FrontEnd, DualStreamModel
from attack_mask import cconv, project, waveform_pgd
from waveforms import NOISE_CLASS, BAND_PLAN, ATTACK_BANDS, gen_signal

OUT = os.path.join(ROOT, "results")
ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"), map_location="cpu",
                  weights_only=False)
model = DualStreamModel(); model.load_state_dict(ckpt["model"]); model.eval()
frontend = FrontEnd(); frontend.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

# --- 0. band-plan occupancy (dataset side, no model needed) ---
rng = np.random.default_rng(0)
print("band-plan occupancy (documented -> measured):")
for cls, key in [(0, "pc5"), (1, "11p"), (2, "wifi")]:
    s = gen_signal(cls, rng)
    S = np.abs(np.fft.fftshift(np.fft.fft(s))) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(len(s), 1 / 20e6)) / 1e6
    thr = S.max() * 1e-6
    occ = f[S > thr]
    print(f"  {key:5s}: doc {BAND_PLAN[key]['occupies']} -> "
          f"measured [{occ.min():.2f}, {occ.max():.2f}] MHz")

# --- 1. cconv vs np.convolve ---
x = torch.complex(torch.randn(4, 2048), torch.randn(4, 2048))
h = np.random.default_rng(1).standard_normal(27) + \
    1j * np.random.default_rng(2).standard_normal(27)
y_t = cconv(x, torch.from_numpy(h.astype(np.complex64)))
y_n = np.stack([np.convolve(x[i].numpy(), h)[13:13 + 2048]
                for i in range(4)])
err = np.abs(y_t.numpy() - y_n).max() / np.abs(y_n).max()
print(f"cconv vs np.convolve: rel err {err:.2e} "
      f"({'OK' if err < 1e-5 else 'MISMATCH'})")

# --- 2. gradient direction check ---
x_rx, y, h_a, p_clean = build_eval_set("urban", 32, seed=7)
x, yy, hh = x_rx[:32], y[:32], h_a[:32]
p_rx = p_clean[:32]                          # clean-signal window energy
pb = p_rx * 1.0                              # PSR = 0 dB (budget-corrected)

delta = torch.zeros_like(x, requires_grad=True)
r = x + cconv(delta, hh)
logits = model.forward_wave(r, frontend)
ce0 = F.cross_entropy(logits, yy).item()
g, = torch.autograd.grad(-F.cross_entropy(logits, yy), delta)
print(f"clean CE = {ce0:.4f}, |grad| = {g.abs().norm():.4e}")

for sign, name in [(+1.0, "delta + a*g (ascent on CE)"),
                   (-1.0, "delta - a*g (descent on CE)")]:
    d2 = (sign * 0.25 * pb.sqrt().unsqueeze(1) * g /
          (g.abs().norm(dim=1, keepdim=True) + 1e-12)).detach()
    with torch.no_grad():
        lg = model.forward_wave(x + cconv(d2, hh), frontend)
        print(f"  {name}: CE -> {F.cross_entropy(lg, yy).item():.4f}")

# --- 3. projection checks (energy budget semantics) ---
band = ATTACK_BANDS["cv2x_attacker"]
d = torch.complex(torch.randn(32, 2048), torch.randn(32, 2048))
d = d / d.abs().norm(dim=1, keepdim=True) * pb.sqrt().unsqueeze(1) * 3  # 3x over
dp = project(d, band, pb)
p_out = (dp.abs() ** 2).sum(dim=1)
print(f"projection: budget {pb[0]:.1f} -> output energy {p_out[0]:.3f} "
      f"(ratio {p_out[0]/pb[0]:.6f})")
D = torch.fft.fft(dp, dim=1)
f = torch.fft.fftfreq(2048, 1/20e6).to(D.device)/1e6
oob = ~((f >= band['lo']) & (f <= band['hi']))
print(f"OOB FFT power fraction (mask): "
      f"{(D.abs()[:, oob]**2).sum()/(D.abs()**2).sum():.2e}")
# genie must have NO PSD cap: a spectral line carrying the whole budget
# should keep ~all its energy under genie, but be clipped by the mask cap.
spec = torch.zeros(32, 2048, dtype=torch.complex64)
spec[:, 700] = (pb * 2048).sqrt()            # one FFT bin = full budget
spike = torch.fft.ifft(spec, dim=1)          # time-domain = pure tone
gsp = project(spike, None, pb)
print(f"genie spectral line retains "
      f"{(gsp.abs()**2).sum(dim=1)[0]/pb[0]*100:.1f}% of budget (no cap)")
msp = project(spike, band, pb)
print(f"mask spectral line retains "
      f"{(msp.abs()**2).sum(dim=1)[0]/pb[0]*100:.2f}% (cap active; "
      f"flat cap = 2/n_in ~ 0.2%)")

# --- 4. end-to-end PGD-20 at PSR 0 dB ---
out = waveform_pgd(model, frontend, x, hh, yy, pb, steps=20, band=None,
                   targeted=False)
flips = (out["preds"] != yy).sum().item()
print(f"genie 20 steps @PSR 0 dB: {flips}/32 flips, "
      f"adv CE {F.cross_entropy(out['logits'], yy).item():.4f}")
out2 = waveform_pgd(model, frontend, x, hh, yy, pb, steps=20,
                    band=band, targeted=True, target_class=NOISE_CLASS)
to_noise = (out2["preds"] == NOISE_CLASS).sum().item()
print(f"cv2x-mask targeted-to-noise 20 steps @PSR 0 dB: "
      f"{to_noise}/32 -> noise")
print("DEBUG DONE")
