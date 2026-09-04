"""waveforms.py — Post-FCC 5.9 GHz ITS-band coexistence waveform generators.

Sensing window (complex baseband, FS = 20 MHz, 2048 samples = 102.4 us):
  The window STRADDLES the 5895 MHz boundary created by FCC 20-164 (2020 First
  R&O; reaffirmed 2024 Second R&O): 5850-5895 MHz is unlicensed (U-NII-4),
  5895-5925 MHz is dedicated ITS (C-V2X). Window center = 5895 MHz, i.e.
  window = [5885, 5905] MHz; all frequencies below are RELATIVE to 5895 MHz.

Band plan (relative MHz):
  Class 0  C-V2X PC5   : SC-FDMA (DFT-s-OFDM), 450 subcarriers x 20 kHz
                         (9 MHz), centered +5 -> occupies [0.5, 9.5]
                         (the 5895-5905 MHz ITS channel).
  Class 1  802.11p     : OFDM, 52 subcarriers x 156.25 kHz (8.125 MHz),
                         centered +5, DC subcarrier nulled -> [0.94, 9.06].
                         CO-CHANNEL with PC5: during the DSRC->C-V2X sunset
                         transition, legacy 802.11p and C-V2X share the ITS
                         band — the coexistence problem this paper studies.
  Class 2  WiFi U-NII-4: OFDM, a 20 MHz channel [5875, 5895] whose upper half
                         falls inside the window. Modeled with the 26
                         in-window subcarriers of the standard 64-IFFT
                         312.5 kHz grid (channel center -10 MHz, DC nulled)
                         + CP 16 (0.8 us, 802.11 GI) -> occupies [-9.69, -1.88].
                         The out-of-window half of the channel is removed by
                         the sensing receiver's anti-alias filtering, hence
                         absent from the generated baseband.
  Class 3  Noise       : complex AWGN (full 20 MHz).

Standards-fidelity disclosure (read before reviewing):
  * 802.11p / WiFi are BLOCK-EXACT at FS = 20 MHz: 156.25 / 312.5 kHz divide
    the FFT grid exactly, symbols are true IFFT blocks + CP, DC subcarrier
    nulled, 52-subcarrier layout per IEEE 802.11-2012. WiFi's partial
    (26-subcarrier) spectrum is the physically correct view of a 20 MHz
    U-NII-4 channel through a boundary-straddling 20 MHz window.
  * PC5 is a RATE-MATCHED STYLIZATION of C-V2X Rel-14 (TS 36.211): Rel-14
    uses 15 kHz SCS x 600 subcarriers (50 PRB — the SAME 9 MHz occupied BW),
    CP 4.69 us, DMRS and 1 ms subframe structure. We use 450 x 20 kHz so the
    block is an exact integer at FS = 20 MHz. Occupied BW, PAPR, single-
    carrier envelope and CP presence are preserved; SCS/DMRS/subframe
    structure are not. No downstream result depends on 15 vs 20 kHz spacing
    (the STFT resolution, 78.125 kHz, is much coarser than both).
  * Data symbols are all-QPSK (no 16/64-QAM, no pilots or preambles).
  * Unit average power per waveform (the noise class achieves 0.9995 +- 0.02
    by finite-N statistics); SNR is applied at the channel.
  * The 2048-sample rectangular window truncates symbols mid-block, which
    splatters -30..-40 dB skirts across the full band (in-band energy
    fraction 98.4-99.7%). Real 802.11 spectral masks are -28/-40 dBr —
    disclose together with the synthetic-waveforms limitation.

Design rationale:
  * PC5 and 802.11p deliberately share the ITS channel: region-energy
    features cannot separate THOSE two classes (the discriminative physics is
    numerology — subcarrier spacing, symbol period, CP — and envelope
    statistics — SC-FDMA's ~4-6 dB PAPR vs OFDM's ~8-10 dB). WiFi is region-
    separable from the ITS pair, which is physics post-FCC (unlicensed
    emissions live below 5895 MHz). The honest 'DL necessity' claim therefore
    rests on the PC5-vs-11p pair and on adversarial robustness, not on 4-class
    separability — stated against an energy-feature logistic-regression
    baseline in run_train.py.
"""

import numpy as np

FS = 20e6                # sampling frequency
N_SAMPLES = 2048         # 102.4 us sensing window
NFFT_STFT = 256          # receiver STFT: 78.125 kHz bins, 12.8 us frames
HOP_STFT = 128

CLASS_NAMES = ["C-V2X-PC5", "802.11p", "WiFi-U-NII4", "Noise"]
NUM_CLASSES = 4
NOISE_CLASS = 3

# --- Band plan (relative MHz; window center = 5895 MHz boundary) ------------
BAND_PLAN = {
    "pc5":  {"center": 5.0,   "n_sub": 450, "block": 1000, "cp": 32,
             "sc_fdma": True,  "side": None,
             "occupies": [0.5, 9.5]},
    "11p":  {"center": 5.0,   "n_sub": 52,  "block": 128,  "cp": 32,
             "sc_fdma": False, "side": None,
             "occupies": [0.94, 9.06]},
    "wifi": {"center": -10.0, "n_sub": 26,  "block": 64,   "cp": 16,
             "sc_fdma": False, "side": +1,
             "occupies": [-9.69, -1.88]},
}

# Attacker spectral allocations (for the emission-mask-constrained threat
# model). A C-V2X attacker transmits in ITS [0, 10] MHz (upper); a U-NII-4
# attacker transmits below the boundary in [-10, 0] MHz and must keep its
# out-of-band emission toward ITS nulled.
ATTACK_BANDS = {
    "cv2x_attacker": {"lo": 0.0,  "hi": 10.0},   # C-V2X device (ITS band)
    "wifi_attacker": {"lo": -10.0, "hi": 0.0},   # U-NII-4 device (below 5895)
}


def _qpsk(rng, n):
    """True QPSK: unit-modulus symbols from the 4-point constellation
    (+-1 +- j)/sqrt(2). Const-modulus data is what gives SC-FDMA its low PAPR
    and OFDM its Rayleigh envelope — Gaussian data would destroy both."""
    bits = rng.integers(0, 4, size=n)
    return ((bits & 1) * 2 - 1 + 1j * ((bits >> 1) * 2 - 1)) / np.sqrt(2)


def _sub_positions(p, center_bin):
    """Subcarrier bin positions on the block FFT grid.

    sc_fdma  : contiguous localized allocation (SC-FDMA property).
    side=None: standard 802.11 layout -n/2..-1, +1..+n/2 (DC subcarrier nulled).
    side=+1  : only the upper half of the channel's subcarriers (the part of
               a 20 MHz U-NII-4 channel that falls inside the sensing window).
    """
    n, block = p["n_sub"], p["block"]
    if p["sc_fdma"]:
        ks = np.arange(-n // 2, -n // 2 + n)             # contiguous
    elif p["side"] is not None:
        ks = p["side"] * np.arange(1, n + 1)             # one-sided offset
    else:
        ks = np.concatenate([np.arange(-n // 2, 0),      # DC nulled
                             np.arange(1, n // 2 + 1)])
    return (center_bin + ks) % block


def _ofdm_symbol(rng, p, center_bin):
    """One (block+cp)-sample symbol. dft_spread=True gives SC-FDMA
    (DFT-s-OFDM): low-PAPR single-carrier envelope."""
    n, block, cp = p["n_sub"], p["block"], p["cp"]
    if p["sc_fdma"]:
        data = _qpsk(rng, n)
        X = np.fft.fft(data) / np.sqrt(n)
    else:
        X = _qpsk(rng, n)
    spec = np.zeros(block, dtype=np.complex128)
    positions = _sub_positions(p, center_bin)
    assert len(positions) == n and len(set(positions.tolist())) == n
    spec[positions] = X
    s = np.fft.ifft(spec) * np.sqrt(block / n)
    return np.concatenate([s[-cp:], s])


def gen_signal(cls: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one unit-power complex baseband waveform of class `cls`."""
    if cls == 3:
        s = (rng.standard_normal(N_SAMPLES) +
             1j * rng.standard_normal(N_SAMPLES)) / np.sqrt(2)
        return s.astype(np.complex64)

    key = {0: "pc5", 1: "11p", 2: "wifi"}[cls]
    p = BAND_PLAN[key]
    center_bin = int(round(p["center"] * 1e6 / (FS / p["block"])))

    parts = []
    total = 0
    while total < N_SAMPLES + p["cp"]:
        parts.append(_ofdm_symbol(rng, p, center_bin))
        total += p["block"] + p["cp"]
    s = np.concatenate(parts)[:N_SAMPLES]
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))       # unit average power
    return s.astype(np.complex64)


def gen_dataset(n_per_class=1000, seed=42):
    """Returns waveforms (N, 2048) complex64, labels (N,)."""
    rng = np.random.default_rng(seed)
    N = n_per_class * NUM_CLASSES
    X = np.empty((N, N_SAMPLES), dtype=np.complex64)
    y = np.empty(N, dtype=np.int64)
    i = 0
    for c in range(NUM_CLASSES):
        for _ in range(n_per_class):
            X[i] = gen_signal(c, rng)
            y[i] = c
            i += 1
    perm = rng.permutation(N)
    return X[perm], y[perm]
