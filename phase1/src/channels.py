"""channels.py — simplified V2X stochastic channel models (TR 37.885-inspired).

Self-contained parameterized implementation (no external dependency). Tap
delays / powers / K-factors / Doppler follow the FLAVOR of 3GPP TR 37.885
V2X profiles; they are NOT extracted from the spec's CDL tables, and the
urban delay profile in particular is UMa-flavored (delay spread ~1.3 us,
larger than typical V2V-urban NLOS ~0.1-0.3 us — conservative in frequency
selectivity). Simplified to a resolvable-tap block-fading model at FS=20 MHz:

  * Tap 0 is Rician (LOS + diffuse); later taps are pure Rayleigh (standard
    CDL practice; earlier versions made every tap Rician, which overstates
    LOS energy on reflections).
  * h(t) is constant within one 102.4 us sensing window (block fading).
    Justification: max Doppler in the worst scenario (highway, 120 km/h @
    5.9 GHz) is ~656 Hz, so the channel decorrelation time (~0.55-0.76 ms)
    far exceeds the window (even 140 km/h gives ~0.55 ms).

Scenarios
  highway : 120 km/h, Rician K = 10 dB, 5 taps, delay spread ~ 0.4 us  (LOS-ish)
  urban   :  50 km/h, Rayleigh  K = 0,   6 taps, delay spread ~ 1.3 us  (NLOS, UMa-flavored)
  rural   :  30 km/h, Rician K = 5 dB,  3 taps, delay spread ~ 0.5 us  (partial LOS)

Used for BOTH the victim link (transmitter -> sensing receiver) and,
independently, the attacker link (attacker -> sensing receiver) in
attack_mask.py.
"""

import numpy as np

FS = 20e6
TAP_DT = 1.0 / FS          # 50 ns resolvable tap spacing

SCENARIOS = {
    "highway": {
        "speed_kmh": 120, "K_dB": 10.0,
        "delays_ns": [0, 50, 120, 200, 400],
        "powers_dB": [0, -8, -12, -16, -20],
    },
    "urban": {
        "speed_kmh": 50, "K_dB": 0.0,
        "delays_ns": [0, 100, 200, 400, 800, 1300],
        "powers_dB": [0, -3, -5, -8, -10, -13],
    },
    "rural": {
        "speed_kmh": 30, "K_dB": 5.0,
        "delays_ns": [0, 100, 500],
        "powers_dB": [0, -6, -12],
    },
}

SCENARIO_NAMES = list(SCENARIOS.keys())


def draw_channel(scenario: str, rng: np.random.Generator) -> np.ndarray:
    """Draw one complex baseband FIR channel realization (unit total power).

    Returns complex64 array of length max_tap+1 (taps quantized to the 50 ns
    grid). Tap 0: Rician with scenario K (sqrt(K/(K+1)) e^{j theta} +
    sqrt(1/(K+1)) CN(0,1), theta ~ U(0,2pi)); taps >= 1: Rayleigh CN(0,1)
    scaled by the tap power profile.
    """
    p = SCENARIOS[scenario]
    K = 10 ** (p["K_dB"] / 10.0)
    taps_idx = [int(round(d / 50.0)) for d in p["delays_ns"]]
    L = max(taps_idx) + 1
    h = np.zeros(L, dtype=np.complex128)
    for j, (idx, pdb) in enumerate(zip(taps_idx, p["powers_dB"])):
        pl = 10 ** (pdb / 10.0)
        if j == 0:                       # first tap: Rician LOS + diffuse
            los = np.sqrt(K / (K + 1.0)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
            dif = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
            tap = los + np.sqrt(1.0 / (K + 1.0)) * dif
        else:                            # later taps: pure Rayleigh
            tap = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        h[idx] += np.sqrt(pl) * tap / np.sqrt(len(taps_idx))
    h = h / np.sqrt(np.sum(np.abs(h) ** 2))        # unit total power
    return h.astype(np.complex64)


def apply_channel_batch(X: np.ndarray, scenario: str, rng: np.random.Generator,
                        snr_db_range=(5.0, 25.0), snr_ref="signal"):
    """Convenience batch helper (kept for reuse in future DeepSense-style
    real-capture pipelines; the experiment runners inline their own loops for
    full control of per-sample SNR and attacker-link draws).

    X: (B, L) complex64 unit-average-power signals.
    Returns (X_rx, meta) where X_rx has the same shape, and meta records per-sample
    SNR and channel norm for reproducibility.
    """
    B, L = X.shape
    X_rx = np.empty_like(X)
    snrs = np.empty(B, dtype=np.float32)
    for i in range(B):
        h = draw_channel(scenario, rng)
        # convolve (full), keep 'same' centered segment
        y = np.convolve(X[i], h)[ (len(h) - 1) // 2: (len(h) - 1) // 2 + L]
        p_sig = np.mean(np.abs(y) ** 2)
        snr = rng.uniform(*snr_db_range)
        n_std = np.sqrt(p_sig / (10 ** (snr / 10.0)) / 2.0)
        y = y + n_std * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        X_rx[i] = y.astype(np.complex64)
        snrs[i] = snr
    return X_rx, {"snr_db": snrs, "scenario": scenario}
