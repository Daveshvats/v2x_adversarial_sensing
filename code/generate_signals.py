"""
generate_signals.py
Generate synthetic V2X RF signals as spectrograms for spectrum sensing.
"""

import os
import random
import numpy as np
import torch
from scipy.signal import stft
from scipy.ndimage import zoom
from tqdm import tqdm

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# 5 modulation types: 4 real modulations + noise-only class
MODULATIONS = ["BPSK", "QPSK", "16QAM", "64QAM", "Noise"]
MOD_MAP = {m: i for i, m in enumerate(MODULATIONS)}

# 7 SNR levels spanning the range from very noisy (-10 dB) to very clean (20 dB)
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]

# 200 samples per modulation type per SNR level
# Total: 5 modulations × 7 SNR levels × 200 = 7,000 samples
SAMPLES_PER_MOD_PER_SNR = 200

# Output spectrogram resolution: 64×64 pixels (standard for CNN input)
SPEC_SIZE = 64

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Sampling frequency: 10 kHz (chosen as a tractable rate for simulation;
# real V2X operates at ~10 MHz, but the relative signal structure is preserved)
FS = 10000


def _constellation(mod, n):
    """Generate random constellation symbols for a given modulation scheme.
    
    Each modulation has a specific constellation layout in the complex I/Q plane:
      - BPSK: 1 bit/symbol, points at ±1 (binary phase shift keying)
      - QPSK: 2 bits/symbol, 4 points equally spaced on the unit circle
        (normalized by 1/√2 for unit average power)
      - 16QAM: 4 bits/symbol, 4×4 grid normalized by 1/√10 for unit power
      - 64QAM: 6 bits/symbol, 8×8 grid normalized by 1/√42 for unit power
    
    Power normalization ensures fair comparison across modulation types,
    since signal power directly affects the SNR calculation.
    """
    if mod == "BPSK":
        # 2 points: {-1, +1} on the real axis
        return np.random.choice([-1.0, 1.0], n).astype(complex)
    elif mod == "QPSK":
        # 4 points at ±45° angles, normalized to unit power
        choices = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        return choices[np.random.randint(0, 4, n)]
    elif mod == "16QAM":
        # 4×4 grid: {-3, -1, +1, +3} on both I and Q axes
        # Normalized by √10 so that E[|x|²] = 1
        pts = np.array([-3,-1,1,3], dtype=float) / np.sqrt(10)
        I = np.random.choice(pts, n); Q = np.random.choice(pts, n)
        return I + 1j * Q
    elif mod == "64QAM":
        # 8×8 grid: {-7, -5, -3, -1, +1, +3, +5, +7} on both axes
        # Normalized by √42 so that E[|x|²] = 1
        pts = np.array([-7,-5,-3,-1,1,3,5,7], dtype=float) / np.sqrt(42)
        I = np.random.choice(pts, n); Q = np.random.choice(pts, n)
        return I + 1j * Q
    # Fallback for Noise class: BPSK-like random symbols (unused in practice)
    return np.random.choice([-1.0, 1.0], n).astype(complex)


def gen_signal(mod, n_samples=2048, sps=10):
    """Generate a baseband complex signal for the given modulation.
    
    Steps:
      1. Generate random constellation symbols at the symbol rate.
      2. Upsample by inserting zeros (sps samples per symbol).
      3. Apply a raised cosine pulse shaping filter.
      4. Normalize to unit average power.
    
    The raised cosine filter (β=0.35) minimizes inter-symbol interference (ISI)
    while providing a good tradeoff between spectral efficiency and pulse width.
    sps=10 (samples per symbol) provides adequate oversampling for the filter.
    
    Args:
        mod: modulation type (BPSK, QPSK, 16QAM, 64QAM, Noise)
        n_samples: total number of output IQ samples
        sps: samples per symbol (oversampling factor)
    
    Returns:
        Complex baseband signal of length n_samples
    """
    # Number of modulation symbols = total samples / oversampling factor
    n_sym = n_samples // sps
    if mod == "Noise":
        # Pure complex Gaussian noise (no modulation structure)
        return np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    
    # Step 1: Generate baseband symbols
    syms = _constellation(mod, n_sym)
    
    # Step 2: Upsample by inserting zeros between symbols (impulse train)
    sig_up = np.zeros(n_sym * sps, dtype=complex)
    sig_up[::sps] = syms
    
    # Step 3: Raised cosine pulse shaping filter
    # The raised cosine filter is defined as:
    #   h(t) = sinc(t/T) * cos(πβt/T) / (1 - (2βt/T)²)
    # where T = sps, β = roll-off factor
    t = np.arange(-3 * sps, 3 * sps + 1).astype(float)
    beta = 0.35  # Roll-off factor: controls excess bandwidth beyond Nyquist
    sinc_v = np.sinc(t / sps)  # sinc(t/T) = sin(πt/T) / (πt/T)
    cos_v = np.cos(np.pi * beta * t / sps)
    den = 1 - (2 * beta * t / sps) ** 2
    # Avoid division by zero at t=0 where den=0 (limit is β/(4β) = 1)
    den = np.where(np.abs(den) < 1e-12, 1.0, den)
    h = sinc_v * cos_v / den
    h[3*sps] = 1.0  # Explicitly fix the center tap (Nyquist criterion)
    # Normalize filter energy to preserve signal power after convolution
    h = h / np.sqrt(np.sum(h**2) + 1e-12)
    
    # Apply filter via convolution and truncate to desired length
    sig = np.convolve(sig_up, h, mode='full')[:n_samples]
    
    # Step 4: Normalize to unit average power for consistent SNR control
    pwr = np.mean(np.abs(sig)**2)
    if pwr > 1e-12:
        sig = sig / np.sqrt(pwr)
    return sig


def add_awgn(sig, snr_db):
    """Add Additive White Gaussian Noise to achieve a target SNR.
    
    SNR is defined as: SNR(dB) = 10 * log10(signal_power / noise_power)
    
    Therefore: noise_power = signal_power / 10^(SNR_dB/10)
    
    The noise is complex Gaussian with independent real and imaginary components,
    each with variance noise_power/2, so that:
      E[|noise|²] = E[Re²] + E[Im²] = noise_power/2 + noise_power/2 = noise_power
    """
    sp = np.mean(np.abs(sig)**2)  # Average signal power
    if sp < 1e-15:
        return sig  # Avoid division by zero for silent signals
    # Convert SNR from dB to linear scale, then compute noise power
    noise_pwr = sp / (10 ** (snr_db / 10))
    # Generate complex Gaussian noise with the computed power
    noise = np.sqrt(noise_pwr / 2) * (np.random.randn(len(sig)) + 1j * np.random.randn(len(sig)))
    return sig + noise


def to_spec(sig, size=SPEC_SIZE):
    """Convert a complex baseband signal to a log-magnitude spectrogram.
    
    Pipeline:
      1. Compute STFT of the real part of the signal
         (scipy's stft uses Hann window internally for spectral leakage control)
      2. Take the magnitude of the complex STFT output
      3. Resize (zoom) to the target spectrogram size (64×64)
      4. Apply log1p compression: log(1 + x*100) — this is a standard
         technique to compress the large dynamic range of spectrogram values
         into a range suitable for neural network input
      5. Cast to float32 for PyTorch compatibility
    
    STFT parameters:
      - nperseg=128: FFT window length (frequency resolution)
      - noverlap=96: 75% overlap between consecutive windows
        (higher overlap = better time resolution, more computation)
    
    Args:
        sig: complex baseband signal
        size: target spectrogram dimension (size × size)
    
    Returns:
        Log-compressed magnitude spectrogram as float32 array of shape (size, size)
    """
    nperseg = 128
    noverlap = 96
    # Use only the real part for STFT (imaginary part is redundant for
    # the magnitude spectrum of a baseband signal)
    f, t, Zxx = stft(sig.real, fs=FS, nperseg=nperseg, noverlap=noverlap,
                     window='hann', return_onesided=True)
    mag = np.abs(Zxx)
    # Bilinear interpolation to resize the spectrogram to 64×64
    mag = zoom(mag, (size / mag.shape[0], size / mag.shape[1]), order=1)[:size, :size]
    # Pad with zeros if the STFT produced fewer frames than needed
    if mag.shape[0] < size or mag.shape[1] < size:
        p = np.zeros((size, size))
        p[:mag.shape[0], :mag.shape[1]] = mag
        mag = p
    # Log compression: log(1 + 100*mag) maps [0, ∞) to [0, ∞) with compression
    # The factor 100 controls the compression strength
    mag = np.log1p(mag * 100)
    return mag.astype(np.float32)


def generate_dataset():
    """Generate the full synthetic V2X RF spectrogram dataset.
    
    For each combination of (modulation × SNR):
      1. Generate a baseband signal with the appropriate modulation
      2. Apply Rayleigh fading (convolved Gaussian for temporal correlation)
         to simulate multipath wireless channel effects
      3. Add AWGN at the target SNR level
      4. Convert to a 64×64 log-magnitude spectrogram
    
    After generation:
      - Shuffle all samples randomly
      - Compute global mean/std from the training split only (to avoid data leakage)
      - Z-score normalize all splits with the training statistics
      - Save train/val/test splits as compressed .npz files
    
    Split ratios: 70% train, 15% val, 15% test
    
    Data stored as float16 to save disk space (~50% reduction vs float32)
    with negligible impact on spectrogram quality.
    """
    print("=" * 60)
    print("GENERATING SYNTHETIC V2X RF SIGNALS")
    total = len(MODULATIONS) * len(SNR_LEVELS) * SAMPLES_PER_MOD_PER_SNR
    print(f"  Total: {total} samples ({SPEC_SIZE}x{SPEC_SIZE})")
    print("=" * 60)

    all_specs = []
    all_labels = []
    all_snrs = []

    for si, snr in enumerate(SNR_LEVELS):
        print(f"\n--- SNR = {snr} dB ({si+1}/{len(SNR_LEVELS)}) ---")
        for mod in MODULATIONS:
            for _ in tqdm(range(SAMPLES_PER_MOD_PER_SNR), desc=f"  {mod:>6s}", leave=False):
                # Generate baseband signal (2048 samples at 10 samples/symbol = 204 symbols)
                iq = gen_signal(mod, n_samples=2048, sps=10)
                if mod != "Noise":
                    # Apply Rayleigh fading to simulate multipath channel
                    # h(t) = complex Gaussian filtered by a 30-sample moving average
                    # This introduces temporal correlation in the fading process,
                    # modeling the slowly varying nature of real wireless channels
                    fade = (np.random.randn(2048) + 1j * np.random.randn(2048)) / np.sqrt(2)
                    # Smooth the fading process (moving average filter with window=30)
                    # This controls the coherence time of the fading channel
                    fade = np.convolve(fade, np.ones(30)/30, mode='same')
                    # Normalize fading to unit average power
                    fade /= np.sqrt(np.mean(np.abs(fade)**2)) + 1e-12
                    iq = iq * fade
                # Add noise at the target SNR
                iq = add_awgn(iq, snr)
                # Convert to spectrogram and append
                all_specs.append(to_spec(iq))
                all_labels.append(MOD_MAP[mod])
                all_snrs.append(si)

    X = np.stack(all_specs)
    y = np.array(all_labels, dtype=np.int64)
    snr_idx = np.array(all_snrs, dtype=np.int64)
    print(f"\nDataset: X={X.shape}, y={y.shape}")

    # Random shuffle with deterministic seed
    idx = np.arange(len(X)); np.random.shuffle(idx)
    n_tr = int(0.70 * len(X)); n_va = int(0.15 * len(X))
    
    # Z-score normalization: fit on training data only to prevent data leakage
    gm, gs = X[idx[:n_tr]].mean(), X[idx[:n_tr]].std()
    print(f"Global: mean={gm:.4f}, std={gs:.4f}")
    X = (X - gm) / (gs + 1e-12)

    # Save each split as a compressed .npz file
    # X stored with newaxis to add channel dimension: (N, 1, 64, 64)
    for name, ii in [("train", idx[:n_tr]), ("val", idx[n_tr:n_tr+n_va]), ("test", idx[n_tr+n_va:])]:
        np.savez_compressed(os.path.join(DATA_DIR, f"{name}.npz"),
                            X=X[ii, np.newaxis].astype(np.float16),
                            y=y[ii], snr=snr_idx[ii])
        print(f"  {name}.npz: {len(ii)} samples")
    return True


if __name__ == "__main__":
    generate_dataset()
