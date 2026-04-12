#!/usr/bin/env python3
"""
V2X Spectrum Sensing Dataset Generation Pipeline
=================================================
Generates realistic RF spectrogram datasets for V2X spectrum sensing research.

Modes:
  1. Simulation-Only  – Pure Python/NumPy; no SUMO or GNU Radio needed.
  2. SUMO + GNU Radio – Creates SUMO configs, parses FCD, generates GNU Radio scripts.

Dependencies: numpy, scipy, matplotlib, h5py  (all standard scientific Python)
Run:          python generate_simulated_dataset.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports – graceful degradation if plotting / HDF5 are missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

warnings.filterwarnings("ignore", category=FutureWarning)

# ===================================================================
# GLOBAL CONFIGURATION
# ===================================================================
@dataclass
class Config:
    """Master configuration for the entire pipeline."""

    # --- Dataset ---
    NUM_SAMPLES: int = 10_000
    SAMPLE_LENGTH: int = 1024        # IQ samples per spectrogram slice
    FS: float = 10e6                 # 10 MHz sampling rate
    NFFT: int = 128                  # FFT size for STFT
    HOP: int = 64                    # Hop size for STFT
    CLASS_NAMES: List[str] = field(default_factory=lambda: ["LTE", "WiFi", "V2X-PC5", "Noise"])
    SEED: int = 42

    # --- RF ---
    V2X_FC: float = 5.9e9            # DSRC / C-V2X carrier frequency (Hz)
    C: float = 2.998e8               # Speed of light (m/s)

    # --- Channel ---
    SNR_RANGE: Tuple[float, float] = (0.0, 30.0)
    NUM_MULTIPATH_TAPS: int = 4
    PDP_DECAY: float = 3.0           # Exponential PDP decay factor (μs)

    # --- Scenarios ---
    SCENARIOS: List[str] = field(default_factory=lambda: ["highway", "urban", "rural"])

    # --- Paths ---
    OUTPUT_DIR: str = "simulated_rf_dataset"
    MODE: str = "simulation"         # "simulation" or "sumo_gnuradio"

    def __post_init__(self):
        # Pre-compute some derived quantities
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        self.SAMPLES_PER_CLASS = self.NUM_SAMPLES // self.NUM_CLASSES


# ===================================================================
# UTILITY HELPERS
# ===================================================================
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def linear_to_db(lin: float) -> float:
    return 10.0 * np.log10(np.maximum(lin, 1e-12))


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def progress_bar(current: int, total: int, prefix: str = "", bar_len: int = 40) -> None:
    frac = current / total
    filled = int(bar_len * frac)
    bar = "█" * filled + "░" * (bar_len - filled)
    sys.stdout.write(f"\r{prefix}|{bar}| {current}/{total} ({100*frac:.1f}%)")
    sys.stdout.flush()
    if current == total:
        print()


# ===================================================================
# SECTION 1 – MOBILITY SIMULATION  (mimics SUMO output)
# ===================================================================
@dataclass
class VehicleState:
    """State of a single vehicle at a single timestep."""
    x: float = 0.0
    y: float = 0.0
    speed_ms: float = 0.0        # m/s
    heading_rad: float = 0.0     # radians, 0 = east
    lane: int = 0


@dataclass
class MobilitySnapshot:
    """Channel-relevant mobility data between a TX–RX vehicle pair."""
    timestep: float = 0.0
    distance_m: float = 100.0
    relative_speed_ms: float = 0.0
    angle_of_arrival_rad: float = 0.0
    tx_speed_ms: float = 0.0
    rx_speed_ms: float = 0.0


class MobilitySimulator:
    """
    Generates synthetic vehicle trajectories for three scenarios.
    Each scenario is characterised by speed profiles, road geometry,
    and vehicle density that match realistic V2X environments.

    The output format mirrors SUMO Floating Car Data (FCD) so that the
    downstream channel extraction code is identical for Mode 1 and Mode 2.
    """

    # Scenario parameter presets
    SCENARIO_PARAMS = {
        "highway": {
            "num_lanes": 3,
            "road_length_m": 5000,
            "speed_range_kmh": (80, 120),
            "num_vehicles": 15,
            "inter_vehicle_dist_m": (20, 100),
            "direction": "forward",
            "description": "Highway V2V – constant high speed",
        },
        "urban": {
            "num_lanes": 2,
            "road_length_m": 1000,
            "speed_range_kmh": (0, 60),
            "num_vehicles": 10,
            "inter_vehicle_dist_m": (5, 50),
            "direction": "mixed",
            "description": "Urban intersection V2I – variable speed with stops",
        },
        "rural": {
            "num_lanes": 2,
            "road_length_m": 3000,
            "speed_range_kmh": (40, 80),
            "num_vehicles": 6,
            "inter_vehicle_dist_m": (50, 200),
            "direction": "forward",
            "description": "Rural road V2V – moderate speed, sparse vehicles",
        },
    }

    def __init__(self, config: Config):
        self.cfg = config
        self.rng = np.random.RandomState(config.SEED + 1)

    # ---- highway trajectory ----
    def _generate_highway_trajectory(self, params: dict) -> List[VehicleState]:
        n_steps = 100
        states: List[VehicleState] = []
        speed_kmh = self.rng.uniform(*params["speed_range_kmh"])
        speed_ms = speed_kmh / 3.6
        lane = self.rng.randint(0, params["num_lanes"])
        lane_offset = lane * 3.7  # lane width ≈ 3.7 m

        x = self.rng.uniform(0, 200)
        for t in range(n_steps):
            # Small speed perturbation ±5 km/h
            speed_ms += self.rng.normal(0, 0.5)
            speed_ms = np.clip(speed_ms, params["speed_range_kmh"][0] / 3.6,
                               params["speed_range_kmh"][1] / 3.6)
            x += speed_ms * 0.1  # dt = 0.1 s per step
            if x > params["road_length_m"]:
                x -= params["road_length_m"]
            states.append(VehicleState(
                x=x, y=lane_offset,
                speed_ms=speed_ms,
                heading_rad=0.0,  # east
                lane=lane,
            ))
        return states

    # ---- urban trajectory (with intersection stops) ----
    def _generate_urban_trajectory(self, params: dict) -> List[VehicleState]:
        n_steps = 200
        states: List[VehicleState] = []
        speed_kmh = self.rng.uniform(20, 50)
        speed_ms = speed_kmh / 3.6
        lane = self.rng.randint(0, params["num_lanes"])
        lane_offset = lane * 3.5
        intersection_positions = [250, 500, 750]  # traffic lights
        x = self.rng.uniform(0, 100)

        # Random direction
        direction = self.rng.choice([-1, 1])

        for t in range(n_steps):
            # Slow down near intersections
            near_light = any(abs(x - ix) < 30 for ix in intersection_positions)
            if near_light and self.rng.random() < 0.4:
                speed_ms *= 0.7  # decelerate
            else:
                speed_ms += self.rng.normal(0, 1.0)
            speed_ms = np.clip(speed_ms, 0, params["speed_range_kmh"][1] / 3.6)

            x += direction * speed_ms * 0.1
            if x > params["road_length_m"]:
                x -= params["road_length_m"]
            elif x < 0:
                x += params["road_length_m"]

            heading = 0.0 if direction > 0 else math.pi
            states.append(VehicleState(
                x=x, y=lane_offset,
                speed_ms=speed_ms,
                heading_rad=heading,
                lane=lane,
            ))
        return states

    # ---- rural trajectory ----
    def _generate_rural_trajectory(self, params: dict) -> List[VehicleState]:
        n_steps = 120
        states: List[VehicleState] = []
        speed_kmh = self.rng.uniform(*params["speed_range_kmh"])
        speed_ms = speed_kmh / 3.6
        lane = self.rng.randint(0, params["num_lanes"])
        lane_offset = lane * 3.7
        x = self.rng.uniform(0, 500)

        for t in range(n_steps):
            speed_ms += self.rng.normal(0, 0.8)
            speed_ms = np.clip(speed_ms, params["speed_range_kmh"][0] / 3.6,
                               params["speed_range_kmh"][1] / 3.6)
            x += speed_ms * 0.1
            if x > params["road_length_m"]:
                x -= params["road_length_m"]
            states.append(VehicleState(
                x=x, y=lane_offset,
                speed_ms=speed_ms,
                heading_rad=0.0,
                lane=lane,
            ))
        return states

    def generate_trajectory(self, scenario: str) -> List[VehicleState]:
        """Generate a single-vehicle trajectory for the given scenario."""
        params = self.SCENARIO_PARAMS[scenario]
        if scenario == "highway":
            return self._generate_highway_trajectory(params)
        elif scenario == "urban":
            return self._generate_urban_trajectory(params)
        else:
            return self._generate_rural_trajectory(params)

    def compute_mobility_snapshot(
        self,
        tx: VehicleState,
        rx: VehicleState,
        timestep: float,
    ) -> MobilitySnapshot:
        """Compute channel-relevant mobility between a TX and RX pair."""
        dx = tx.x - rx.x
        dy = tx.y - rx.y
        distance = math.sqrt(dx ** 2 + dy ** 2) + 1e-3  # avoid zero

        # Angle of arrival (from RX toward TX)
        aoa = math.atan2(dy, dx)

        # Relative speed (projection onto the TX–RX axis)
        tx_vx = tx.speed_ms * math.cos(tx.heading_rad)
        tx_vy = tx.speed_ms * math.sin(tx.heading_rad)
        rx_vx = rx.speed_ms * math.cos(rx.heading_rad)
        rx_vy = rx.speed_ms * math.sin(rx.heading_rad)
        rel_vx = tx_vx - rx_vx
        rel_vy = tx_vy - rx_vy
        # Project onto the line of sight
        unit_los_x = dx / distance
        unit_los_y = dy / distance
        relative_speed = rel_vx * unit_los_x + rel_vy * unit_los_y

        return MobilitySnapshot(
            timestep=timestep,
            distance_m=distance,
            relative_speed_ms=relative_speed,
            angle_of_arrival_rad=aoa,
            tx_speed_ms=tx.speed_ms,
            rx_speed_ms=rx.speed_ms,
        )

    def generate_vehicle_pair_snapshots(
        self, scenario: str, n_snapshots: int = 1
    ) -> List[MobilitySnapshot]:
        """Generate *n_snapshots* mobility snapshots for a random vehicle pair."""
        tx_traj = self.generate_trajectory(scenario)
        rx_traj = self.generate_trajectory(scenario)

        length = min(len(tx_traj), len(rx_traj))
        indices = sorted(self.rng.choice(length, size=min(n_snapshots, length), replace=False))
        return [
            self.compute_mobility_snapshot(tx_traj[i], rx_traj[i], float(i) * 0.1)
            for i in indices
        ]


# ===================================================================
# SECTION 2 – CHANNEL PARAMETER EXTRACTION
# ===================================================================
@dataclass
class ChannelParams:
    """All channel parameters needed for waveform generation."""
    pathloss_db: float = 100.0
    doppler_hz: float = 0.0
    rician_k_db: float = 5.0
    shadowing_db: float = 0.0
    snr_db: float = 15.0
    coherence_time_ms: float = 10.0
    multipath_delays_us: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.3, 0.7, 1.5]))
    multipath_powers_db: np.ndarray = field(default_factory=lambda: np.array([0, -3, -7, -12]))
    distance_m: float = 100.0
    relative_speed_ms: float = 0.0
    scenario: str = "highway"


class ChannelModel:
    """
    Compute physical channel parameters from mobility snapshots.

    Models (all from standard wireless communications references):
        - Pathloss: 3GPP TR 36.885 Urban Macro (PL = 128.1 + 37.6 log10(d_km))
        - Doppler:  f_d = v_rel * cos(θ) * f_c / c  (classical Doppler equation)
        - Rician K-factor: scenario-dependent ranges (from WINNER II channel measurements)
        - Shadowing: log-normal, σ = 4–10 dB depending on environment clutter
        - Coherence time: T_c = 0.423 / f_d  (Clark's model, widely used approximation)
        - Power delay profile: exponential decay with scenario-dependent spread
    """

    SCENARIO_RICIAN_K = {
        "highway": (3.0, 6.0),    # moderate LOS, V2V
        "urban":   (1.0, 3.0),    # weak LOS due to buildings, V2I
        "rural":   (6.0, 10.0),   # strong LOS, V2V
    }

    SCENARIO_SHADOWING_SIGMA = {
        "highway": 4.0,
        "urban":   10.0,
        "rural":   6.0,
    }

    SCENARIO_PDP_SPREAD = {
        "highway": 0.5,   # μs – relatively compact
        "urban":   2.0,   # μs – lots of multipath from buildings
        "rural":   1.0,   # μs – moderate
    }

    def __init__(self, config: Config):
        self.cfg = config
        self.rng = np.random.RandomState(config.SEED + 2)

    def compute_pathloss(self, distance_m: float) -> float:
        """3GPP Urban Macro pathloss model.
        
        PL(dB) = 128.1 + 37.6 * log10(d_km)  for d > 35m (3GPP TR 36.885)
        This models the large-scale signal attenuation over distance.
        """
        d_km = distance_m / 1000.0
        pl = 128.1 + 37.6 * math.log10(max(d_km, 1e-6))
        return pl

    def compute_doppler(
        self, relative_speed_ms: float, angle_of_arrival_rad: float
    ) -> float:
        """Doppler shift: f_d = v_radial * f_c / c.
        
        Only the radial component of velocity (projected along the TX-RX line
        of sight via cos(θ)) contributes to the Doppler shift. This is the
        classical Doppler effect: f_d = f_c * v/c where v is radial velocity.
        """
        v_radial = relative_speed_ms * math.cos(angle_of_arrival_rad)
        return abs(v_radial * self.cfg.V2X_FC / self.cfg.C)

    def compute_rician_k(self, scenario: str) -> float:
        """Draw Rician K-factor from scenario-dependent range."""
        lo, hi = self.SCENARIO_RICIAN_K[scenario]
        return self.rng.uniform(lo, hi)

    def compute_shadowing(self, scenario: str) -> float:
        """Log-normal shadowing."""
        sigma = self.SCENARIO_SHADOWING_SIGMA[scenario]
        return self.rng.normal(0, sigma)

    def compute_coherence_time(self, doppler_hz: float) -> float:
        """Clark's coherence time: T_c = 0.423 / f_d (in ms).
        
        Coherence time is the duration over which the channel appears constant.
        It determines how often the channel needs to be re-estimated.
        At f_d < 1 mHz, we treat the channel as effectively static (1000s ms).
        """
        if doppler_hz < 1e-3:
            return 1000.0  # effectively static
        return 0.423 / doppler_hz * 1000.0  # ms

    def generate_multipath(self, scenario: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate multipath delay profile (exponential PDP)."""
        n_taps = self.cfg.NUM_MULTIPATH_TAPS
        spread = self.SCENARIO_PDP_SPREAD[scenario]
        # Delays: exponential sampling
        delays = np.sort(self.rng.exponential(spread * 0.3, size=n_taps))
        delays[0] = 0.0  # LOS tap at delay 0
        # Powers: exponential decay
        powers = np.array([0.0] + list(
            -self.rng.exponential(self.cfg.PDP_DECAY, size=n_taps - 1)
        ))
        return delays, powers

    def extract_channel_params(
        self, mob: MobilitySnapshot, scenario: str
    ) -> ChannelParams:
        """Full channel parameter extraction from a single mobility snapshot."""
        doppler = self.compute_doppler(mob.relative_speed_ms, mob.angle_of_arrival_rad)
        rician_k = self.compute_rician_k(scenario)
        shadowing = self.compute_shadowing(scenario)
        pathloss = self.compute_pathloss(mob.distance_m) + shadowing

        # SNR is drawn from the configured range
        snr_db = self.rng.uniform(*self.cfg.SNR_RANGE)

        delays, powers = self.generate_multipath(scenario)

        return ChannelParams(
            pathloss_db=pathloss,
            doppler_hz=doppler,
            rician_k_db=rician_k,
            shadowing_db=shadowing,
            snr_db=snr_db,
            coherence_time_ms=self.compute_coherence_time(doppler),
            multipath_delays_us=delays,
            multipath_powers_db=powers,
            distance_m=mob.distance_m,
            relative_speed_ms=mob.relative_speed_ms,
            scenario=scenario,
        )


# ===================================================================
# SECTION 3 – WAVEFORM GENERATION  (mimics GNU Radio + gr-ieee802-11)
# ===================================================================
class WaveformGenerator:
    """
    Generate baseband complex IQ waveforms for four signal classes.

    Each waveform is carefully crafted to produce distinct spectrogram
    features that mimic what a real SDR would capture:

      - LTE:   OFDM, 64 subcarriers, 15 kHz spacing, ±35% BW offset
      - WiFi:  OFDM, 52 subcarriers, 312.5 kHz spacing, +15% freq offset
      - V2X-PC5: SC-FDMA (DFT-spread OFDM), 12 subcarriers, -10% offset
      - Noise: Complex AWGN
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.rng = np.random.RandomState(config.SEED + 3)

    def _qpsk_symbols(self, n: int) -> np.ndarray:
        """Generate random QPSK symbols."""
        idx = self.rng.randint(0, 4, size=n)
        constellation = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / math.sqrt(2)
        return constellation[idx]

    def _qam16_symbols(self, n: int) -> np.ndarray:
        """Generate random 16-QAM symbols."""
        levels = np.array([-3, -1, 1, 3]) / math.sqrt(10)
        real = self.rng.choice(levels, size=n)
        imag = self.rng.choice(levels, size=n)
        return real + 1j * imag

    def _ofdm_modulate(
        self,
        n_symbols: int,
        n_subcarriers: int,
        subcarrier_spacing_hz: float,
        n_cp: int = 16,
        data_symbols: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Generic OFDM modulator.

        Parameters
        ----------
        n_symbols : int
            Number of OFDM symbols.
        n_subcarriers : int
            Total number of subcarriers (data + pilot + guard).
        subcarrier_spacing_hz : float
            Subcarrier spacing in Hz (for setting signal bandwidth).
        n_cp : int
            Cyclic prefix length in samples.
        data_symbols : np.ndarray or None
            Pre-generated constellation symbols. If None, QPSK is used.

        Returns
        -------
        np.ndarray
            Complex baseband time-domain signal.
        """
        total_samples_per_symbol = n_subcarriers + n_cp
        signal = np.zeros(n_symbols * total_samples_per_symbol, dtype=complex)

        n_data = int(n_subcarriers * 0.75)  # ~75% data subcarriers
        n_guard = (n_subcarriers - n_data) // 2

        for sym_idx in range(n_symbols):
            # Build frequency-domain OFDM symbol
            fd = np.zeros(n_subcarriers, dtype=complex)
            if data_symbols is None:
                data = self._qpsk_symbols(n_data)
            else:
                start = sym_idx * n_data
                end = start + n_data
                data = data_symbols[start:end]
            fd[n_guard:n_guard + n_data] = data

            # IFFT to get time domain
            td = np.fft.ifft(np.fft.ifftshift(fd)) * math.sqrt(n_subcarriers)

            # Add cyclic prefix
            cp = td[-n_cp:]
            ofdm_sym = np.concatenate([cp, td])

            start_idx = sym_idx * total_samples_per_symbol
            signal[start_idx:start_idx + total_samples_per_symbol] = ofdm_sym

        return signal

    def generate_lte(self, length: int) -> np.ndarray:
        """
        Generate LTE-like waveform.

        OFDM with 64 subcarriers, 15 kHz spacing, ±35% bandwidth occupation.
        Bandwidth ≈ 64 × 15 kHz = 960 kHz relative to the 10 MHz observation BW.
        """
        n_sc = 64
        sc_spacing = 15e3
        n_cp = 16
        # Bandwidth occupation ≈ 35% → center ±35% of the total 10 MHz BW
        bw_fraction = 0.35
        n_symbols_needed = int(length / (n_sc + n_cp)) + 2

        base = self._ofdm_modulate(n_symbols_needed, n_sc, sc_spacing, n_cp,
                                    data_symbols=self._qam16_symbols(n_symbols_needed * 48))
        base = base[:length]

        # Frequency shift to place signal within ±35% of the observation BW
        freq_offset = bw_fraction * self.cfg.FS * self.rng.choice([-1, 1])
        t = np.arange(length) / self.cfg.FS
        base = base * np.exp(1j * 2 * np.pi * freq_offset * t)

        # Apply power normalisation
        base = base / (np.max(np.abs(base)) + 1e-10) * 0.8

        # Pad or truncate to exact length
        if len(base) < length:
            base = np.concatenate([base, np.zeros(length - len(base), dtype=complex)])
        return base[:length]

    def generate_wifi(self, length: int) -> np.ndarray:
        """
        Generate WiFi (802.11a/g/p) -like waveform.

        OFDM with 52 data subcarriers, 312.5 kHz spacing, +15% freq offset.
        """
        n_sc = 64  # FFT size (52 data + 4 pilot + 8 guard)
        sc_spacing = 312.5e3
        n_cp = 16
        n_symbols_needed = int(length / (n_sc + n_cp)) + 2

        # Use 52 active subcarriers out of 64
        base = self._ofdm_modulate(n_symbols_needed, n_sc, sc_spacing, n_cp,
                                    data_symbols=self._qpsk_symbols(n_symbols_needed * 48))
        base = base[:length]

        # Frequency offset: +15% of observation BW
        freq_offset = 0.15 * self.cfg.FS
        t = np.arange(length) / self.cfg.FS
        base = base * np.exp(1j * 2 * np.pi * freq_offset * t)

        base = base / (np.max(np.abs(base)) + 1e-10) * 0.7

        if len(base) < length:
            base = np.concatenate([base, np.zeros(length - len(base), dtype=complex)])
        return base[:length]

    def generate_v2x_pc5(self, length: int) -> np.ndarray:
        """
        Generate V2X PC5 (C-V2X sidelink) -like waveform.

        SC-FDMA (DFT-spread OFDM) with 12 subcarriers (one PRB),
        -10% frequency offset.
        """
        n_sc = 64  # IFFT size
        n_rb = 12   # one physical resource block
        sc_spacing = 15e3
        n_cp = 16
        n_symbols_needed = int(length / (n_sc + n_cp)) + 2

        signal = np.zeros(n_symbols_needed * (n_sc + n_cp), dtype=complex)
        n_guard = (n_sc - n_rb) // 2

        for sym_idx in range(n_symbols_needed):
            # SC-FDMA: DFT-precoding of QPSK symbols
            qpsk = self._qpsk_symbols(n_rb)
            dft_out = np.fft.fft(qpsk) * math.sqrt(n_rb)  # DFT-spread

            # Map to subcarriers (centered)
            fd = np.zeros(n_sc, dtype=complex)
            fd[n_guard:n_guard + n_rb] = dft_out

            # IFFT
            td = np.fft.ifft(np.fft.ifftshift(fd)) * math.sqrt(n_sc)

            # Add cyclic prefix
            cp = td[-n_cp:]
            ofdm_sym = np.concatenate([cp, td])

            start_idx = sym_idx * (n_sc + n_cp)
            end_idx = start_idx + (n_sc + n_cp)
            if end_idx > len(signal):
                end_idx = len(signal)
            signal[start_idx:end_idx] = ofdm_sym[:end_idx - start_idx]

        base = signal[:length]

        # Frequency offset: -10%
        freq_offset = -0.10 * self.cfg.FS
        t = np.arange(length) / self.cfg.FS
        base = base * np.exp(1j * 2 * np.pi * freq_offset * t)

        base = base / (np.max(np.abs(base)) + 1e-10) * 0.6

        if len(base) < length:
            base = np.concatenate([base, np.zeros(length - len(base), dtype=complex)])
        return base[:length]

    def generate_noise(self, length: int) -> np.ndarray:
        """Generate complex AWGN."""
        return (self.rng.randn(length) + 1j * self.rng.randn(length)) / math.sqrt(2)

    def generate(self, signal_type: str, length: int) -> np.ndarray:
        """Generate a waveform of the specified signal type."""
        if signal_type == "LTE":
            return self.generate_lte(length)
        elif signal_type == "WiFi":
            return self.generate_wifi(length)
        elif signal_type == "V2X-PC5":
            return self.generate_v2x_pc5(length)
        elif signal_type == "Noise":
            return self.generate_noise(length)
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")


# ===================================================================
# SECTION 4 – CHANNEL APPLICATION
# ===================================================================
class ChannelApplicator:
    """
    Apply realistic V2X channel effects to baseband waveforms.

    Effects applied (in order):
      1. Free-space / pathloss attenuation
      2. Rician fading (time-varying K-factor, sample-by-sample)
      3. Doppler spread (time-varying phase rotation)
      4. Multipath (filtered delay taps with exponential PDP)
      5. AWGN at the target SNR
    """

    def __init__(self, config: Config):
        self.cfg = config

    def apply_pathloss(self, signal: np.ndarray, pathloss_db: float) -> np.ndarray:
        """Scale signal by pathloss attenuation."""
        gain = 10.0 ** (-pathloss_db / 20.0)
        return signal * gain

    def apply_rician_fading(
        self, signal: np.ndarray, k_db: float, doppler_hz: float, seed: int = 0
    ) -> np.ndarray:
        """
        Apply Rician fading with time-varying K-factor.

        The Rician channel is: h = sqrt(K/(K+1)) * LOS + sqrt(1/(K+1)) * NLOS
        where NLOS is Rayleigh-faded.
        """
        n = len(signal)
        k_linear = db_to_linear(k_db)
        rng = np.random.RandomState(seed)

        # LOS component (constant with slow Doppler-induced phase drift)
        los = np.ones(n, dtype=complex)
        if doppler_hz > 0:
            t = np.arange(n) / self.cfg.FS
            phase_drift = 2 * np.pi * doppler_hz * 0.3 * t  # reduced Doppler on LOS
            los = np.exp(1j * phase_drift)

        # NLOS component: Rayleigh fading via filtered Gaussian process
        # Generate complex Gaussian and apply Doppler spectrum filtering
        nlos_re = rng.randn(n)
        nlos_im = rng.randn(n)

        # Simple Doppler filter: moving average (approximates Jakes spectrum)
        if doppler_hz > 1.0:
            # Number of samples per coherence window (cap at signal length)
            filter_len = min(n, max(1, int(self.cfg.FS / (10 * doppler_hz))))
            kernel = np.ones(filter_len) / filter_len
            nlos_re = np.convolve(nlos_re, kernel, mode="same")[:n]
            nlos_im = np.convolve(nlos_im, kernel, mode="same")[:n]

        nlos = (nlos_re + 1j * nlos_im) / math.sqrt(2)

        # Combine LOS and NLOS
        k_factor = k_linear / (k_linear + 1)
        nlos_factor = 1.0 / math.sqrt(k_linear + 1)
        h = math.sqrt(k_factor) * los + nlos_factor * nlos

        return signal * h

    def apply_doppler_spread(
        self, signal: np.ndarray, doppler_hz: float
    ) -> np.ndarray:
        """
        Apply Doppler spread as a time-varying phase rotation.
        Models the frequency dispersion experienced in mobile channels.
        """
        if doppler_hz < 0.1:
            return signal
        n = len(signal)
        t = np.arange(n) / self.cfg.FS
        # Doppler causes a frequency shift that varies over time
        # We model this as a linearly varying frequency offset (chirp-like)
        freq_var = doppler_hz * 0.1 * np.cumsum(np.random.randn(n) * 0.01)
        freq_var = np.convolve(freq_var, np.ones(50) / 50, mode="same")  # smooth
        phase = 2 * np.pi * np.cumsum(freq_var) / self.cfg.FS
        return signal * np.exp(1j * phase)

    def apply_multipath(
        self, signal: np.ndarray,
        delays_us: np.ndarray,
        powers_db: np.ndarray,
    ) -> np.ndarray:
        """
        Apply frequency-selective fading via multipath.

        Implements a tapped-delay-line model with the given delays and powers.
        """
        n = len(signal)
        output = np.zeros(n, dtype=complex)
        powers_lin = db_to_linear(powers_db)

        for delay_us, power_lin in zip(delays_us, powers_lin):
            # Convert delay from μs to samples
            delay_samples = int(delay_us * 1e-6 * self.cfg.FS)
            if delay_samples >= n:
                delay_samples = n - 1
            if delay_samples < 0:
                delay_samples = 0

            # Each tap gets a random phase (from scattering)
            tap_phase = np.exp(1j * np.random.uniform(0, 2 * np.pi))
            gain = math.sqrt(power_lin) * tap_phase

            output[delay_samples:] += gain * signal[:n - delay_samples]

        # Normalise to preserve average power
        if np.max(np.abs(output)) > 0:
            output = output / np.std(output) * np.std(signal)

        return output

    def apply_awgn(self, signal: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
        """Add AWGN to achieve the target SNR."""
        rng = np.random.RandomState(seed)
        sig_power = np.mean(np.abs(signal) ** 2)
        if sig_power < 1e-15:
            return signal
        noise_power = sig_power / db_to_linear(snr_db)
        noise = rng.randn(len(signal)) + 1j * rng.randn(len(signal))
        noise = noise * math.sqrt(noise_power / 2)
        return signal + noise

    def apply_channel(
        self,
        signal: np.ndarray,
        params: ChannelParams,
        seed: int = 0,
    ) -> np.ndarray:
        """Apply the full channel model to a signal."""
        # Ensure correct dtype
        sig = np.asarray(signal, dtype=np.complex128)
        # 1. Multipath (frequency-selective fading)
        sig = self.apply_multipath(sig, params.multipath_delays_us, params.multipath_powers_db)
        # 2. Rician fading (small-scale)
        sig = self.apply_rician_fading(sig, params.rician_k_db, params.doppler_hz, seed=seed)
        # 3. Doppler spread
        sig = self.apply_doppler_spread(sig, params.doppler_hz)
        # 4. Pathloss attenuation (large-scale)
        sig = self.apply_pathloss(sig, params.pathloss_db)
        # 5. AWGN at target SNR
        sig = self.apply_awgn(sig, params.snr_db, seed=seed + 100)
        # Final safety: replace any NaN/Inf and enforce dtype
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        sig = np.asarray(sig, dtype=np.complex128)
        return sig


# ===================================================================
# SECTION 5 – SPECTROGRAM COMPUTATION
# ===================================================================
class SpectrogramComputer:
    """
    Compute dual-stream spectrograms (log-magnitude + instantaneous frequency).

    Uses STFT with configurable NFFT and hop size. Produces normalised
    spectrograms ready for ML model input.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.window = np.hanning(config.NFFT)

    def stft(self, signal: np.ndarray) -> np.ndarray:
        """Short-Time Fourier Transform.

        Uses np.fft.fft for complex-valued IQ signals (np.fft.rfft is
        designed for real input only).  For complex input the full N-point
        spectrum is returned; the lower N//2+1 bins cover the positive
        frequencies which are the most informative for spectrogram ML.
        """
        signal = np.asarray(signal, dtype=np.complex128)
        n = len(signal)
        nfft = self.cfg.NFFT
        hop = self.cfg.HOP
        win = self.window.astype(np.complex128)

        n_frames = max(1, (n - nfft) // hop + 1)
        # Use full FFT (complex input → full spectrum), take positive freq bins
        n_freq = nfft // 2 + 1
        spectrum = np.zeros((n_freq, n_frames), dtype=complex)

        for i in range(n_frames):
            start = i * hop
            end = start + nfft
            if end > n:
                break
            frame = signal[start:end] * win
            full_fft = np.fft.fft(frame)
            spectrum[:, i] = full_fft[:n_freq]

        return spectrum

    def log_magnitude(self, spectrum: np.ndarray) -> np.ndarray:
        """Compute log-magnitude spectrogram."""
        mag = np.abs(spectrum)
        mag = np.maximum(mag, 1e-10)
        return 20.0 * np.log10(mag)

    def instantaneous_frequency(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous frequency via phase differences.

        Inst. freq = Δφ / Δt  (normalised to [-π, π] per frame).
        """
        phase = np.angle(spectrum)
        # Phase difference along frequency axis (vertical)
        dphi = np.diff(phase, axis=0)
        # Unwrap
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        # Pad to original size
        dphi = np.concatenate([dphi, dphi[-1:]], axis=0)
        return dphi

    def z_score_normalise(self, x: np.ndarray) -> np.ndarray:
        """Z-score normalise along both axes."""
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma < 1e-10:
            return x - mu
        return (x - mu) / sigma

    def compute(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dual-stream spectrogram.

        Returns
        -------
        log_mag : np.ndarray  – Z-score normalised log-magnitude spectrogram
        inst_freq : np.ndarray – Z-score normalised instantaneous frequency
        """
        spectrum = self.stft(signal)
        log_mag = self.log_magnitude(spectrum)
        inst_freq = self.instantaneous_frequency(spectrum)

        log_mag = self.z_score_normalise(log_mag)
        inst_freq = self.z_score_normalise(inst_freq)

        return log_mag, inst_freq


# ===================================================================
# SECTION 6 – DATASET OUTPUT (HDF5, plots, statistics)
# ===================================================================
class DatasetWriter:
    """Save the generated dataset to HDF5 and create visualisation plots."""

    def __init__(self, config: Config):
        self.cfg = config
        self.out_dir = ensure_dir(config.OUTPUT_DIR)
        self.rng = np.random.RandomState(config.SEED + 5)

    def save_hdf5(
        self,
        log_mags: np.ndarray,
        inst_freqs: np.ndarray,
        labels: np.ndarray,
        metadata: List[dict],
    ) -> str:
        """Save dataset to HDF5 file with rich metadata."""
        fpath = str(self.out_dir / "dataset.h5")

        if not HAS_H5PY:
            # Fallback: save as numpy npz
            np.savez_compressed(
                fpath.replace(".h5", ".npz"),
                log_magnitude=log_mags,
                instantaneous_frequency=inst_freqs,
                labels=labels,
            )
            print(f"  [INFO] h5py not available; saved as {fpath.replace('.h5', '.npz')}")
            return fpath.replace(".h5", ".npz")

        with h5py.File(fpath, "w") as hf:
            # Spectrogram data
            hf.create_dataset("log_magnitude", data=log_mags, compression="gzip",
                              compression_opts=9)
            hf.create_dataset("instantaneous_frequency", data=inst_freqs,
                              compression="gzip", compression_opts=9)
            hf.create_dataset("labels", data=labels, compression="gzip")

            # Scalar metadata
            meta_group = hf.create_group("metadata")
            meta_group.attrs["num_samples"] = self.cfg.NUM_SAMPLES
            meta_group.attrs["num_classes"] = self.cfg.NUM_CLASSES
            meta_group.attrs["class_names"] = json.dumps(self.cfg.CLASS_NAMES)
            meta_group.attrs["sample_length"] = self.cfg.SAMPLE_LENGTH
            meta_group.attrs["fs_hz"] = self.cfg.FS
            meta_group.attrs["nfft"] = self.cfg.NFFT
            meta_group.attrs["hop"] = self.cfg.HOP
            meta_group.attrs["v2x_fc_hz"] = self.cfg.V2X_FC
            meta_group.attrs["seed"] = self.cfg.SEED

            # Per-sample metadata
            if metadata:
                dt = h5py.special_dtype(vlen=str)
                meta_ds = meta_group.create_dataset("per_sample", (len(metadata),), dtype=dt)
                for i, m in enumerate(metadata):
                    meta_ds[i] = json.dumps(m)

        print(f"  [OK] Saved HDF5 dataset: {fpath}")
        return fpath

    def _plot_scenario_distribution(self, labels: np.ndarray, metadata: List[dict]):
        """Bar chart of samples per scenario."""
        if not HAS_MPL:
            return
        scenarios = [m["scenario"] for m in metadata]
        classes = [self.cfg.CLASS_NAMES[l] for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Per scenario
        scenario_counts = {}
        for s in set(scenarios):
            scenario_counts[s] = scenarios.count(s)
        ax = axes[0]
        bars = ax.bar(scenario_counts.keys(), scenario_counts.values(),
                      color=["#e74c3c", "#2ecc71", "#f39c12"])
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Samples per Scenario")
        for bar, count in zip(bars, scenario_counts.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    str(count), ha="center", va="bottom", fontweight="bold")

        # Per class
        class_counts = {}
        for c in set(classes):
            class_counts[c] = classes.count(c)
        ax = axes[1]
        bars = ax.bar(class_counts.keys(), class_counts.values(),
                      color=["#3498db", "#e74c3c", "#9b59b6", "#95a5a6"])
        ax.set_xlabel("Signal Class")
        ax.set_ylabel("Number of Samples")
        ax.set_title("Samples per Signal Class")
        for bar, count in zip(bars, class_counts.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                    str(count), ha="center", va="bottom", fontweight="bold")

        plt.tight_layout()
        plt.savefig(str(self.out_dir / "scenario_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_snr_distribution(self, metadata: List[dict]):
        """Histogram of SNR values."""
        if not HAS_MPL:
            return
        snrs = [m["snr_db"] for m in metadata]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(snrs, bins=50, color="#3498db", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(snrs), color="#e74c3c", linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(snrs):.1f} dB")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Count")
        ax.set_title("SNR Distribution Across Dataset")
        ax.legend()

        plt.tight_layout()
        plt.savefig(str(self.out_dir / "snr_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_doppler_distribution(self, metadata: List[dict]):
        """Histogram of Doppler shift values."""
        if not HAS_MPL:
            return
        dopplers = [m["doppler_hz"] for m in metadata]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(dopplers, bins=50, color="#e67e22", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(dopplers), color="#e74c3c", linestyle="--", linewidth=2,
                    label=f"Mean = {np.mean(dopplers):.1f} Hz")
        ax.set_xlabel("Doppler Shift (Hz)")
        ax.set_ylabel("Count")
        ax.set_title("Doppler Shift Distribution Across Dataset")
        ax.legend()

        plt.tight_layout()
        plt.savefig(str(self.out_dir / "doppler_distribution.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_sample_spectrograms(
        self,
        log_mags: np.ndarray,
        inst_freqs: np.ndarray,
        labels: np.ndarray,
        metadata: List[dict],
    ):
        """Plot example spectrograms from each class/scenario combination."""
        if not HAS_MPL:
            return

        # Pick one example per class-scenario combination
        seen = set()
        examples = []
        for i, m in enumerate(metadata):
            key = (self.cfg.CLASS_NAMES[labels[i]], m["scenario"])
            if key not in seen:
                seen.add(key)
                examples.append((i, key))
            if len(seen) == 12:  # 4 classes × 3 scenarios
                break

        n_examples = len(examples)
        if n_examples == 0:
            return

        fig, axes = plt.subplots(2, n_examples, figsize=(3.5 * n_examples, 7))

        if n_examples == 1:
            axes = axes.reshape(2, 1)

        for col, (idx, (cls, scenario)) in enumerate(examples):
            # Log-magnitude
            ax = axes[0, col]
            im = ax.imshow(log_mags[idx], aspect="auto", cmap="viridis",
                           origin="lower", vmin=-3, vmax=3)
            ax.set_title(f"{cls}\n{scenario}", fontsize=9)
            ax.set_ylabel("Freq bin" if col == 0 else "")
            ax.tick_params(labelsize=7)

            # Instantaneous frequency
            ax = axes[1, col]
            ax.imshow(inst_freqs[idx], aspect="auto", cmap="plasma",
                      origin="lower", vmin=-3, vmax=3)
            ax.set_xlabel("Time frame" if col == n_examples // 2 else "")
            ax.set_ylabel("Freq bin" if col == 0 else "")
            ax.tick_params(labelsize=7)

        axes[0, 0].set_ylabel("Log-Magnitude\nFreq bin", fontsize=9)
        axes[1, 0].set_ylabel("Inst. Frequency\nFreq bin", fontsize=9)

        plt.suptitle("Sample Spectrograms by Class and Scenario", fontsize=13, y=1.02)
        plt.tight_layout()
        plt.savefig(str(self.out_dir / "sample_spectrograms.png"), dpi=150, bbox_inches="tight")
        plt.close()

    def save_channel_stats(self, metadata: List[dict]) -> str:
        """Compute and save channel parameter statistics as JSON."""
        stats = {
            "total_samples": len(metadata),
            "per_class": {},
            "per_scenario": {},
            "global": {},
        }

        # Collect arrays
        snrs = [m["snr_db"] for m in metadata]
        dopplers = [m["doppler_hz"] for m in metadata]
        distances = [m["distance_m"] for m in metadata]
        speeds = [m["speed_kmh"] for m in metadata]
        rician_ks = [m["rician_k_db"] for m in metadata]

        def _desc(arr):
            a = np.array(arr)
            return {
                "mean": float(np.mean(a)),
                "std": float(np.std(a)),
                "min": float(np.min(a)),
                "max": float(np.max(a)),
                "median": float(np.median(a)),
            }

        stats["global"] = {
            "snr_db": _desc(snrs),
            "doppler_hz": _desc(dopplers),
            "distance_m": _desc(distances),
            "speed_kmh": _desc(speeds),
            "rician_k_db": _desc(rician_ks),
        }

        # Per class
        for cls_idx, cls_name in enumerate(self.cfg.CLASS_NAMES):
            cls_meta = [m for m, lbl in zip(metadata,
                         [m2["label_idx"] for m2 in metadata]) if lbl == cls_idx]
            if not cls_meta:
                continue
            stats["per_class"][cls_name] = {
                "count": len(cls_meta),
                "snr_db": _desc([m["snr_db"] for m in cls_meta]),
                "doppler_hz": _desc([m["doppler_hz"] for m in cls_meta]),
                "distance_m": _desc([m["distance_m"] for m in cls_meta]),
            }

        # Per scenario
        for scenario in self.cfg.SCENARIOS:
            scen_meta = [m for m in metadata if m["scenario"] == scenario]
            if not scen_meta:
                continue
            stats["per_scenario"][scenario] = {
                "count": len(scen_meta),
                "snr_db": _desc([m["snr_db"] for m in scen_meta]),
                "doppler_hz": _desc([m["doppler_hz"] for m in scen_meta]),
                "distance_m": _desc([m["distance_m"] for m in scen_meta]),
                "speed_kmh": _desc([m["speed_kmh"] for m in scen_meta]),
                "rician_k_db": _desc([m["rician_k_db"] for m in scen_meta]),
            }

        fpath = str(self.out_dir / "channel_stats.json")
        with open(fpath, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  [OK] Channel stats: {fpath}")
        return fpath

    def generate_all_plots(
        self,
        log_mags: np.ndarray,
        inst_freqs: np.ndarray,
        labels: np.ndarray,
        metadata: List[dict],
    ):
        """Generate all summary plots."""
        if not HAS_MPL:
            print("  [WARN] matplotlib not available; skipping plots.")
            return
        print("  Generating plots...")
        self._plot_scenario_distribution(labels, metadata)
        self._plot_snr_distribution(metadata)
        self._plot_doppler_distribution(metadata)
        self._plot_sample_spectrograms(log_mags, inst_freqs, labels, metadata)
        print("  [OK] All plots saved.")


# ===================================================================
# SECTION 7 – MAIN PIPELINE (Mode 1: Simulation-Only)
# ===================================================================
class SimulationPipeline:
    """
    Orchestrates the entire simulation-only dataset generation pipeline.

    Steps:
      1. Generate mobility trajectories
      2. Extract channel parameters
      3. Generate waveforms
      4. Apply channel effects
      5. Compute spectrograms
      6. Save dataset and visualisations
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.mobility = MobilitySimulator(config)
        self.channel_model = ChannelModel(config)
        self.waveform_gen = WaveformGenerator(config)
        self.channel_app = ChannelApplicator(config)
        self.spectrogram = SpectrogramComputer(config)
        self.writer = DatasetWriter(config)

    def _sample_scenario_for_class(self, class_idx: int, sample_idx: int) -> str:
        """Distribute samples across scenarios, biased by class."""
        scenarios = self.cfg.SCENARIOS
        # Round-robin with jitter for diversity
        base = scenarios[sample_idx % len(scenarios)]
        # Some classes are more common in certain scenarios
        rng = np.random.RandomState(self.cfg.SEED + class_idx * 1000 + sample_idx)
        if self.cfg.CLASS_NAMES[class_idx] == "V2X-PC5":
            # V2X is more common on highway and rural
            weights = [0.5, 0.2, 0.3]
        elif self.cfg.CLASS_NAMES[class_idx] == "WiFi":
            # WiFi is more common in urban
            weights = [0.2, 0.6, 0.2]
        else:
            weights = [1/3, 1/3, 1/3]
        return rng.choice(scenarios, p=weights)

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Run the full simulation pipeline."""
        print("=" * 70)
        print("  V2X Spectrum Sensing Dataset Generation Pipeline")
        print(f"  Mode: Simulation-Only")
        print(f"  Total samples: {self.cfg.NUM_SAMPLES}")
        print(f"  Classes: {self.cfg.CLASS_NAMES}")
        print(f"  Scenarios: {self.cfg.SCENARIOS}")
        print(f"  Output dir: {self.cfg.OUTPUT_DIR}")
        print("=" * 70)

        # Pre-allocate arrays
        # Spectrogram shape: (n_fft//2 + 1, time_frames)
        n_freq = self.cfg.NFFT // 2 + 1
        n_frames = max(1, (self.cfg.SAMPLE_LENGTH - self.cfg.NFFT) // self.cfg.HOP + 1)

        log_mags = np.zeros((self.cfg.NUM_SAMPLES, n_freq, n_frames), dtype=np.float32)
        inst_freqs = np.zeros((self.cfg.NUM_SAMPLES, n_freq, n_frames), dtype=np.float32)
        labels = np.zeros(self.cfg.NUM_SAMPLES, dtype=np.int64)
        metadata: List[dict] = []

        set_seed(self.cfg.SEED)
        t_start = time.time()

        sample_idx = 0
        for class_idx, class_name in enumerate(self.cfg.CLASS_NAMES):
            n_class = self.cfg.SAMPLES_PER_CLASS
            print(f"\n  Generating class [{class_idx+1}/{self.cfg.NUM_CLASSES}]: {class_name} "
                  f"({n_class} samples)")

            for j in range(n_class):
                progress_bar(j + 1, n_class, prefix=f"    ")

                # 1. Choose scenario
                scenario = self._sample_scenario_for_class(class_idx, sample_idx)

                # 2. Generate mobility snapshot
                mob = self.mobility.generate_vehicle_pair_snapshots(scenario, n_snapshots=1)[0]

                # 3. Extract channel parameters
                ch = self.channel_model.extract_channel_params(mob, scenario)

                # 4. Generate baseband waveform
                waveform = self.waveform_gen.generate(class_name, self.cfg.SAMPLE_LENGTH)

                # 5. Apply channel effects
                if class_name != "Noise":
                    waveform = self.channel_app.apply_channel(
                        waveform, ch, seed=self.cfg.SEED + sample_idx
                    )

                # 6. Compute spectrogram
                lm, inf = self.spectrogram.compute(waveform)

                # Handle varying frame counts (pad/truncate to fixed size)
                actual_frames = lm.shape[1]
                if actual_frames < n_frames:
                    lm = np.pad(lm, ((0, 0), (0, n_frames - actual_frames)), mode="constant")
                    inf = np.pad(inf, ((0, 0), (0, n_frames - actual_frames)), mode="constant")
                elif actual_frames > n_frames:
                    lm = lm[:, :n_frames]
                    inf = inf[:, :n_frames]

                log_mags[sample_idx] = lm.astype(np.float32)
                inst_freqs[sample_idx] = inf.astype(np.float32)
                labels[sample_idx] = class_idx

                # Build metadata
                meta = {
                    "signal_type": class_name,
                    "label_idx": int(class_idx),
                    "scenario": scenario,
                    "speed_kmh": round(mob.tx_speed_ms * 3.6, 2),
                    "distance_m": round(mob.distance_m, 2),
                    "snr_db": round(ch.snr_db, 2),
                    "rician_k_db": round(ch.rician_k_db, 2),
                    "doppler_hz": round(ch.doppler_hz, 2),
                    "pathloss_db": round(ch.pathloss_db, 2),
                    "shadowing_db": round(ch.shadowing_db, 2),
                    "coherence_time_ms": round(ch.coherence_time_ms, 2),
                    "relative_speed_ms": round(mob.relative_speed_ms, 2),
                    "angle_of_arrival_deg": round(math.degrees(mob.angle_of_arrival_rad), 2),
                }
                metadata.append(meta)

                sample_idx += 1

        elapsed = time.time() - t_start
        print(f"\n  Generation completed in {elapsed:.1f}s "
              f"({self.cfg.NUM_SAMPLES / elapsed:.1f} samples/s)")

        # Save everything
        print("\n  Saving dataset...")
        self.writer.save_hdf5(log_mags, inst_freqs, labels, metadata)
        self.writer.save_channel_stats(metadata)
        self.writer.generate_all_plots(log_mags, inst_freqs, labels, metadata)

        print(f"\n  Dataset summary:")
        print(f"    Log-magnitude shape:    {log_mags.shape}")
        print(f"    Inst-frequency shape:   {inst_freqs.shape}")
        print(f"    Labels shape:           {labels.shape}")
        print(f"    Class distribution:     {dict(zip(*np.unique(labels, return_counts=True)))}")
        print(f"    Scenario distribution:  {dict(zip(*np.unique([m['scenario'] for m in metadata], return_counts=True)))}")

        return log_mags, inst_freqs, labels, metadata


# ===================================================================
# SECTION 8 – MODE 2: SUMO + GNU RADIO INTEGRATION
# ===================================================================
class SumoScenarioGenerator:
    """
    Generate SUMO configuration files for three V2X scenarios.

    Creates:
      - .net.xml  (road network definition)
      - .rou.xml  (vehicle routing)
      - .sumocfg  (simulation configuration)
      - .add.xml  (FCD output configuration)
    """

    TEMPLATES = {
        "highway": {
            "net_xml": """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by V2X Dataset Pipeline – Highway Scenario -->
<net version="1.9" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,{road_length},11.10" origBoundary="0.00,0.00,{road_length},11.10" projParameter="!">
    </location>
    <edge id="E0" from="N0" to="N1" priority="2" numLanes="3" speed="{speed_max}" length="{road_length}" shape="0.00,3.70 {road_length},3.70">
        <lane id="E0_0" index="0" speed="{speed_max}" length="{road_length}" shape="0.00,1.85 {road_length},1.85"/>
        <lane id="E0_1" index="1" speed="{speed_max}" length="{road_length}" shape="0.00,5.55 {road_length},5.55"/>
        <lane id="E0_2" index="2" speed="{speed_max}" length="{road_length}" shape="0.00,9.25 {road_length},9.25"/>
    </edge>
    <junction id="N0" type="dead_end" x="0.00" y="3.70" incLanes="" intLanes="" shape="0.00,0.00 0.00,7.40"/>
    <junction id="N1" type="dead_end" x="{road_length}" y="3.70" incLanes="E0_0 E0_1 E0_2" intLanes="" shape="{road_length},0.00 {road_length},7.40"/>
</net>""",
            "rou_xml": """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by V2X Dataset Pipeline – Highway Routes -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car_highway" vClass="passenger" maxSpeed="{speed_max}" length="5.0" minGap="2.5" accel="2.6" decel="4.5" sigma="0.5" color="1,1,0"/>
    <flow id="flow_hw" type="car_highway" route="route_hw" begin="0" end="{sim_end}" vehsPerHour="{flow_rate}" departLane="best" departSpeed="{speed_min},{speed_max}"/>
    <route id="route_hw" edges="E0"/>
</routes>""",
            "add_xml": """<?xml version="1.0" encoding="UTF-8"?>
<add xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <fcd-export id="fcd_hw" file="fcd_highway.xml" freq="0.1" begin="0" end="{sim_end}"/>
</add>""",
        },
        "urban": {
            "net_xml": """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by V2X Dataset Pipeline – Urban Intersection Scenario -->
<net version="1.9" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="-500.00,-500.00,1500.00,1500.00" origBoundary="-500.00,-500.00,1500.00,1500.00" projParameter="!">
    </location>
    <edge id="E_W" from="W" to="E_node" priority="2" numLanes="2" speed="16.67" length="1000">
        <lane id="E_W_0" index="0" speed="16.67" length="1000" shape="-500.00,498.15 500.00,498.15"/>
        <lane id="E_W_1" index="1" speed="16.67" length="1000" shape="-500.00,501.85 500.00,501.85"/>
    </edge>
    <edge id="E_E" from="E_node" to="E" priority="2" numLanes="2" speed="16.67" length="1000">
        <lane id="E_E_0" index="0" speed="16.67" length="1000" shape="500.00,498.15 1500.00,498.15"/>
        <lane id="E_E_1" index="1" speed="16.67" length="1000" shape="500.00,501.85 1500.00,501.85"/>
    </edge>
    <edge id="E_N" from="S" to="N_node" priority="2" numLanes="2" speed="16.67" length="1000">
        <lane id="E_N_0" index="0" speed="16.67" length="1000" shape="501.85,-500.00 501.85,500.00"/>
        <lane id="E_N_1" index="1" speed="16.67" length="1000" shape="498.15,-500.00 498.15,500.00"/>
    </edge>
    <edge id="E_S" from="N_node" to="N" priority="2" numLanes="2" speed="16.67" length="1000">
        <lane id="E_S_0" index="0" speed="16.67" length="1000" shape="501.85,500.00 501.85,1500.00"/>
        <lane id="E_S_1" index="1" speed="16.67" length="1000" shape="498.15,500.00 498.15,1500.00"/>
    </edge>
    <junction id="W" type="dead_end" x="-500.00" y="500.00" incLanes="" intLanes="" shape="-500.00,496.30 -500.00,503.70"/>
    <junction id="E" type="dead_end" x="1500.00" y="500.00" incLanes="E_E_0 E_E_1" intLanes="" shape="1500.00,496.30 1500.00,503.70"/>
    <junction id="S" type="dead_end" x="500.00" y="-500.00" incLanes="" intLanes="" shape="496.30,-500.00 503.70,-500.00"/>
    <junction id="N" type="dead_end" x="500.00" y="1500.00" incLanes="E_S_0 E_S_1" intLanes="" shape="496.30,1500.00 503.70,1500.00"/>
    <junction id="E_node" type="priority" x="500.00" y="500.00" incLanes="E_W_0 E_W_1 E_N_0 E_N_1" intLanes=":E_node_0_0 :E_node_1_0 :E_node_2_0 :E_node_3_0 :E_node_4_0 :E_node_5_0 :E_node_6_0 :E_node_7_0" shape="496.30,496.30 503.70,496.30 503.70,503.70 496.30,503.70">
        <request index="0" response="00000000" foes="00010000" cont="0"/>
        <request index="1" response="11000000" foes="11011100" cont="0"/>
        <request index="2" response="00000010" foes="00000010" cont="0"/>
        <request index="3" response="00000110" foes="00100110" cont="0"/>
        <request index="4" response="00000000" foes="00000000" cont="0"/>
        <request index="5" response="00000000" foes="00000000" cont="0"/>
        <request index="6" response="00110000" foes="00110000" cont="0"/>
        <request index="7" response="00110000" foes="00110000" cont="0"/>
    </junction>
    <connection from="E_W" to="E_E" fromLane="0" toLane="0" via=":E_node_0_0" dir="s" state="M"/>
    <connection from="E_W" to="E_E" fromLane="1" toLane="1" via=":E_node_1_0" dir="s" state="M"/>
    <connection from="E_N" to="E_S" fromLane="0" toLane="0" via=":E_node_4_0" dir="s" state="M"/>
    <connection from="E_N" to="E_S" fromLane="1" toLane="1" via=":E_node_5_0" dir="s" state="M"/>
    <connection from="E_W" to="E_N" fromLane="0" toLane="0" via=":E_node_2_0" dir="l" state="m"/>
    <connection from="E_W" to="E_N" fromLane="1" toLane="1" via=":E_node_3_0" dir="l" state="m"/>
    <connection from="E_N" to="E_E" fromLane="0" toLane="0" via=":E_node_6_0" dir="r" state="m"/>
    <connection from="E_N" to="E_E" fromLane="1" toLane="1" via=":E_node_7_0" dir="r" state="m"/>
    <tlLogic id="E_node" type="static" programID="0" offset="0">
        <phase duration="30" state="GGggrrrrGGggrrrr"/>
        <phase duration="5"  state="yyggrrrryyggrrrr"/>
        <phase duration="30" state="rrrrGGggrrrrGGgg"/>
        <phase duration="5"  state="rrrryyggrrrryygg"/>
    </tlLogic>
</net>""",
            "rou_xml": """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by V2X Dataset Pipeline – Urban Routes -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car_urban" vClass="passenger" maxSpeed="16.67" length="5.0" minGap="2.0" accel="2.0" decel="4.0" sigma="0.8" color="0,1,0"/>
    <flow id="flow_ew" type="car_urban" route="route_ew" begin="0" end="{sim_end}" vehsPerHour="{flow_rate}" departLane="best" departSpeed="0,16.67"/>
    <flow id="flow_ns" type="car_urban" route="route_ns" begin="0" end="{sim_end}" vehsPerHour="{flow_rate}" departLane="best" departSpeed="0,16.67"/>
    <route id="route_ew" edges="E_W E_E"/>
    <route id="route_ns" edges="E_N E_S"/>
</routes>""",
            "add_xml": """<?xml version="1.0" encoding="UTF-8"?>
<add xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <fcd-export id="fcd_urban" file="fcd_urban.xml" freq="0.1" begin="0" end="{sim_end}"/>
</add>""",
        },
        "rural": {
            "net_xml": """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by V2X Dataset Pipeline – Rural Road Scenario -->
<net version="1.9" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,{road_length},7.40" origBoundary="0.00,0.00,{road_length},7.40" projParameter="!">
    </location>
    <edge id="E0" from="N0" to="N1" priority="1" numLanes="2" speed="{speed_max}" length="{road_length}" shape="0.00,3.70 {road_length},3.70">
        <lane id="E0_0" index="0" speed="{speed_max}" length="{road_length}" shape="0.00,1.85 {road_length},1.85"/>
        <lane id="E0_1" index="1" speed="{speed_max}" length="{road_length}" shape="0.00,5.55 {road_length},5.55"/>
    </edge>
    <junction id="N0" type="dead_end" x="0.00" y="3.70" incLanes="" intLanes="" shape="0.00,0.00 0.00,7.40"/>
    <junction id="N1" type="dead_end" x="{road_length}" y="3.70" incLanes="E0_0 E0_1" intLanes="" shape="{road_length},0.00 {road_length},7.40"/>
</net>""",
            "rou_xml": """<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated by V2X Dataset Pipeline – Rural Routes -->
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="car_rural" vClass="passenger" maxSpeed="{speed_max}" length="5.0" minGap="3.0" accel="2.0" decel="3.5" sigma="0.6" color="0,0,1"/>
    <flow id="flow_rural" type="car_rural" route="route_rural" begin="0" end="{sim_end}" vehsPerHour="{flow_rate}" departLane="best" departSpeed="{speed_min},{speed_max}"/>
    <route id="route_rural" edges="E0"/>
</routes>""",
            "add_xml": """<?xml version="1.0" encoding="UTF-8"?>
<add xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/additional_file.xsd">
    <fcd-export id="fcd_rural" file="fcd_rural.xml" freq="0.1" begin="0" end="{sim_end}"/>
</add>""",
        },
    }

    SCENARIO_CONFIG = {
        "highway": {
            "road_length": "5000",
            "speed_max": "33.33",   # 120 km/h
            "speed_min": "22.22",   # 80 km/h
            "flow_rate": "1200",
            "sim_end": "600",       # 10 minutes
        },
        "urban": {
            "road_length": "1000",
            "speed_max": "16.67",   # 60 km/h
            "speed_min": "0.00",
            "flow_rate": "800",
            "sim_end": "600",
        },
        "rural": {
            "road_length": "3000",
            "speed_max": "22.22",   # 80 km/h
            "speed_min": "11.11",   # 40 km/h
            "flow_rate": "400",
            "sim_end": "600",
        },
    }

    def __init__(self, config: Config):
        self.cfg = config
        self.scenario_dir = ensure_dir(str(Path(config.OUTPUT_DIR) / "sumo_scenarios"))

    def generate(self):
        """Generate SUMO configuration files for all scenarios."""
        print("\n" + "=" * 70)
        print("  Mode 2: SUMO Scenario Generation")
        print("=" * 70)

        for scenario in self.cfg.SCENARIOS:
            print(f"\n  Generating SUMO configs for: {scenario}")
            sdir = ensure_dir(str(self.scenario_dir / scenario))
            scfg = self.SCENARIO_CONFIG[scenario]
            tmpl = self.TEMPLATES[scenario]

            # .net.xml
            net_content = tmpl["net_xml"].format(**scfg)
            net_path = sdir / f"{scenario}.net.xml"
            with open(net_path, "w") as f:
                f.write(net_content)
            print(f"    → {net_path}")

            # .rou.xml
            rou_content = tmpl["rou_xml"].format(**scfg)
            rou_path = sdir / f"{scenario}.rou.xml"
            with open(rou_path, "w") as f:
                f.write(rou_content)
            print(f"    → {rou_path}")

            # .add.xml (FCD output)
            add_content = tmpl["add_xml"].format(**scfg)
            add_path = sdir / f"{scenario}.add.xml"
            with open(add_path, "w") as f:
                f.write(add_content)
            print(f"    → {add_path}")

            # .sumocfg
            sumocfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{scenario}.net.xml"/>
        <route-files value="{scenario}.rou.xml"/>
        <additional-files value="{scenario}.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{scfg['sim_end']}"/>
        <step-length value="0.1"/>
    </time>
    <processing>
        <time-to-teleport value="-1"/>
        <collision.action value="warn"/>
    </processing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
    </report>
</configuration>"""
            cfg_path = sdir / f"{scenario}.sumocfg"
            with open(cfg_path, "w") as f:
                f.write(sumocfg_content)
            print(f"    → {cfg_path}")

        print("\n  [OK] All SUMO configuration files generated.")
        print(f"  To run:  sumo -c <scenario>.sumocfg  (in each scenario directory)")


class FcdParser:
    """
    Parse SUMO FCD (Floating Car Data) output into mobility snapshots.

    Expects the standard SUMO FCD XML format:
      <timestep time="...">
        <vehicle id="..." x="..." y="..." speed="..." angle="..." lane="..."/>
      </timestep>
    """

    def __init__(self, config: Config):
        self.cfg = config

    def parse_fcd_file(self, fcd_path: str) -> List[MobilitySnapshot]:
        """
        Parse a single FCD XML file and return mobility snapshots
        for all vehicle pairs at each timestep.
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(fcd_path)
        root = tree.getroot()

        # Group vehicle states by timestep
        timesteps: Dict[float, List[VehicleState]] = {}
        for ts_elem in root.findall("timestep"):
            ts_time = float(ts_elem.get("time", "0"))
            vehicles = []
            for veh_elem in ts_elem.findall("vehicle"):
                x = float(veh_elem.get("x", "0"))
                y = float(veh_elem.get("y", "0"))
                speed = float(veh_elem.get("speed", "0"))
                angle_deg = float(veh_elem.get("angle", "0"))
                lane = int(veh_elem.get("lane", "0").split("_")[0].replace("E", "").replace("W", ""))
                vehicles.append(VehicleState(
                    x=x, y=y,
                    speed_ms=speed,
                    heading_rad=math.radians(angle_deg),
                    lane=lane,
                ))
            timesteps[ts_time] = vehicles

        # Compute pairwise mobility snapshots
        snapshots: List[MobilitySnapshot] = []
        for ts_time, vehicles in timesteps.items():
            for i in range(len(vehicles)):
                for j in range(i + 1, len(vehicles)):
                    dx = vehicles[i].x - vehicles[j].x
                    dy = vehicles[i].y - vehicles[j].y
                    distance = math.sqrt(dx ** 2 + dy ** 2) + 1e-3
                    aoa = math.atan2(dy, dx)

                    # Relative speed
                    vi_x = vehicles[i].speed_ms * math.cos(vehicles[i].heading_rad)
                    vi_y = vehicles[i].speed_ms * math.sin(vehicles[i].heading_rad)
                    vj_x = vehicles[j].speed_ms * math.cos(vehicles[j].heading_rad)
                    vj_y = vehicles[j].speed_ms * math.sin(vehicles[j].heading_rad)
                    rel_vx = vi_x - vj_x
                    rel_vy = vi_y - vj_y
                    unit_x = dx / distance
                    unit_y = dy / distance
                    rel_speed = rel_vx * unit_x + rel_vy * unit_y

                    snapshots.append(MobilitySnapshot(
                        timestep=ts_time,
                        distance_m=distance,
                        relative_speed_ms=rel_speed,
                        angle_of_arrival_rad=aoa,
                        tx_speed_ms=vehicles[i].speed_ms,
                        rx_speed_ms=vehicles[j].speed_ms,
                    ))

        return snapshots


class GnuradioScriptGenerator:
    """
    Generate GNU Radio Python scripts for IQ data capture.

    Creates standalone Python scripts that use GNU Radio's Python API
    with gr-ieee802-11 blocks for 802.11p (V2X) waveform generation.
    These scripts can be run separately to produce .cfile IQ recordings.
    """

    TEMPLATE_WAVEFORM_GEN = '''#!/usr/bin/env python3
"""
GNU Radio IQ Waveform Generation Script
=========================================
Generated by V2X Dataset Pipeline for scenario: {scenario}
Signal type: {signal_type}

Prerequisites:
  - GNU Radio (>= 3.8)
  - gr-ieee802-11 (for 802.11p support): https://github.com/bastibl/gr-ieee802-11
  - Run: python {filename}

Output:
  - {output_cfile}  (complex64, raw IQ samples)
"""

import sys
import json
import numpy as np

# ---------------------------------------------------------------------------
# Channel parameters (from SUMO FCD + channel model)
# ---------------------------------------------------------------------------
CHANNEL_PARAMS = {channel_params_json}

def generate_ofdm_signal(num_samples, num_subcarriers, subcarrier_spacing_hz, 
                          num_symbols, cp_length, freq_offset_hz, power=0.8):
    """Generate a generic OFDM signal."""
    total_per_sym = num_subcarriers + cp_length
    signal = np.zeros(num_symbols * total_per_sym, dtype=np.complex128)
    
    n_data = int(num_subcarriers * 0.75)
    n_guard = (num_subcarriers - n_data) // 2
    window = np.hanning(num_subcarriers)
    
    for sym in range(num_symbols):
        fd = np.zeros(num_subcarriers, dtype=np.complex128)
        # QPSK data
        bits = np.random.randint(0, 4, n_data)
        const = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        fd[n_guard:n_guard+n_data] = const[bits]
        
        td = np.fft.ifft(np.fft.ifftshift(fd)) * np.sqrt(num_subcarriers)
        td *= window
        cp = td[-cp_length:]
        ofdm_sym = np.concatenate([cp, td])
        
        start = sym * total_per_sym
        end = min(start + total_per_sym, len(signal))
        signal[start:end] = ofdm_sym[:end-start]
    
    # Frequency offset
    fs = {fs}
    t = np.arange(len(signal)) / fs
    signal *= np.exp(1j * 2 * np.pi * freq_offset_hz * t)
    
    # Normalise
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * power
    
    return signal[:num_samples]


def apply_rician_channel(signal, k_db, doppler_hz, fs):
    """Apply Rician fading channel."""
    n = len(signal)
    k_lin = 10 ** (k_db / 10.0)
    
    # LOS
    t = np.arange(n) / fs
    los = np.exp(1j * 2 * np.pi * doppler_hz * 0.3 * t)
    
    # NLOS (Rayleigh)
    h_re = np.convolve(np.random.randn(n), np.ones(50)/50, mode="same")
    h_im = np.convolve(np.random.randn(n), np.ones(50)/50, mode="same")
    nlos = (h_re + 1j * h_im) / np.sqrt(2)
    
    k_f = np.sqrt(k_lin / (k_lin + 1))
    n_f = 1.0 / np.sqrt(k_lin + 1)
    h = k_f * los + n_f * nlos
    
    return signal * h


def apply_multipath(signal, delays_us, powers_db, fs):
    """Apply multipath delay line."""
    n = len(signal)
    output = np.zeros(n, dtype=np.complex128)
    powers_lin = 10 ** (np.array(powers_db) / 10.0)
    
    for d_us, p_lin in zip(delays_us, powers_lin):
        d_samp = int(d_us * 1e-6 * fs)
        d_samp = max(0, min(d_samp, n-1))
        gain = np.sqrt(p_lin) * np.exp(1j * np.random.uniform(0, 2*np.pi))
        output[d_samp:] += gain * signal[:n-d_samp]
    
    # Normalise power
    if np.std(output) > 0:
        output = output / np.std(output) * np.std(signal)
    return output


def add_awgn(signal, snr_db):
    """Add AWGN at target SNR."""
    sig_pow = np.mean(np.abs(signal)**2)
    if sig_pow < 1e-15:
        return signal
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    noise = np.random.randn(len(signal)) + 1j * np.random.randn(len(signal))
    noise *= np.sqrt(noise_pow / 2)
    return signal + noise


def main():
    params = CHANNEL_PARAMS
    fs = {fs}
    num_samples = {sample_length}
    
    print(f"Generating {{params['signal_type']}} signal for scenario {{params['scenario']}}")
    print(f"  Samples: {{num_samples}},  fs: {{fs/1e6:.1f}} MHz")
    print(f"  SNR: {{params['snr_db']:.1f}} dB,  Doppler: {{params['doppler_hz']:.1f}} Hz")
    
    # --- Generate waveform ---
    if params["signal_type"] == "LTE":
        signal = generate_ofdm_signal(
            num_samples, num_subcarriers=64, subcarrier_spacing_hz=15e3,
            num_symbols=16, cp_length=16, freq_offset_hz=0.35*fs, power=0.8
        )
    elif params["signal_type"] == "WiFi":
        signal = generate_ofdm_signal(
            num_samples, num_subcarriers=64, subcarrier_spacing_hz=312.5e3,
            num_symbols=16, cp_length=16, freq_offset_hz=0.15*fs, power=0.7
        )
    elif params["signal_type"] == "V2X-PC5":
        # SC-FDMA (DFT-spread OFDM)
        n_sc = 64; n_rb = 12; n_cp = 16; n_sym = 16
        total_per_sym = n_sc + n_cp
        sig = np.zeros(n_sym * total_per_sym, dtype=np.complex128)
        n_guard = (n_sc - n_rb) // 2
        for s in range(n_sym):
            qpsk = (np.random.randint(0, 4, n_rb).astype(np.complex128) * 2 - 1) + \
                   1j * (np.random.randint(0, 4, n_rb).astype(np.complex128) * 2 - 1)
            qpsk /= np.sqrt(2)
            dft_out = np.fft.fft(qpsk) * np.sqrt(n_rb)
            fd = np.zeros(n_sc, dtype=np.complex128)
            fd[n_guard:n_guard+n_rb] = dft_out
            td = np.fft.ifft(np.fft.ifftshift(fd)) * np.sqrt(n_sc)
            cp = td[-n_cp:]
            ofdm_sym = np.concatenate([cp, td])
            start = s * total_per_sym
            end = min(start + total_per_sym, len(sig))
            sig[start:end] = ofdm_sym[:end-start]
        sig = sig[:num_samples]
        t = np.arange(len(sig)) / fs
        sig *= np.exp(1j * 2 * np.pi * (-0.10) * fs * t)
        sig /= (np.max(np.abs(sig)) + 1e-10)
        sig *= 0.6
        signal = sig
    else:
        signal = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
    
    # Pad/truncate
    if len(signal) < num_samples:
        signal = np.concatenate([signal, np.zeros(num_samples - len(signal), dtype=np.complex128)])
    signal = signal[:num_samples]
    
    # --- Apply channel ---
    if params["signal_type"] != "Noise":
        signal = apply_multipath(signal, params["multipath_delays"], params["multipath_powers"], fs)
        signal = apply_rician_channel(signal, params["rician_k_db"], params["doppler_hz"], fs)
        signal = add_awgn(signal, params["snr_db"])
    
    # --- Save as .cfile ---
    output_path = "{output_cfile}"
    signal.astype(np.complex64).tofile(output_path)
    print(f"  Saved IQ data to {{output_path}}")
    print(f"  File size: {{os.path.getsize(output_path) / 1024:.1f}} KB")
    
    # --- Also save metadata ---
    meta_path = output_path.replace(".cfile", ".json")
    with open(meta_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"  Saved metadata to {{meta_path}}")


if __name__ == "__main__":
    import os
    main()
'''

    def __init__(self, config: Config):
        self.cfg = config
        self.script_dir = ensure_dir(str(Path(config.OUTPUT_DIR) / "gnuradio_scripts"))

    def generate_scripts(
        self, metadata: List[dict], max_scripts: int = 100
    ):
        """Generate GNU Radio Python scripts from channel metadata."""
        print("\n" + "=" * 70)
        print("  Mode 2: GNU Radio Script Generation")
        print("=" * 70)

        # Subsample if too many
        indices = np.random.choice(len(metadata), size=min(max_scripts, len(metadata)),
                                    replace=False)

        n_generated = 0
        for idx in indices:
            m = metadata[idx]

            # Prepare channel params dict
            ch_params = {
                "signal_type": m["signal_type"],
                "scenario": m["scenario"],
                "snr_db": m["snr_db"],
                "doppler_hz": m["doppler_hz"],
                "rician_k_db": m["rician_k_db"],
                "multipath_delays": [0.0, 0.3, 0.7, 1.5],
                "multipath_powers": [0, -3, -7, -12],
                "distance_m": m["distance_m"],
                "speed_kmh": m["speed_kmh"],
            }

            ch_params_json = json.dumps(ch_params, indent=4)

            filename = (f"gen_{m['scenario']}_{m['signal_type']}"
                       f"_snr{int(m['snr_db'])}_d{int(m['distance_m'])}.py")
            output_cfile = filename.replace(".py", ".cfile")
            fpath = self.script_dir / filename

            script_content = self.TEMPLATE_WAVEFORM_GEN.format(
                scenario=m["scenario"],
                signal_type=m["signal_type"],
                filename=filename,
                output_cfile=output_cfile,
                channel_params_json=ch_params_json,
                fs=self.cfg.FS,
                sample_length=self.cfg.SAMPLE_LENGTH,
            )

            with open(fpath, "w") as f:
                f.write(script_content)

            n_generated += 1

        print(f"\n  [OK] Generated {n_generated} GNU Radio scripts in: {self.script_dir}")
        print(f"  To run:  python {self.script_dir}/gen_<scenario>_<class>_snr<N>.py")
        print(f"  Note: Requires GNU Radio >= 3.8 and NumPy.")


class SumoGnuradioPipeline:
    """
    Full SUMO + GNU Radio integration pipeline (Mode 2).

    This pipeline:
      1. Generates SUMO config files for three scenarios
      2. (Optionally) runs SUMO and parses FCD data
      3. Generates GNU Radio Python scripts with channel parameters
      4. Provides post-processing code for IQ → spectrogram conversion
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.sumo_gen = SumoScenarioGenerator(config)
        self.fcd_parser = FcdParser(config)
        self.gr_gen = GnuradioScriptGenerator(config)

    def run(self, metadata: Optional[List[dict]] = None):
        """Run the full Mode 2 pipeline."""
        print("\n" + "=" * 70)
        print("  V2X Dataset Pipeline – Mode 2: SUMO + GNU Radio Integration")
        print("=" * 70)

        # Step 1: Generate SUMO configs
        self.sumo_gen.generate()

        # Step 2: Check for FCD files (if SUMO has already been run)
        fcd_dir = Path(self.cfg.OUTPUT_DIR) / "sumo_scenarios"
        fcd_found = False
        for scenario in self.cfg.SCENARIOS:
            fcd_path = fcd_dir / scenario / f"fcd_{scenario}.xml"
            if fcd_path.exists():
                print(f"\n  Found FCD data: {fcd_path}")
                snapshots = self.fcd_parser.parse_fcd_file(str(fcd_path))
                print(f"    → {len(snapshots)} vehicle-pair snapshots extracted")
                fcd_found = True

        if not fcd_found:
            print("\n  [INFO] No FCD data found. Run SUMO first:")
            print("    cd simulated_rf_dataset/sumo_scenarios/<scenario>")
            print("    sumo -c <scenario>.sumocfg")
            print("  Then re-run this pipeline to parse FCD and generate GR scripts.")

        # Step 3: Generate GNU Radio scripts
        # If no metadata from Mode 1, create a minimal set from scenario defaults
        if metadata is None:
            metadata = self._generate_default_metadata()

        self.gr_gen.generate_scripts(metadata)

        # Step 4: Generate post-processing script
        self._generate_postprocess_script()

        print("\n" + "=" * 70)
        print("  Mode 2 setup complete. Workflow:")
        print("    1. cd simulated_rf_dataset/sumo_scenarios/<scenario>")
        print("    2. sumo -c <scenario>.sumocfg          # Run SUMO")
        print("    3. cd ../../../gnuradio_scripts/")
        print("    4. python gen_<...>.py                  # Generate IQ data")
        print("    5. python postprocess_iq_to_spectrogram.py  # Convert to dataset")
        print("=" * 70)

    def _generate_default_metadata(self) -> List[dict]:
        """Generate default channel metadata for GR script creation."""
        rng = np.random.RandomState(self.cfg.SEED + 10)
        metadata = []
        for cls in self.cfg.CLASS_NAMES:
            for scenario in self.cfg.SCENARIOS:
                for _ in range(5):
                    metadata.append({
                        "signal_type": cls,
                        "scenario": scenario,
                        "snr_db": round(rng.uniform(0, 30), 2),
                        "doppler_hz": round(rng.uniform(0, 1000), 2),
                        "rician_k_db": round(rng.uniform(1, 10), 2),
                        "distance_m": round(rng.uniform(10, 500), 2),
                        "speed_kmh": round(rng.uniform(20, 120), 2),
                    })
        return metadata

    def _generate_postprocess_script(self):
        """Generate a post-processing script to convert IQ files to spectrograms."""
        script = '''#!/usr/bin/env python3
"""
Post-Processing: IQ (.cfile) → Spectrogram Dataset
====================================================
Converts GNU Radio output .cfile files to STFT spectrograms
and saves them in the same format as the simulation pipeline.

Usage:
    python postprocess_iq_to_spectrogram.py [--input-dir .] [--output dataset_post.h5]
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:
    print("ERROR: h5py is required. Install with: pip install h5py")
    sys.exit(1)


def read_cfile(path, dtype=np.complex64):
    """Read a GNU Radio .cfile (complex64 raw IQ)."""
    return np.fromfile(path, dtype=dtype)


def compute_spectrogram(signal, fs, nfft=128, hop=64):
    """Compute log-magnitude and instantaneous-frequency spectrograms."""
    window = np.hanning(nfft)
    n = len(signal)
    n_frames = max(1, (n - nfft) // hop + 1)
    n_freq = nfft // 2 + 1

    spectrum = np.zeros((n_freq, n_frames), dtype=np.complex128)
    for i in range(n_frames):
        start = i * hop
        end = start + nfft
        if end > n:
            break
        frame = signal[start:end] * window
        spectrum[:, i] = np.fft.rfft(frame)

    # Log magnitude
    mag = np.abs(spectrum)
    mag = np.maximum(mag, 1e-10)
    log_mag = 20.0 * np.log10(mag)

    # Instantaneous frequency
    phase = np.angle(spectrum)
    dphi = np.diff(phase, axis=0)
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
    dphi = np.concatenate([dphi, dphi[-1:]], axis=0)

    # Z-score normalise
    def zscore(x):
        mu, sigma = np.mean(x), np.std(x)
        return (x - mu) / sigma if sigma > 1e-10 else x - mu

    return zscore(log_mag).astype(np.float32), zscore(dphi).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert IQ files to spectrogram dataset")
    parser.add_argument("--input-dir", default=".", help="Directory with .cfile files")
    parser.add_argument("--output", default="dataset_post.h5", help="Output HDF5 file")
    parser.add_argument("--nfft", type=int, default=128)
    parser.add_argument("--hop", type=int, default=64)
    parser.add_argument("--fs", type=float, default=10e6)
    args = parser.parse_args()

    cfiles = sorted(glob.glob(os.path.join(args.input_dir, "*.cfile")))
    if not cfiles:
        print(f"No .cfile files found in {args.input_dir}")
        return

    print(f"Found {len(cfiles)} .cfile files")

    # Determine class mapping
    class_names = sorted(set(
        Path(f).stem.split("_")[2] for f in cfiles if "_" in Path(f).stem
    ))
    if not class_names:
        class_names = ["LTE", "WiFi", "V2X-PC5", "Noise"]
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    print(f"Classes: {class_names}")

    # Fixed dimensions from first file
    sig0 = read_cfile(cfiles[0])
    n_freq = args.nfft // 2 + 1
    n_frames = max(1, (len(sig0) - args.nfft) // args.hop + 1)

    log_mags = np.zeros((len(cfiles), n_freq, n_frames), dtype=np.float32)
    inst_freqs = np.zeros((len(cfiles), n_freq, n_frames), dtype=np.float32)
    labels = np.zeros(len(cfiles), dtype=np.int64)
    metadata = []

    for i, cf in enumerate(cfiles):
        print(f"  Processing [{i+1}/{len(cfiles)}]: {os.path.basename(cf)}")
        signal = read_cfile(cf)
        lm, inf = compute_spectrogram(signal, args.fs, args.nfft, args.hop)

        # Pad/truncate to fixed size
        actual = lm.shape[1]
        if actual < n_frames:
            lm = np.pad(lm, ((0, 0), (0, n_frames - actual)))
            inf = np.pad(inf, ((0, 0), (0, n_frames - actual)))
        else:
            lm = lm[:, :n_frames]
            inf = inf[:, :n_frames]

        log_mags[i] = lm
        inst_freqs[i] = inf

        # Load metadata
        json_path = cf.replace(".cfile", ".json")
        if os.path.exists(json_path):
            with open(json_path) as f:
                meta = json.load(f)
        else:
            meta = {"signal_type": "Unknown"}
        metadata.append(meta)
        labels[i] = class_to_idx.get(meta.get("signal_type", "Unknown"), 0)

    # Save
    with h5py.File(args.output, "w") as hf:
        hf.create_dataset("log_magnitude", data=log_mags, compression="gzip")
        hf.create_dataset("instantaneous_frequency", data=inst_freqs, compression="gzip")
        hf.create_dataset("labels", data=labels, compression="gzip")
        mg = hf.create_group("metadata")
        mg.attrs["num_samples"] = len(cfiles)
        mg.attrs["class_names"] = json.dumps(class_names)
        mg.attrs["nfft"] = args.nfft
        mg.attrs["hop"] = args.hop
        mg.attrs["fs_hz"] = args.fs
        dt = h5py.special_dtype(vlen=str)
        mds = mg.create_dataset("per_sample", (len(metadata),), dtype=dt)
        for i, m in enumerate(metadata):
            mds[i] = json.dumps(m)

    print(f"\\nSaved dataset to {args.output}")
    print(f"  Samples: {len(cfiles)}")
    print(f"  Spectrogram shape: ({n_freq}, {n_frames})")
    print(f"  Class distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")


if __name__ == "__main__":
    main()
'''
        fpath = Path(self.cfg.OUTPUT_DIR) / "gnuradio_scripts" / "postprocess_iq_to_spectrogram.py"
        with open(fpath, "w") as f:
            f.write(script)
        print(f"\n  [OK] Post-processing script: {fpath}")


# ===================================================================
# SECTION 9 – MAIN ENTRY POINT
# ===================================================================
def main():
    parser = argparse.ArgumentParser(
        description="V2X Spectrum Sensing Dataset Generation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 1: Simulation-only (default, no hardware needed)
  python generate_simulated_dataset.py --mode simulation

  # Mode 2: SUMO + GNU Radio integration (generates config files + scripts)
  python generate_simulated_dataset.py --mode sumo_gnuradio

  # Custom parameters
  python generate_simulated_dataset.py --num-samples 5000 --seed 123

  # Full pipeline: generate simulation data AND SUMO/GR configs
  python generate_simulated_dataset.py --mode both
""",
    )
    parser.add_argument(
        "--mode", choices=["simulation", "sumo_gnuradio", "both"],
        default="simulation",
        help="Pipeline mode (default: simulation)",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10_000,
        help="Total number of samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--sample-length", type=int, default=1024,
        help="IQ samples per spectrogram slice (default: 1024)",
    )
    parser.add_argument(
        "--fs", type=float, default=10e6,
        help="Sampling rate in Hz (default: 10e6)",
    )
    parser.add_argument(
        "--nfft", type=int, default=128,
        help="FFT size for STFT (default: 128)",
    )
    parser.add_argument(
        "--hop", type=int, default=64,
        help="Hop size for STFT (default: 64)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="simulated_rf_dataset",
        help="Output directory (default: simulated_rf_dataset)",
    )
    parser.add_argument(
        "--fc", type=float, default=5.9e9,
        help="V2X carrier frequency in Hz (default: 5.9e9)",
    )

    args = parser.parse_args()

    # Build config
    config = Config(
        NUM_SAMPLES=args.num_samples,
        SAMPLE_LENGTH=args.sample_length,
        FS=args.fs,
        NFFT=args.nfft,
        HOP=args.hop,
        SEED=args.seed,
        OUTPUT_DIR=args.output_dir,
        V2X_FC=args.fc,
        MODE=args.mode,
    )

    # Adjust samples to be evenly divisible
    config.NUM_SAMPLES = (config.NUM_SAMPLES // config.NUM_CLASSES) * config.NUM_CLASSES
    config.SAMPLES_PER_CLASS = config.NUM_SAMPLES // config.NUM_CLASSES

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         V2X Spectrum Sensing Dataset Generation Pipeline         ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Configuration:")
    print(f"    Mode:           {args.mode}")
    print(f"    Samples:        {config.NUM_SAMPLES} ({config.SAMPLES_PER_CLASS}/class)")
    print(f"    Sample length:  {config.SAMPLE_LENGTH} IQ samples")
    print(f"    Sampling rate:  {config.FS/1e6:.1f} MHz")
    print(f"    FFT size:       {config.NFFT}")
    print(f"    Hop size:       {config.HOP}")
    print(f"    Carrier freq:   {config.V2X_FC/1e9:.1f} GHz")
    print(f"    Seed:           {config.SEED}")
    print(f"    Output:         {config.OUTPUT_DIR}/")
    print()

    # Check dependencies
    missing = []
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    try:
        import h5py
    except ImportError:
        missing.append("h5py")

    if missing:
        print(f"  [WARN] Missing optional packages: {', '.join(missing)}")
        if "numpy" in missing:
            print("  [ERROR] numpy is REQUIRED. Install with: pip install numpy")
            sys.exit(1)

    # Ensure output directory
    ensure_dir(config.OUTPUT_DIR)

    metadata_for_mode2 = None

    # --- Run Mode 1: Simulation ---
    if args.mode in ("simulation", "both"):
        pipeline = SimulationPipeline(config)
        log_mags, inst_freqs, labels, metadata = pipeline.run()
        metadata_for_mode2 = metadata

    # --- Run Mode 2: SUMO + GNU Radio ---
    if args.mode in ("sumo_gnuradio", "both"):
        pipeline2 = SumoGnuradioPipeline(config)
        pipeline2.run(metadata=metadata_for_mode2)

    # --- Print final summary ---
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                    Pipeline Complete!                            ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Output directory: {config.OUTPUT_DIR}/")
    out_path = Path(config.OUTPUT_DIR)
    if args.mode in ("simulation", "both"):
        files = sorted(out_path.glob("*"))
        for f in files:
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"    {f.name:40s}  {size_kb:>10.1f} KB")
    if args.mode in ("sumo_gnuradio", "both"):
        for subdir in ["sumo_scenarios", "gnuradio_scripts"]:
            sub = out_path / subdir
            if sub.exists():
                n_files = sum(1 for _ in sub.rglob("*") if _.is_file())
                print(f"    {subdir}/{'':<{35 - len(subdir)}} {n_files:>10d} files")
    print()
    print("  Ready for ML training! Load dataset.h5 and use with v3.py")
    print()


if __name__ == "__main__":
    main()
