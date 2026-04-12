# Dual-Stream Phase-Aware Inception-Time CNN for Adversarially Robust Spectrum Sensing in V2X Networks

[![Paper](https://img.shields.io/badge/Paper-ICE2CT--2026-blue)](https://github.com/Daveshvats/v2x_adversarial_sensing)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Official code repository for the ICE2CT-2026 paper:** *"Dual-Stream Phase-Aware Inception-Time CNN for Adversarially Robust Spectrum Sensing in V2X Networks"*

---

## Overview

This repository contains the complete implementation of a **Dual-Stream Phase-Aware Inception-Time CNN** designed for robust multi-class spectrum sensing in Vehicle-to-Everything (V2X) communication networks. The model classifies RF signals into four categories: **LTE, WiFi, V2X-PC5, and Noise**, operating in the **5.9 GHz DSRC band**.

### Key Contributions

- **Dual-stream architecture** that fuses log-magnitude spectrograms with instantaneous frequency representations for richer spectral feature extraction
- **Inception-Time multi-scale convolution blocks** with parallel 1x1, 3x3, and 5x5 kernels for capturing temporal patterns at multiple resolutions
- **TF-CutMix augmentation** with label smoothing and Gaussian noise injection for loss landscape smoothing
- **Comprehensive adversarial robustness evaluation** using six white-box attacks (FGSM, PGD-20, APGD-CE, APGD-DLR, FAB, Square) across six perturbation budgets
- **Channel-aware adversarial evaluation** revealing a Sim-to-Real amplification effect where low-SNR channel noise increases adversarial ASR by 2.5x
- **Cross-architecture transferability study** showing dual-stream fusion achieves the lowest transfer ASR (14-17%)
- **Mobility scenario-stratified evaluation** across highway, urban, and rural conditions with Rician/Doppler channel models

---

## Repository Structure

```
v2x_adversarial_sensing/
|-- README.md                          # This file
|-- requirements.txt                   # Python dependencies
|-- LICENSE                            # MIT License
|
|-- code/
|   |-- model.py                       # Dual-Stream Phase-Aware Inception-Time CNN architecture
|   |-- generate_signals.py            # Synthetic V2X RF signal generation (Rayleigh fading)
|   |-- adversarial_attacks.py         # FGSM & PGD-20 attack implementations
|   |-- defenses.py                    # Adversarial training & input denoising defenses
|   |-- run_experiments.py             # Main experiment runner (training + evaluation)
|   |-- run_all.py                     # Automated full pipeline (train + attack + defend)
|   |-- plot_results.py                # Publication-quality figure generation
|
|-- scripts/
|   |-- autoattack_eval.py             # AutoAttack robustness evaluation
|   |-- official_autoattack_eval.py    # Official AutoAttack library integration
|   |-- transferability_study.py       # Cross-architecture transferability experiments
|   |-- mobility_scenario_eval.py      # Highway/urban/rural scenario evaluation
|   |-- v3_rician_doppler.py           # Rician fading + Doppler shift channel models
|   |-- latency_benchmark.py           # CPU inference latency benchmarking
|   |-- generate_simulated_dataset.py  # Large-scale dataset generation pipeline
|
|-- results/                           # Experiment results (JSON format)
|-- figures/                           # Generated plots and visualizations
|-- checkpoints/                       # Trained model weights
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended) or CPU
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Daveshvats/v2x_adversarial_sensing.git
cd v2x_adversarial_sensing

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | >= 2.0 | Neural network framework |
| NumPy | >= 1.24 | Numerical computing |
| SciPy | >= 1.10 | Signal processing |
| Matplotlib | >= 3.7 | Visualization |
| Scikit-learn | >= 1.3 | Evaluation metrics |
| Torchvision | >= 0.15 | Data utilities |
| AutoAttack | >= 0.4 | Robustness evaluation |
| tqdm | >= 4.65 | Progress bars |

---

## Quick Start

### 1. Generate Synthetic Dataset

```bash
python code/generate_signals.py
```

This generates 4,000 synthetic RF samples across 4 signal classes (LTE, WiFi, V2X-PC5, Noise) with Rayleigh fading channel simulation at configurable SNR levels.

### 2. Train the Model

```bash
python code/run_experiments.py --mode train
```

The dual-stream model trains for 100 epochs with TF-CutMix augmentation, label smoothing (alpha=0.1), and Gaussian noise injection (std=0.01). Training follows a 3-seed protocol for statistical significance.

### 3. Evaluate Adversarial Robustness

```bash
# FGSM attack
python code/adversarial_attacks.py --attack fgsm --epsilon 0.03

# PGD-20 attack
python code/adversarial_attacks.py --attack pgd --epsilon 0.03 --iterations 20

# Full AutoAttack evaluation (APGD-CE, APGD-DLR, FAB, Square)
python scripts/official_autoattack_eval.py --epsilon 0.03
```

### 4. Run Complete Pipeline

```bash
# Train + Attack + Defend in one command
python code/run_all.py
```

### 5. Generate Figures

```bash
python code/plot_results.py
```

---

## Model Architecture

```
Input (2 x 128 x 128)
    |
    +-- Stream 1: Log-Magnitude Spectrogram (1 x 128 x 128)
    |       |
    |       +-- Conv2D(64, 7x7, stride=2)
    |       +-- MaxPool(3x3, stride=2)
    |       +-- InceptionBlock x4 (multi-scale: 1x1, 3x3, 5x5)
    |       +-- GlobalAvgPool
    |       +-- FC(128) + Dropout(0.3)
    |
    +-- Stream 2: Instantaneous Frequency (1 x 128 x 128)
    |       |
    |       +-- Conv2D(64, 7x7, stride=2)
    |       +-- MaxPool(3x3, stride=2)
    |       +-- InceptionBlock x4 (multi-scale: 1x1, 3x3, 5x5)
    |       +-- GlobalAvgPool
    |       +-- FC(128) + Dropout(0.3)
    |
    +-- Concatenation (256)
    +-- FC(128) + ReLU + Dropout(0.5)
    +-- FC(4) -- Softmax
```

---

## Key Results

| Metric | Standard Training | Adversarial Training |
|--------|-------------------|---------------------|
| Clean Accuracy | 86.67% (+-0.72%) | 87.46% (+-0.48%) |
| FGSM ASR (eps=0.03) | 23.25% | 21.12% |
| PGD-20 ASR (eps=0.03) | 24.33% | 22.08% |
| Inference Latency | 1.43 ms | 1.43 ms |

### Critical Findings

- **Smooth loss landscape**: FGSM-PGD ASR gap < 1.8 pp at eps <= 0.05, meaning iterative attacks provide minimal advantage
- **Transferability resistance**: Dual-stream fusion achieves lowest cross-architecture transfer ASR (14-17%)
- **Sim-to-Real gap**: Adversarial ASR increases from 23% (no channel) to >57% at -5 dB SNR
- **Scenario-stratified vulnerability**: Urban Rayleigh environments (ASR up to 50%) vs. Rician fading (ASR < 2.5%)

---

## Attack Configurations

| Attack | Type | Perturbation Budgets |
|--------|------|---------------------|
| FGSM | Single-step gradient | eps in {0.01, 0.03, 0.05, 0.07, 0.09, 0.11} |
| PGD-20 | Multi-step projected gradient | eps in {0.01, 0.03, 0.05, 0.07, 0.09, 0.11} |
| APGD-CE | Adaptive step-size CE | eps in {0.01, 0.03, 0.05} |
| APGD-DLR | Adaptive DLR loss | eps in {0.01, 0.03, 0.05} |
| FAB | Fast Adaptive Boundary | eps in {0.01, 0.03, 0.05} |
| Square | Query-based | eps in {0.01, 0.03, 0.05} |

---

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `scripts/autoattack_eval.py` | Custom AutoAttack benchmark |
| `scripts/official_autoattack_eval.py` | Official AutoAttack library wrapper |
| `scripts/transferability_study.py` | Cross-architecture attack transfer |
| `scripts/mobility_scenario_eval.py` | Highway/urban/rural evaluation |
| `scripts/v3_rician_doppler.py` | Rician + Doppler channel models |
| `scripts/latency_benchmark.py` | CPU inference timing |
| `scripts/generate_simulated_dataset.py` | Large-scale dataset generation |

---

## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@inproceedings{v2x_adversarial_sensing_2026,
  title={Dual-Stream Phase-Aware Inception-Time CNN for Adversarially Robust Spectrum Sensing in V2X Networks},
  author={Author Name},
  booktitle={Proceedings of the International Conference on Computing, Communication, and Technologies (ICE2CT-2026)},
  year={2026},
  organization={IEEE}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author**: Parveen Dhankhar, Davesh vats
- **Affiliation**: Department of Computer Science and Engineering
Vaish College of Engineering, Rohtak, India
- **Email**:  parveendhankhar2005@gmail.com,vatsdavesh@gmail.com
- **Repository**: [https://github.com/Daveshvats/v2x_adversarial_sensing](https://github.com/Daveshvats/v2x_adversarial_sensing)