# AUDIT.md — Honesty Audit of the v2x_adversarial_sensing Artifacts

Purpose: every number that has ever appeared in a public artifact of this project is
accounted for here — what it is, whether it reproduces, and what to trust. This document
exists because earlier repository states mixed results from **three different code
universes**, and one JSON cannot be reproduced from any preserved code.

## §1 The three code universes

| Universe | Where its code lived | What it computed | Its numbers |
|---|---|---|---|
| **v1** (stale) | `code/*.py` in the private zip — 5-class modulation-recognition CNN (V2XSpectrumCNN, 64×64 single-channel spectrograms, ~800K params) | old sensing experiments, early defenses | 35–48 % accuracy in `intermediate.json`, `defense_results.json` |
| **v3** (the paper) | `scripts/autoattack_eval.py` (self-contained; also duplicated inside the other `scripts/*.py`) | dataset gen, Dual-Stream Inception-Time CNN (86,052 params), training, FGSM/PGD/APGD/FAB/Square | **86.67 ± 0.72 % clean, 23.25 % FGSM @ ε=0.03 — the paper's Tables II/III/V** |
| **AA-official** (lost) | an unpreserved variant of `scripts/official_autoattack_eval.py` | official AutoAttack library re-evaluation | **58.2 % clean, 40.3 % ASR @ ε=0.005** in `autoattack_official_results.json` |

The v1 `code/` directory and the v3 numbers coexisted in the same repository/zip, which is
why the repo appeared to contradict the paper. In this release v1 is removed and v3 is the
only canonical pipeline.

## §2 The 58.2 % result: what happened and what we did about it

`results/legacy/autoattack_official_results.json` reports clean accuracy 58.2 % and much
higher ASRs than the paper. Evidence it is not reproducible from any preserved code:

1. The shipped `official_autoattack_eval.py` trains the same architecture with the same
   recipe as `v3_pipeline.py` (verified by code inspection; model and data-gen sections are
   equivalent), on the same device (the script hardcodes `DEVICE = torch.device("cpu")`).
2. The JSON records `"device": "cuda"` — i.e. it was produced by a *different, unpreserved*
   code state on a CUDA machine.
3. The v3 pipeline reproduces the paper's numbers on CPU (see §4), so the discrepancy is
   not hardware.

**Decision:** the 58.2 % JSON is retained under `results/legacy/` for the record but marked
**INVALID — DO NOT CITE**. The lesson recorded here: any experiment whose code state is not
preserved verbatim does not belong in a results directory. All new results in
`results/reproduction/` embed their git-style config, seed, and library versions.

## §3 Metric correction: raw ASR overstates low-ε attack success

The paper defines ASR over *all* test samples. Let clean accuracy be a and the attack flip
a fraction f of the *correctly classified* samples. Then:

    raw_asr = (1 − a) + a·f      (approximately, when attacks only flip correct samples)

With a ≈ 0.87, at ε = 0.005 the paper's raw ASR 14.87 % decomposes as ≈ 13 % base error +
≈ 1.8 % true attack effect. The metric most readers *intend* by "attack success rate" is
the conditional one:

    cond_asr = (# samples correctly classified clean AND misclassified after attack)
               / (# samples correctly classified clean)

From this release forward, **both robust accuracy and conditional ASR are reported**. This
also explains the paper's "five identical attacks" pattern: at low ε, raw ASR is dominated
by the base error term, which is attack-independent — hence FGSM = PGD = APGD = FAB to
within a few tenths of a point. The convergence claim should be (re-)validated on
conditional ASR, which `reproduce/run_reproduction.py` computes.

## §4 Reproduction evidence (this release)

Environment: PyTorch 2.14.0+cpu, Python 3.12, Linux, CPU-only. Seed 42, ε = 0.03.

| Quantity | v3 pipeline output |
|---|---|
| Parameters | 86,052 (exact match) |
| Clean accuracy | 88.00 % (paper 3-seed mean 86.67 ± 0.72) |
| FGSM ASR | 26.63 % (paper 23.25 ± 1.53) |
| PGD-20 ASR | 27.00 % (paper 23.50 ± 1.48) |
| APGD-CE ASR | 27.00 % (paper 23.50) |
| APGD-DLR ASR | 26.63 % (paper 23.25) |

FAB and Square did not complete under the audit sandbox's ~5-minute process cap; they are
included in the harness for unrestricted machines. Full logs: `results/reproduction/`.

## §5 Other corrections

- **Latency:** paper §7.8 says "1.43 ms on an NVIDIA GPU"; `latency_results.json` says
  `device: cpu`, batch-1 mean 0.985 ms. The 1.43 ms figure is a CPU number (likely with
  different warm-up). Corrected text should read "≈1–1.5 ms on CPU".
- **README architecture:** the old README diagram (7×7 stem, 4 Inception blocks, 128×128
  input, 2×128×128 input shape) matches no preserved code. The real model: per stream
  Conv(1→16, 3×3) → BN → ReLU → MaxPool → InceptionBlock(16→32) → MaxPool →
  InceptionBlock(32→64) → GAP; concat 128-d → FC(128→64) → dropout → FC(64→4). Input per
  stream: 1×65×15.
- **License:** the old README referenced a LICENSE file that was never committed. MIT is
  now included.
- **ε grids:** README mentioned ε ∈ {0.01…0.11}; paper uses {0.005, 0.01, 0.02, 0.03,
  0.05, 0.08}. The paper's grid is canonical.

## §6 What remains open (honest limitations carried into the next paper)

1. The threat model is **feature-space** (perturbations applied to spectrograms inside the
   receiver), not physically realizable by an RF transmitter. The follow-up work moves the
   attack to the waveform domain with emission-mask constraints.
2. Classes occupy non-overlapping sub-bands → an energy-per-region baseline nearly matches
   the CNN. The follow-up uses overlapping-band coexistence classes and reports the energy
   baseline alongside.
3. The instantaneous-frequency stream contributes almost no class information (IF-only
   ≈ 30 % ≈ chance on 4 classes). The "phase-aware" framing is fragile and is demoted to a
   pre-registered ablation in the follow-up.
