# v2x_phase1 — The Compliant Attacker (Phase 1 of the upgrade)

**Working title:** *The Price of Compliance: Emission-Mask-Constrained Adversarial
Attacks on Deep Learning Spectrum Sensing in the 5.9 GHz ITS Band*

This package contains the complete Phase-1 research framework: post-FCC coexistence
waveform generators, TR 37.885-inspired channels, a differentiable receiver front-end,
the mask-constrained waveform-domain attack (the core innovation), experiment
runners, results, and the paper with its claim-verification matrix.

**2026-09-05 audit corrections** (see CLAIMS.md header): the PSR budget is now the
window ENERGY of the transmit waveform referenced to the clean received-signal
energy (an earlier mean-power/energy mismatch shifted all absolute PSR labels by
33.11 dB); the genie baseline no longer carries the PSD cap; the band plan mirrors
the physically correct post-FCC layout (ITS allocation in the upper 10 MHz of a
window straddling 5895 MHz; U-NII-4 Wi-Fi below the boundary). Pre-fix results are
archived in `results/archive_pre_axis_fix/` (do not cite).

## Repository layout

```
src/
  waveforms.py      post-FCC band plan; block-exact PC5/802.11p/WiFi/Noise generators
  channels.py       TR 37.885-inspired highway/urban/rural block-fading channels
  receiver.py       differentiable torch front-end (STFT -> mag/IF) + models
  attack_mask.py    THE INNOVATION: mask-constrained waveform PGD + genie baseline,
                    MEAP / Price-of-Compliance metrics (censoring-aware)
scripts/
  run_train.py            train dual model + honesty baselines (checkpoints)
  run_mag_baseline.py     mag-only ablation (same budget) + baselines
  run_attack.py           compliant-attacker PSR sweeps (chunkable per scenario/mode)
  run_plot.py             merge + physical metrics + money figure
  debug_attack.py         surgical attack verification (run after any change)
  check_waveforms.py (../scripts) band-plan occupancy + PAPR sanity
results/           checkpoints + JSONs (all embed config/seed; pre-fix archive inside)
paper/             main.tex + CLAIMS.md + figs/
```

## Prototype results (in this repo, CPU; corrected axis)

| Setting | Urban | Highway | Rural |
|---|---|---|---|
| Genie MEAP untargeted (20% cond-ASR) | −44.2 dB | −43.9 dB | −44.5 dB |
| Compliant MEAP untargeted | −37.1 dB | −37.2 dB | −39.1 dB |
| **Price of Compliance, untargeted** | **7.1 dB** | **6.7 dB** | **5.4 dB** |
| Compliant MEAP, targeted cloaking | −14.8 dB | −14.2 dB | −13.0 dB |
| **Price of Compliance, cloaking** | **16.0 dB** | **17.4 dB** | **18.1 dB** |
| Compliant ASR @ PSR −10 dB (untargeted) | 96.7% | 92.0% | (see JSON) |
| Compliant ASR @ PSR −30 dB (untargeted) | 39.3% | 39.7% | (see JSON) |
| Cloaking @ PSR −10 dB | 38.3% | 36.7% | (see JSON) |

Clean task: dual CNN 100%, mag-only 100%, energy-LR 4-class 88.62%,
**PC5-vs-11p pair: LR 77.75% vs CNN 100%** (the honest DL-necessity number).

**Defense (mask-matched adversarial training)**: clean 100% retained; genie MEAP
+6.8 dB (−44.2→−37.5); compliant MEAP **+14.0 dB** (−37.1→−23.1); PoC doubled
(7.1→14.4 dB); compliant ASR@−25 dB 58.3→16.7%, @−20 dB 78.0→25.3%, @−10 dB
96.7→55.3%; cloaking@−10 dB 38.3→15.0%. Honest boundary: robustness confined
to the trained PSR range [−25,−5] dB (94.7% at 0 dB).
`results/at_defense_results.json`.

**Feature-space equivalence (C14, corrected after the Wave-4 audit found a
layout bug)**: the legacy ε-ball at ε=1.0 (1σ, z-units) implies received
power −24.2 dB (median); as evaluated in the legacy setting it achieves 71.3%
cond-ASR, but the SAME perturbation made physical (fed through the true
front-end) realizes only 14.0%; transmitted attacks at the same power: genie
99.3%, compliant 62.3%. The legacy model hides the power budget AND
overstates physical realizability. ε≤0.3 → 0–1.7% at −34..−44 dB.
`results/feature_space_equiv.json`.

**CSI robustness (C13)**: 0.3-relative CSI error costs the attack only 1–2 pp
(MEAP within 0.5 dB); NO CSI moves the compliant 20%-ASR threshold from −37 to
≈−16 dB (~21 dB penalty). `results/csi_mismatch.json`.

**Metric robustness**: PGD-50 shifts mask MEAP by −1.7 dB and raises mid-grid
ASR by up to +37 pp (PGD-10 numbers are conservative lower bounds); attack
seed 11 vs 7: PoC spread 0.28 dB; PSD-cap margin 2→10: PoC 7.1→6.2 dB (the
allocation constraint, not the flat cap, drives the cost).

Key findings: (1) an *undefended* sensing CNN is broken by a compliant attacker
tens of dB below the victim signal level; (2) emission-mask compliance costs the
attacker ~5–7 dB (untargeted) and 16–18 dB (targeted cloaking); (3) compliance
delays but does not prevent the attack; (4) target specificity (cloaking) is
where the mask bites hardest; (5) at high power the targeted attack overshoots
into non-target wrong classes (label-keyed pipelines beware); (6) mask-matched AT
and compliance compound (~7–8 dB each).

## Full protocol (run locally, ~2–4 h CPU or <30 min on any GPU)

```bash
# 1. three seeds, deeper training
python scripts/run_train.py --seeds 42 123 456 --epochs 40

# 2. full attack grid: seeds x scenarios x modes (chunkable; each invocation
#    writes its own JSON, run_plot merges everything)
for s in 11 22 33; do
  for m in untargeted targeted_noise; do
    python scripts/run_attack.py --scenarios urban highway rural \
        --psr -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 --steps 50 \
        --n-eval 200 --seed $s --modes $m
  done
done

# 3. PSD-cap idealization sensitivity: repeat key points with --psd-margin 10
# 4. mask-matched adversarial training defense (per paper §Defenses)
# 5. regenerate figure + metrics
python scripts/run_plot.py
```

Every result JSON embeds its config; `paper/CLAIMS.md` tracks which paper claims
are VERIFIED vs PENDING. **Do not write any claim into the manuscript that is not
VERIFIED in CLAIMS.md.** Paper-side pending markers are the greppable `\FULLP{}`
macro — never delete one until the backing JSON exists.

## Requirements

torch >= 2.0, numpy, scipy, scikit-learn, matplotlib. CPU works; GPU optional.
