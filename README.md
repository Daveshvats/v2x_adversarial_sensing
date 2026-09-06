# v2x_adversarial_sensing — The Compliant Attacker

[![CI](https://github.com/Daveshvats/v2x_adversarial_sensing/actions/workflows/ci.yml/badge.svg)](https://github.com/Daveshvats/v2x_adversarial_sensing/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Release](https://img.shields.io/badge/release-phase1--v1.2-blue.svg)](https://github.com/Daveshvats/v2x_adversarial_sensing/releases)
[![Data: CC BY-NC-SA 4.0](https://img.shields.io/badge/data%20captures-CC%20BY--NC--SA%204.0-lightgrey.svg)](data/real_wifi/ATTRIBUTION.md)

**Working title:** *The Price of Compliance: Emission-Mask-Constrained Adversarial
Attacks on Deep Learning Spectrum Sensing in the 5.9 GHz ITS Band*

> Can a transmitter that follows **every spectrum rule** — right frequency, right
> allocation, inside its emission mask, passing every spectrum monitor — still break
> a deep-learning coexistence sensor? Yes. And this repository is the complete,
> audited research framework that shows how, what it costs the attacker, how to
> defend against it, and where the remaining honest gaps are.

Read [`ONE_PAGER.md`](ONE_PAGER.md) for the plain-language summary (no math),
or [`paper/main.pdf`](paper/main.pdf) for the full manuscript draft. Every number
in the paper is traced to a result JSON via the claim-verification matrix
[`paper/CLAIMS.md`](paper/CLAIMS.md).

## Quick start (the red-team harness)

```bash
pip install -r requirements.txt          # CPU-only PyTorch works

# evaluate any released victim against a rule-compliant attacker,
# with physics checks and a certification-style report:
python scripts/v2x_redteam.py evaluate --victim dual --scenario urban
python scripts/v2x_redteam.py report --json results/redteam_dual_cv2x_attacker_urban.json

# victims: dual | mag | resnet | dual-at | dual-realft
# real-OTA-WiFi eval set:  add --real-wifi
```

The report gives MEAP (minimum received power at which the compliant attacker
crosses a chosen conditional-ASR threshold), the Price of Compliance, the full
ASR-vs-power sweep, and in-run verification that the attack actually respected
the emission mask (post-projection budget ratio and out-of-band leakage).

## Repository layout

```
src/                      the research package
  waveforms.py              post-FCC band plan; block-exact PC5/802.11p/WiFi/Noise generators
  channels.py               TR 37.885-inspired highway/urban/rural block-fading channels
  receiver.py               differentiable torch front-end (STFT -> mag/IF) + models
  attack_mask.py            THE INNOVATION: mask-constrained waveform PGD + genie baseline,
                            MEAP / Price-of-Compliance metrics (censoring-aware)
  real_wifi.py              real OTA capture loader (burst gating, placement, provenance)
  victim_resnet.py          second victim architecture
scripts/                  experiment runners + independent audit checkers
  v2x_redteam.py            ONE-COMMAND red-team evaluation + report (start here)
  run_train.py              train dual model + honesty baselines (checkpoints, --seed/--tag)
  run_mag_baseline.py       mag-only ablation (same budget) + baselines
  run_attack.py             compliant-attacker PSR sweeps (--checkpoint for any victim)
  run_victim2.py            train + attack the second victim (ResNet) + transfer grids
  run_real_wifi.py          real-OTA-WiFi eval + leakage-free fine-tune probe
  run_trades.py             TRADES-style ablation (resumable chunks)
  run_papr_pa.py            PA reality check: PAPR/regrowth vs IBO, MEAP arms,
                            PA-aware re-optimization (resumable --arms chunks)
  run_csi_mismatch.py       CSI/no-CSI attack variants
  run_at_defense.py         mask-matched adversarial training (resumable)
  run_at_eval.py            defense evaluation
  run_snr_sweep.py          clean-task SNR sweep (honest negative)
  run_plot.py               merge + physical metrics + money figure
  debug_attack.py           surgical attack verification (run after any change)
  w6_check.py, w6b_check.py     Wave-6 physics gates + independent exit-gate checker
  w11_check1..5_*.py            Wave-11 independent audit checkers (43 assertions)
results/                  checkpoints + JSONs (all embed config/seed; archive inside)
data/real_wifi/           9 raw USRP captures (CC BY-NC-SA; see ATTRIBUTION.md)
paper/                    main.tex + main.pdf + CLAIMS.md + figs/
ONE_PAGER.md              plain-language summary for engineers and regulators
REGULATORY_COMMENT_DRAFT.md   draft public comment on learning-based coexistence sensing
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
allocation constraint, not the flat cap, drives the cost). **Training seeds
42/123/456**: compliant MEAP −37.1/−38.1/−37.9 (1.1 dB spread), PoC
7.1/6.9/7.1 (lower bound 2/3), clean 3×100%. `results/three_seed_summary.json`.

**Second victim (ResNet, 493k params)**: white-box genie −29.2 / compliant
−25.5 / PoC 3.7 dB — architecture shifts the compliant threshold by 12–15 dB.
**Surrogate transfer**: 11–25 dB penalty (white-box disclosure = genuine upper
bound). `results/victim2_transfer.json`.

**Real OTA WiFi in the loop** (Fontaine/UGent USRP captures, 5240 MHz):
frozen model 66.2% real-WiFi accuracy; compliant MEAP −35.9 dB (frozen),
−37.7 dB after leakage-free fine-tune (PoC 7.3) — the canonical conclusion
replicates on real signals. OOD margin collapse: 0.0% synthetic flips vs
68.5% real flips at −45 dB PSR (synthetic-only evaluation overstates
robustness). `results/real_wifi_attack.json`.

**TRADES ablation (C17)**: mask MEAP −19.1 dB (vs standard AT −23.1), PoC
18.6, at 99.25% clean — stronger compliant-axis defense at 0.75 pp clean cost.
**AT rural transfer**: −21.9 dB (vs −23.1 urban) — the defense transfers to an
unseen scenario at ~1.2 dB cost.

**PA reality check (Wave 11, C27–C30)**: the optimized attack waveform is
the highest-PAPR signal on the air (11.05 dB vs 6.26–9.02 benign); through
a Rapp PA it needs 13 dB (p=3) / 15 dB (p=2) backoff for post-PA strict-
mask compliance (0% pass at the typical 6 dB), while losing almost no
effectiveness at any backoff; a PA-aware re-optimization (Rapp inside the
PGD chain) is *stronger* than the no-PA control (−1.2 dB MEAP) AND fully
post-PA compliant at backoff. Verification of compliance must happen at
the PA output port. `results/papr_pa_results.json` (pre-registered
hypotheses H1–H3 embedded).

Key findings: (1) an *undefended* sensing CNN is broken by a compliant attacker
tens of dB below the victim signal level; (2) emission-mask compliance costs the
attacker ~5–7 dB (untargeted) and 16–18 dB (targeted cloaking); (3) compliance
delays but does not prevent the attack; (4) target specificity (cloaking) is
where the mask bites hardest; (5) at high power the targeted attack overshoots
into non-target wrong classes (label-keyed pipelines beware); (6) mask-matched AT
and compliance compound (~7–8 dB each).

## Research integrity (audit trail)

This repository was produced under an adversarial internal-review protocol:
every wave of experiments was followed by an independent code+paper audit, and
every finding was fixed or disclosed before shipping. The audit trail is part
of the artifact:

- [`paper/CLAIMS.md`](paper/CLAIMS.md) — claim-by-claim verification matrix
  (VERIFIED / PENDING), the contract between paper text and result JSONs.
- `results/w11_audit_report.md` — the Wave-11 independent audit report
  (PAPR/PA reality check: PASS, with pre-ship fixes applied).
- `results/archive_pre_axis_fix/` — pre-correction results, kept for the
  record, **do not cite**.
- `scripts/w6_check.py`, `scripts/w11_check*.py` — re-runnable independent
  checkers (own re-implementations, no shared code with the audited scripts).

**2026-09-05 audit corrections** (see CLAIMS.md header): the PSR budget is the
window ENERGY of the transmit waveform referenced to the clean received-signal
energy (an earlier mean-power/energy mismatch shifted all absolute PSR labels by
33.11 dB); the genie baseline no longer carries the PSD cap; the band plan mirrors
the physically correct post-FCC layout (ITS allocation in the upper 10 MHz of a
window straddling 5895 MHz; U-NII-4 Wi-Fi below the boundary). Pre-fix results are
archived in `results/archive_pre_axis_fix/`.

## Full protocol (extended scale; the 3-seed core is ALREADY in results/)

Already completed in this release: 3 training seeds + grids
(three_seed_summary.json), rural/PGD-50/seed-11/margin sensitivity, CSI
mismatch, defense + TRADES + rural transfer, real-OTA WiFi study, second
victim + transfer, feature-space equivalence, PA reality check.

Remaining extensions (local, CPU-hours):

```bash
# 1. deeper training per seed (recipe identical, longer budget)
python scripts/run_train.py --seed 42 --epochs 40 --tag _e40

# 2. full-grid PGD-50 across seeds x scenarios x modes (each invocation
#    writes its own JSON; run_plot merges the canonical set)
for s in 11 22 33; do
  for m in untargeted targeted_noise; do
    python scripts/run_attack.py --scenarios urban highway rural \
        --psr -45 -40 -35 -30 -25 -20 -15 -10 -5 0 5 10 --steps 50 \
        --n-eval 200 --seed $s --modes $m --tag-out _pgd50_$s
  done
done

# 3. PSD-cap idealization sensitivity: repeat key points with --psd-margin 10
# 4. defense seeds 123/456 (run_at_defense.py --seed, then run_attack
#    --checkpoint checkpoint_dual_at_s*.pt)
# 5. regenerate figure + metrics
python scripts/run_plot.py
```

Every result JSON embeds its config; `paper/CLAIMS.md` tracks which paper claims
are VERIFIED vs PENDING. **Do not write any claim into the manuscript that is not
VERIFIED in CLAIMS.md.** Paper-side pending markers are the greppable `\FULLP{}`
macro — never delete one until the backing JSON exists.

## Requirements

`torch >= 2.0`, `numpy`, `scipy`, `matplotlib` — see
[`requirements.txt`](requirements.txt). CPU works; GPU optional.
All shipped results were produced on CPU with fixed seeds.

## The Phase-0 record (this repository's history)

The first release of this repository (tag
[`v2.0.0`](https://github.com/Daveshvats/v2x_adversarial_sensing/releases/tag/v2.0.0))
was the Phase-0 honest re-release of the ICE2CT-2026 paper pipeline
(*Dual-Stream Phase-Aware Inception-Time CNN for Adversarially Robust Spectrum
Sensing in V2X Networks*, Dhankhar & Vats): its source, legacy results, and the
full honesty audit (`AUDIT.md`, `results/legacy/`) are preserved under that tag
and its GitHub release. The current tree is the Phase-1 research framework,
which supersedes the Phase-0 pipeline and adds the compliance-constrained
threat model, physical-layer realism, defenses, and the audit protocol
described above.

## Citation

```bibtex
@software{vats_v2x_phase1_2026,
  author = {Dhankhar, Parveen and Vats, Davesh},
  title  = {The Compliant Attacker: emission-mask-constrained adversarial attacks
            on deep learning spectrum sensing in the 5.9 GHz ITS band
            (research framework)},
  year   = {2026},
  publisher = {GitHub},
  url    = {https://github.com/Daveshvats/v2x_adversarial_sensing}
}
```

See also [`CITATION.cff`](CITATION.cff) (software) and the manuscript draft in
[`paper/`](paper/) (cite the paper once published).

## License

- **Code:** [MIT License](LICENSE) — Copyright (c) 2026 Parveen Dhankhar, Davesh Vats.
- **Data:** the 9 over-the-air 802.11 captures in `data/real_wifi/` are
  redistributed under **CC BY-NC-SA 4.0** — see
  [`data/real_wifi/ATTRIBUTION.md`](data/real_wifi/ATTRIBUTION.md) for source,
  provenance, and terms.
