# CLAIMS.md — Claim → Evidence → Status verification matrix

Every claim the paper will make, mapped to the artifact that proves it.
**RULE: a claim does not enter the manuscript until its row here is VERIFIED.**
Status values: `VERIFIED (prototype)` = reproduced in this repo at reduced scale;
`PENDING (full protocol)` = needs the full-scale local/GPU run listed in
`README.md (Full protocol)`.

**Axis correction 2026-09-05 (Wave-1 audit F1/F4):** all attack results were
re-run after (a) fixing the PSR budget unit (window energy, not mean power —
the old labels were shifted by 10·log10(2048) = 33.11 dB), (b) removing the
silent PSD cap on the genie baseline, and (c) mirroring the band plan to the
physically correct post-FCC layout. Pre-fix JSONs are archived in
`results/archive_pre_axis_fix/` (do not cite). All numbers below are from the
corrected runs.

## A. Clean-task claims

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | Post-FCC boundary-straddling task: window [5885,5905] MHz; PC5/11p co-channel in ITS upper 10 MHz; WiFi = partial view of U-NII-4 channel below the boundary; PC5-vs-11p pair NOT separable by region energy | `src/waveforms.py` band plan; occupancy check (scripts/check_waveforms.py): in-band fractions 99.7/97.2/95.0% | VERIFIED (design invariant) |
| C2 | Energy-LR 4-class 88.62%; **PC5-vs-11p pair 77.75%**; CNN (dual) 100%, mag-only 100%, CNN pair 100% at SNR [5,25] dB (same val split for all) | `results/train_report.json` | VERIFIED (seed 42) |
| C3 | PAPR physics: SC-FDMA ≈ 6.1 dB < OFDM ≈ 8.8/8.6 dB | check_waveforms output (50 draws) | VERIFIED |
| C4 | SNR sweep: CNN advantage persists to 0 dB SNR (73.0% vs LR 62.6% at [0,10] dB; pair 63.6%); vanishes at [−10,0] (25.6% vs 25.0%) — both starve, honest negative | results/snr_sweep.json; scripts/run_snr_sweep.py | VERIFIED (prototype scale) |
| C5 | IF/phase-difference stream contributes ~nothing (phase-aware framing retired) | C2 (dual = mag-only = 100%) | VERIFIED (seed 42); full: 3 seeds |

## B. Attack claims (corrected PSR axis; PSR = 10 log10(E_attack / E_clean_rx))

| # | Claim | Evidence | Status |
|---|---|---|---|
| C6 | Waveform-domain attack chain differentiable and effective: genie 100% cond-ASR at PSR 0 dB; 96.7% (urban) compliant at −10 dB | `results/attack_results_{urban,highway}_{untargeted,targeted_noise}.json` | VERIFIED (300 samples, PGD-10, seed 7) |
| C7 | Projection correctness: post-projection power == budget exactly (ratio 1.000000); OOB fraction 1.05e-14; genie has NO PSD cap (spectral line retains 100%); mask cap active (0.20% = 2/n_in) | `scripts/debug_attack.py` output | VERIFIED |
| C8 | Price of Compliance ≈ 7.1 dB (urban) / 6.7 dB (highway), untargeted, 20% threshold; all MEAPs exact (no censoring) | `results/attack_results_merged.json` summary | VERIFIED (prototype scale) |
| C9 | Compliance ≠ safety: compliant attacker 96.7% (urban) / 92.0% (highway) untargeted at PSR −10 dB; 39.3% at −30 dB | same | VERIFIED (prototype scale) |
| C10 | Cloaking (targeted-to-noise) costs 16.0 dB (urban) / 17.4 dB (highway) more under compliance; 0% below −25 dB; 38.3% at −10 dB; saturation/overshoot at +10 dB (target-specificity drops while untargeted damage saturates) | same | VERIFIED (prototype scale) |
| C11 | Scenario robustness: urban ≈ highway attack curves (PoC spread < 0.5 dB untargeted, < 1.5 dB targeted) | same | VERIFIED (prototype scale) |
| C12 | 3-seed error bars; PGD-50 convergence; rural scenario; psd_margin sensitivity {2,10} | rural: attack_results_rural_*.json (PoC 5.4/18.1 dB); PGD-50: pgd50_urban_untargeted.json (mask MEAP −38.8 vs −37.1, +3–6 pp at low power); seed 11: attack_results_urban_untargeted_seed11.json (PoC 6.86 vs 7.14 dB, spread 0.28 dB); margin 10: sensitivity_margin10_urban_untargeted.json (PoC 6.23 vs 7.14 dB → allocation constraint dominates) | VERIFIED (prototype scale; 3-seed TRAINING protocol still pending) |
| C13 | CSI-error robustness of the attack: 0.3-relative CSI error costs 1–2 pp (MEAP within 0.5 dB — attack robust to CSI error); NO CSI (independent draw) moves the compliant 20%-ASR threshold from −37 to ≈−16 dB (~21 dB penalty) — channel-realization knowledge is what makes the white-box threat catastrophic | results/csi_mismatch.json; scripts/run_csi_mismatch.py | VERIFIED (prototype scale; noise-realization knowledge variant still open) |
| C14 | Feature-space (legacy) attack physical-power equivalence (CORRECTED after Wave-4 audit found a layout-mixing bug that mirrored the perturbation and corrupted PSR_eq by ~14 dB): ε=1.0 (1σ) → intended (legacy-setting) ASR 71.3%, implied received power −24.2 dB (median); the SAME perturbation made physical (fed through the true front-end) realizes only 14.0%; transmitted attacks at the same power: genie 99.3%, mask 62.3%. ε≤0.3 → 0–1.7% at −34..−44 dB. The legacy model hides the power budget AND overstates physical realizability | results/feature_space_equiv.json; scripts/feature_space_equiv.py | VERIFIED (prototype scale, corrected) |
| C15 | Attacker worst-case disclosure (uses exact received realization incl. noise): all ASRs are upper bounds | threat-model text in paper + attack_mask.py docstring | VERIFIED (disclosed) |

## C. Defense claims

| # | Claim | Evidence | Status |
|---|---|---|---|
| C16 | Mask-matched adversarial training (AT 0.5, 5-step PGD, PSR U[−25,−5], seed 42): clean val acc stays 100%; genie MEAP −44.2→−37.5 (+6.8 dB); compliant MEAP −37.1→−23.1 (+14.0 dB); PoC 7.1→14.4 dB (doubled); compliant ASR@−25 dB 58.3→16.7%, @−20 dB 78.0→25.3%, @−10 dB 96.7→55.3%; cloaking@−10 dB 38.3→15.0%; honest boundary: robustness confined to trained PSR range (94.7% at 0 dB) | results/at_defense_results.json + at_train_report.json + checkpoint_dual_at.pt; scripts/run_at_defense.py, run_at_eval.py | VERIFIED (prototype scale) |
| C17 | TRADES-style variant as ablation | not yet implemented | PENDING |

## D. Reproducibility claims

| # | Claim | Evidence | Status |
|---|---|---|---|
| C18 | All numbers traceable: JSONs embed config/seed/env | results/*.json | VERIFIED |
| C19 | Phase-0 release reproduces ICE2CT-2026 Tables II/III/V regime | `v2x_release/results/reproduction/` | VERIFIED (seed 42) |
| C20 | cconv (torch attack chain) == np.convolve (numpy dataset chain) to 1.7e-07 rel err | debug_attack.py §1 | VERIFIED |

## Bibliography verification queue — ALL VERIFIED (2026-09-05, W1-C)

1. Girmay et al. 2023 — Vehicular Communications 39:100563 ✓
2. Kim & Sagduyu 2020 — CISS 2020, pp. 1–6 ✓
3. "Sagduyu et al. TWC 2021" — actually B. Kim first author, TWC 21(6):3868–3880, 2022 ✓
4. Liu et al. 2022 — IEEE Trans. Reliability 72(2):431–444, 2023 ✓
5. Zheng et al. TMLCN 2026 — vol. 4, pp. 950–965 ✓
6. RadioShock 2026 — IEEE TDSC 23(4):8874–8890 ✓
7. Zhao et al. INFOCOM 2024 — pp. 691–700 ✓
8. Habler et al. 2025 — J. Network and Computer Applications 236:104090 ✓
9. FCC 20-164 (2020) + FCC 24-123 (2024 Second R&O) ✓; ETSI EN 302 571 V2.1.1 ✓;
   TR 37.885 V15.3.0 ✓; TS 36.211 V15.14.0 ✓; IEEE 802.11-2012 ✓
Correction caught: O'Shea co-author "J. Nath" (wrong) → T. Roy.

## Known limitations carried into Limitations section

- Synthetic waveforms (standard-parameterized, not OTA captures; QPSK-only, no pilots)
- Idealized emission mask (flat in-band cap stricter than real regs; hard OOB null)
- TR 37.885-inspired channels (first-tap-only Rician; UMa-flavored urban spread)
- Block-fading (justified by Doppler coherence; approximate)
- Worst-case attacker knowledge (exact received realization incl. noise — upper bound; disclosed)
- Prototype-scale: seed 7 attack / seed 42 training, 300 samples, PGD-10 — full protocol pending (C12)
