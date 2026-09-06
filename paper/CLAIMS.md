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
| C3 | PAPR physics: PC5 6.26 dB < 802.11p 8.77 / WiFi 8.69 dB (noise 9.02; p95 6.95/9.89/9.67/10.32) | results/papr_pa_results.json (benign PAPR arms, 200 windows/class, rng 123) | VERIFIED |
| C4 | SNR sweep: CNN advantage persists to 0 dB SNR (73.0% vs LR 62.6% at [0,10] dB; pair 63.6%); vanishes at [−10,0] (25.6% vs 25.0%) — both starve, honest negative | results/snr_sweep.json; scripts/run_snr_sweep.py | VERIFIED (prototype scale) |
| C5 | IF/phase-difference stream contributes ~nothing (phase-aware framing retired) | C2 (dual = mag-only = 100%) | VERIFIED (seed 42); full: 3 seeds |

## B. Attack claims (corrected PSR axis; PSR = 10 log10(E_attack / E_clean_rx))

| # | Claim | Evidence | Status |
|---|---|---|---|
| C6 | Waveform-domain attack chain differentiable and effective: genie 100% cond-ASR at PSR 0 dB; 96.7% (urban) compliant at −10 dB | `results/attack_results_{urban,highway}_{untargeted,targeted_noise}.json` | VERIFIED (300 samples, PGD-10, seed 7) |
| C7 | Projection correctness: post-projection power == budget exactly (ratio 1.000000); OOB fraction 1.05e-14; genie has NO PSD cap (spectral line retains 100%); mask cap active (0.20% = 2/n_in) | `scripts/debug_attack.py` output | VERIFIED |
| C8 | Price of Compliance ≈ 7.1 dB (urban) / 6.7 dB (highway), untargeted, 20% threshold; all MEAPs exact (no censoring) | `results/attack_results_merged.json` summary | VERIFIED (prototype scale) |
| C9 | Compliance ≠ safety: compliant attacker 96.7% (urban) / 92.0% (highway) untargeted at PSR −10 dB; 39.3% at −30 dB | same | VERIFIED (prototype scale) |
| C10 | Cloaking (targeted-to-noise) costs 16.0 dB (urban) / 17.4 dB (highway) more under compliance; ≤0.4% at PSR ≤ −30 dB (0.33% at −30, highway/rural); 38.3% at −10 dB; saturation/overshoot at +10 dB (target-specificity drops while untargeted damage saturates) | same | VERIFIED (prototype scale) |
| C11 | Scenario robustness: urban ≈ highway attack curves (PoC spread < 0.5 dB untargeted, < 1.5 dB targeted) | same | VERIFIED (prototype scale) |
| C12 | 3-seed error bars; PGD-50 convergence; rural scenario; psd_margin sensitivity {2,10} | rural: attack_results_rural_*.json (PoC 5.4/18.1 dB); PGD-50: pgd50_urban_untargeted.json (mask MEAP −38.8 vs −37.1 = −1.7 dB; +3–6 pp at ≤−35 dB but +24.7/+37.0/+21.3 pp at −30/−25/−20 dB — PGD-10 numbers are lower bounds); seed 11: attack_results_urban_untargeted_seed11.json (PoC 6.86 vs 7.14 dB, spread 0.28 dB); margin 10: sensitivity_margin10_urban_untargeted.json (PoC 6.23 vs 7.14 dB → allocation constraint dominates); **3-seed TRAINING (Wave 7): seeds 123/456 retrained + full grids — mask MEAP −37.1/−38.1/−37.9 (spread 1.1 dB), PoC 7.1/6.9/7.1 (lower bound, genie floor-censored 2/3), clean 3×100%, CNN pair 3×100%, LR pair 74.0–77.8** | VERIFIED (three_seed_summary.json; defense+TRADES remain seed 42) |
| C13 | CSI-error robustness of the attack: 0.3-relative CSI error costs ≤2.3 pp on the mask curves, ≤2.7 pp on the genie curves (the MEAP shift itself is not resolvable — the 20% crossing sits below the stored grid floor); NO CSI (independent draw) moves the compliant 20%-ASR threshold from −37 to ≈−16 dB (~21 dB penalty) — channel-realization knowledge is what makes the white-box threat catastrophic | results/csi_mismatch.json; scripts/run_csi_mismatch.py | VERIFIED (prototype scale; noise-realization knowledge variant still open) |
| C14 | Feature-space (legacy) attack physical-power equivalence (CORRECTED after Wave-4 audit found a layout-mixing bug that mirrored the perturbation and corrupted PSR_eq by ~14 dB): ε=1.0 (1σ) → intended (legacy-setting) ASR 71.3%, implied received power −24.2 dB (median); the SAME perturbation made physical (fed through the true front-end) realizes only 14.0%; transmitted attacks at the same power: genie 99.3%, mask 62.3%. ε≤0.3 → 0–1.7% at −34..−44 dB. The legacy model hides the power budget AND overstates physical realizability | results/feature_space_equiv.json; scripts/feature_space_equiv.py | VERIFIED (prototype scale, corrected) |
| C15 | Attacker worst-case disclosure (uses exact received realization incl. noise): all ASRs are upper bounds | threat-model text in paper + attack_mask.py docstring | VERIFIED (disclosed) |

## C. Defense claims

| # | Claim | Evidence | Status |
|---|---|---|---|
| C16 | Mask-matched adversarial training (AT 0.5, 5-step PGD, PSR U[−25,−5], seed 42): clean val acc stays 100%; genie MEAP −44.2→−37.5 (+6.8 dB); compliant MEAP −37.1→−23.1 (+14.0 dB); PoC 7.1→14.4 dB (doubled); compliant ASR@−25 dB 58.3→16.7%, @−20 dB 78.0→25.3%, @−10 dB 96.7→55.3%; cloaking@−10 dB 38.3→15.0%; honest boundary: robustness confined to trained PSR range (94.7% at 0 dB) | results/at_defense_results.json + at_train_report.json + checkpoint_dual_at.pt; scripts/run_at_defense.py, run_at_eval.py | VERIFIED (prototype scale) |
| C17 | TRADES-style ablation (β=6, mask-matched generator, prob 0.5, PSR [−25,−5], seed 42): mask MEAP −19.1 dB vs standard AT −23.1 (PoC 18.6 vs 14.4) at 99.25% clean (−0.75 pp) — stronger compliant-axis defense at small clean cost | results/trades_train_report.json; checkpoint_dual_trades.pt; attack_results_urban_untargeted_trades.json; scripts/run_trades.py | VERIFIED (prototype scale, seed 42) |

## D. Reproducibility claims

| # | Claim | Evidence | Status |
|---|---|---|---|
| C18 | All numbers traceable: JSONs embed config/seed/env | results/*.json | VERIFIED |
| C19 | Phase-0 release reproduces ICE2CT-2026 Tables II/III/V regime | `v2x_release/results/reproduction/` | VERIFIED (seed 42) |
| C20 | cconv (torch attack chain) == np.convolve (numpy dataset chain) to 1.7e-07 rel err | debug_attack.py §1 | VERIFIED |
| C21 | Real OTA WiFi in the loop (Wave 6 G1): Fontaine/UGent USRP captures @5240 MHz (CC BY-NC-SA), DC-removed, ×2 resampled, placed [−10,0] MHz, flatness+power burst-gated (96 windows; eval=uz 74, train=rabot/reep 22, disjoint locations); frozen model real-WiFi acc 66.2%; frozen mask MEAP −35.9 dB; fine-tuned 73.0% acc, mask MEAP −37.7 dB, PoC 7.3 (lb) — canonical conclusion replicates on real signals; OOD margin collapse: 0.0% synthetic flips vs 67.35% real flips @−45 dB (stored frozen real-only curve; fine-tuned arm 68.5% by derivation from per-class fractions, not stored) | results/real_wifi_attack.json; real_wifi_checks.json; checkpoint_dual_realft.pt; src/real_wifi.py; scripts/run_real_wifi.py; w6_check.py | VERIFIED (prototype scale) |
| C22 | Second victim (2-ch early-fusion ResNet, 493k params, 100% clean, same task/recipe/seed): white-box genie −29.2 / mask −25.5 / PoC 3.7 dB vs dual −44.2/−37.1/7.1 — architecture shifts compliant threshold 11.6–15.1 dB, halves PoC; test suite discriminates victims | results/victim2_train_report.json; victim2_transfer.json; checkpoint_resnet.pt; src/victim_resnet.py; scripts/run_victim2.py | VERIFIED (prototype scale) |
| C23 | Surrogate-model transfer (M7): D2R genie −18.4/mask −10.0; R2D genie −21.0/mask −18.0 — transfer costs attacker 10.8–23.2 dB (white-box disclosure = genuine upper bound); PoC 8.4/3.0 in transfer | results/victim2_transfer.json (resumable grid state: victim2_grid_state.json) | VERIFIED (prototype scale) |
| C24 | AT defense rural transfer: trained urban/highway, rural eval mask −21.9 dB (vs −23.1 urban; PoC 13.4 vs 14.4) — defense transfers to unseen scenario at ~1.2 dB cost | results/attack_results_rural_untargeted_atrural.json; attack_results_urban_untargeted_at.json (protocol-parity check vs at_defense_results: exact) | VERIFIED (prototype scale, seed 42) |
| C25 | v2x-redteam CLI: one-command red-team evaluation + certification-style report; reproduces canonical −44.22/−37.08/7.14 exactly and the real-wifi track −37.67 | scripts/v2x_redteam.py; results/redteam_*.json + _report.md | VERIFIED |
| C26 | Wave-6 exit gate (D1): independent numpy re-derivation of all Wave-6 MEAPs/PoCs (match), curves monotone, transfer ≤ white-box, projection physics (budget 1.000000000, OOB 4.8e-32, numpy==torch to 2e-7) | scripts/w6b_check.py (exit: PASS) | VERIFIED |
| C27 | Wave-11 PA reality check (pre-registered H1–H3 in the JSON): optimized mask delta PAPR 11.05 dB mean (p95 12.8) vs benign PC5 6.26 / 11p 8.77 / WiFi 8.69 / noise 9.02 — H1 confirmed (attack is the highest-PAPR signal on the air) | results/papr_pa_results.json: papr_benign_db, papr_optimized_db, H1 | VERIFIED |
| C28 | Post-PA regrowth: at IBO 6 dB (p=3) peak OOB PSD −19.3 dBr p95, 92% of windows fail even the lenient −28 dBr gate (all fail strict); strict-gate (−40 dBr, 95% windows) requires IBO*=13 dB (p=3) / 15 dB (p=2); benign PA increment at IBO 6: +1.0 dB (PC5, the cleanest skirt at −29.7 dBr pre-PA) — attack pedestal ~9.4 dB above the cleanest benign neighbor's post-PA skirt | results/papr_pa_results.json: regrowth, ibo_star_strict_db, benign_regrowth_increment | VERIFIED |
| C29 | Effectiveness through PA: MEAP shifts +0.60 (IBO 0) / −0.09 (IBO 6) / 0.00 (IBO 12, 13, p=2 arms) dB vs in-process no-PA control MEAP −37.00 dB (canonical −37.08 on the coarser grid) — H2 (0.5–4 dB penalty) FALSIFIED in the attack's favor | results/papr_pa_results.json: curves.*__meap, control_meap_db | VERIFIED |
| C30 | PA-aware re-optimization (Rapp inside the PGD chain, budget-anchored power control): at IBO* 13 dB MEAP −38.23 dB (−1.23 dB vs control) with 100% strict-gate post-PA compliance (−60.2 dBr p95 OOB, post-PA PAPR 11.06 dB); at IBO 0 MEAP −38.32 dB but non-compliant (−12.8 dBr) with post-PA PAPR 4.95 dB — PA-aware attacker is stronger AND compliant at backoff; H3 kill criterion not triggered | results/papr_pa_results.json: curves.pa_aware_p3_ibo13__meap (incl. post_tx_papr_mean_db), pa_aware_p3_ibo0__meap; scripts/run_papr_pa.py (pa_aware_pgd) | VERIFIED |
| C31 | Two NaN/units bugs in the first PA-aware implementation were caught by the pre-registered physical-sanity gate (D2) and fixed before any publication use: (a) renorm 0/0 at zero-init → NaN gradients mimicking a 13 dB "improvement" (66.7% = argmax-of-NaN signature, 20/30); (b) asat sized to window energy instead of mean power (+33.11 dB phantom backoff, same class as the Wave-3 PSR bug); merge-on-write clobber also fixed (prev curves read before update). Deterministic seeds; all arms re-run post-fix | worklog Task 19; scripts/run_papr_pa.py comments (chain docstring) | VERIFIED (process claim) |
| C32 | PUEA lineage now cited and differentiated (Anand/Jin/Subbalakshmi DySPAN 2008; Ambhika Wireless Networks 30(5) 2024); threat-model table includes the PUEA row; 2026 band status grounded (FCC final rule eff. Feb 11 2025, two-year DSRC sunset; >50 C-V2X waivers per ITS America 2024; 5GAA Dec 2024 roadmap) | paper/main.tex related work + Table 1 + bib (anand2008, ambhika2024, itsa2024, 5gaa2024) | VERIFIED |

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
10. Wave-11 additions (2026-09-05, W11-B): Anand, Jin, Subbalakshmi, DySPAN 2008
    (pp. 1–12, cited-by 223 per ADS) ✓; Ambhika, Wireless Networks 30(5):3135–,
    2024 (Springer citation metadata fetched; single author) ✓; ITS America
    "Future of V2X in 5.9 GHz Report" May 2024 (PDF URL live; 50-waiver stat) ✓;
    5GAA C-V2X roadmap update Dec 2024 (org report, no page numbers) ✓.
    Not verified page-exact: DySPAN page range (1–12 from ADS) — minor.
Correction caught: O'Shea co-author "J. Nath" (wrong) → T. Roy.

## Known limitations carried into Limitations section

- Synthetic waveforms (standard-parameterized, not OTA captures; QPSK-only, no pilots)
- Idealized emission mask (flat in-band cap stricter than real regs; hard OOB null)
- TR 37.885-inspired channels (first-tap-only Rician; UMa-flavored urban spread)
- Block-fading (justified by Doppler coherence; approximate)
- Worst-case attacker knowledge (exact received realization incl. noise — upper bound; disclosed)
- Prototype-scale: seed 7 attack / seed 42 training, 300 samples, PGD-10 — full protocol pending (C12)
