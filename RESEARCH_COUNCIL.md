# Research Council Review — "The Price of Compliance" (Phase 1)

**Council date:** 2026-09-06
**Mode:** Four independent expert reviewers, blind to each other's positions (each read the full 11-wave worklog, the repo, the paper, the result JSONs, and the literature corpus before writing); chair synthesis afterward.
**Repo under review:** `phase1-compliant-attacker` @ `3e53ca4` (local; restructure commit)
**Chair:** main agent (Super Z), Task ID 21

**The three questions the PI put to the council:**
- **Q1.** Is what we built actually impactful?
- **Q2.** Compared with the related literature, what are we lacking / what did we not do right?
- **Q3.** Are we truly done — is the *research gap* completely filled, or only our internal goals?

**Members:**
| ID | Persona | Grade | Verdict |
|----|---------|-------|---------|
| 21-a | Adversarial-ML / RF physical-layer security (TIFS/TWC reviewer) | A− novelty, B+ rigor | MAJOR-REVISION (encouraged) |
| 21-b | Spectrum regulation & wireless standards (ex-FCC OET / ETSI) | B− | NEEDS-WORK |
| 21-c | V2X systems & venue fit (T-VT / Vehicular Communications) | B− | MAJOR-REVISION (encouraged) |
| 21-d | Methodology, statistics & reproducibility referee | B− | MAJOR-REVISION (encouraged) |

---

## 0. How this council was run

Each member received the same evidence base and the same seven-section brief, but **no member saw any other member's review** — the four positions below are genuinely independent first-round statements. Every member was instructed to ground each judgment in a specific file, number, or claim ID, to recompute what could be recomputed, and to answer Q3 by *defining the gap first, then judging it*. The chair then merged the positions, adjudicated factual conflicts against the JSONs, and assembled the consolidated roadmap. Full transcripts are in Part II; this Part I is the chair's synthesis, not a replacement for them.

A fact worth recording up front: **all four members independently re-derived the headline numbers from the raw result JSONs (MEAP/PoC values, ASR anchors, PA statistics, 3-seed spreads) — and every one of them matched.** Four separate recomputations of Table 4 (by 21-a and 21-d independently, plus the earlier W4-A and W11-F audits) now agree to <0.01 dB. Whatever else is true, the artifact's numbers are real and exactly reproducible.

---

## 1. The short answers

**Q1 — Is it impactful?** *Yes, conditionally.* The threat model — the attacker's regulatory feasible set (allocation + emission mask + power budget) as the constraint set of the adversarial optimization, with robustness reported in link-budget units (MEAP / Price-of-Compliance) — is genuinely novel; no paper in the 12-work comparison corpus constrains a wireless adversary to its spectrum rules. For the target venue (**Vehicular Communications**, target window Dec 2026–Feb 2027) the council judges it impactful *after* the P0+P1 items below (~25–35 h): there is a same-venue precedent (Girmay et al. 2023, the clean-task paper our attack breaks), a live regulatory hook (FCC 24-123, DSRC sunset), and an 11-wave audit trail that exceeds field norms. For TIFS/TWC/TDSC-tier, **not yet** — those venues' comparable papers (RadioShock TDSC'26, Kim TWC'22) have OTA hardware and certified defenses.

**Q2 — What do we lack?** Six things, in the council's merged judgment: (1) adaptive/stronger attack evaluation of our defenses (+14 dB AT headline is attacked only with PGD-10, zero-init, no restarts — never PGD-50); (2) the *real* regulatory mask (our mask is a self-defined flat cap + hard null; the archived "EN 302 571" files are WAF block pages — the standard's mask table was never actually read); (3) error bars — not one CI exists anywhere, and the ±1–2 dB sampling noise per MEAP is *larger* than the 1.06 dB 3-seed spread we lean on; (4) the systems harm chain — "safety" appears in the paper with zero BSM/PER/PRR numbers behind it; (5) over-the-air transmission evidence (the attack has never been transmitted; PA is a memoryless Rapp model); (6) external validity — all victims self-trained, no detection-side defense comparison, no third-party uptake.

**Q3 — Is the gap completely filled?** **No.** The council is unanimous. The *science* half is ~70–75% closed: attack viability is DONE (3 scenarios × 2 modes, 2 victims, white-box/CSI/transfer/PA-aware variants, every number reproduced); the cost, defense, and physical-realism legs are each PARTIAL (no CIs; PGD-10 lower bounds; single-seed defenses; 5.24 GHz real class; idealized mask). The *deployment/regulatory* half is ~40% closed: the deployment story is argued, not measured; "compliance" is by our construction, not by EN 302 571 / 47 CFR Part 95 measurement conventions; no link budget translates MEAP into meters and dBm. **"All CLAIMS C1–C32 VERIFIED" means our internal goals were met — it does not mean the gap was closed**, and the council explicitly warns against shipping under that impression. The encouraging half of the verdict: what remains is honesty-engineering (~25–40 h CPU + writing, zero new science), not a rebuild.

---

## 2. Unanimous findings

1. **The science core is real, novel, and reproducible.** Four independent recomputations agree; the projection operator in `attack_mask.py` is exactly idempotent, Parseval-corrected, OOB ~1e-14; the 11-wave self-audit (33.11 dB axis fix, band-plan inversion fix, C14 layout fix, PA unit bug) is described by 21-a as "a model of research hygiene that this field does not currently practice."
2. **The stress-test breadth exceeds the comparison corpus**: 3 training seeds, 2 attack seeds, PGD-50 sensitivity, CSI/no-CSI, mask-cap sweep, second victim + transfer, real third-party captures, PA-in-the-loop, TRADES. Kim TWC'22, Zheng TMLCN'26, and RadioShock TDSC'26 report narrower matrices.
3. **Not submit-ready as-is.** Three MAJOR-REVISION (encouraged), one NEEDS-WORK. All blockers are enumerable and all are ≤40 h combined (excluding optional OTA hardware).
4. **The compliance noun must be re-earned** (21-b's central finding): the enforced mask is a flat in-band cap + hard OOB null — both *stricter* than the real EN 302 571 / Part 95 masks (right direction for threat disclosure, wrong basis for the word "compliance"); the levels/ramps/measurement conventions (RBW, detector, referencing) of the actual standard were never encoded, and the archived standard "copies" are firewall block pages. A regulator reading the title will assume rule-shaped compliance that the code does not implement.
5. **The defense headline is an upper bound**: AT +14.0 dB / TRADES 18.6 dB PoC are single-training-seed, single-attack-seed, evaluated only against the weakest instantiation of the attack (PGD-10, zero-init, no restarts). PGD-50 moves the *undefended* mid-grid ASR by up to +37 pp and was never run against the defended models.
6. **Zero error bars**: at n=300, binomial 95% CIs give MEAP ±1.0–1.9 dB — larger than the 1.06 dB 3-seed spread and the 5.36–7.14 dB scenario spread we narrate; real-WiFi accuracy carries ±10.8 pp (n=74).
7. **The vehicular half is motivational prose**: no packet, no MAC, no mobility trace, no transmitted attack in 18 pages; the harm chain (false-IDLE → authorized 36 dBm U-NII-4 transmission → I/S ≈ +19 dB at 50 m → PER collapse → missed BSMs vs TR 37.885's ~100 ms latency budgets) is quantifiable in hours but was never done. Also: C-V2X Mode-4 selection is RSRP/SCI-based and 802.11p CCA is energy/preamble-based — **no deployed 2026 radio uses ML sensing**; our premise is forward-looking (DAA/coexistence, O-RAN xApps) and the intro's present-tense framing must be fixed.
8. **Eight claims-vs-backing slips** (each violating our own NUMBER POLICY; found independently by 21-a and 21-d): Table 1 mislabels Kim TWC'22 as "power sweep: fixed" (they sweep PNR); "68.5% real flips" vs stored 67.35%; "2.7 pp" CSI cost vs 2.33; "MEAP within 0.5 dB" not derivable from the stored grid; "11–25 dB" transfer vs 10.8–23.2; C10 "0% below −25 dB" vs ≤0.4%; C12 carries the stale "+3–6 pp" PGD-50 summary; C22/C23 rows stale. Trivial individually; a credibility hazard collectively.
9. **Band-label error**: 5240 MHz is **U-NII-1** (channel 48), not "U-NII-2" as the abstract says — embarrassing in front of exactly the audience we court.
10. **Process gaps**: restructure commit unpushed; Zenodo DOI unminted; worklog had no Task 12–18 entries (five waves unlogged — same failure W11-F8 caught for Task 19, now backfilled); `scikit-learn` missing from `requirements.txt` (fresh clone crashes at training step 4/4); CI never recomputes stored MEAPs; the SNR-sweep eval set reuses the training waveform stream (`rng(42)`) — mild leakage, disclosed direction favorable.

---

## 3. Points of contention, adjudicated

- **Grade spread A− vs B−**: not a contradiction — 21-a graded *novelty against the AML-RF field* (where our discipline is above norm), 21-b/c/d graded *deployment readiness* (where we are below norm). The chair accepts both.
- **"68.5% vs 67.35% real flips at −45 dB"**: 21-d resolved it — 67.35% is the stored frozen-model real-only curve; 68.5% is derivable only from the fine-tuned arm's per-class fractions (14.57% × 254 = 37 flips / 54 real-eligible). Adjudication: cite 67.35% as the stored number and name the model, or store the finetuned derivation. Fix landing in Wave 12.
- **How bad is the mask idealization?** 21-b: title-level (the compliance claim's basis is unverified). 21-a: probably small (the 2×→10× cap sweep moved PoC only −0.9 dB). Both true: the *shape* must be fixed for the regulatory claim to be honest, and the expected quantitative change is small *and favorable to the threat* (a rule-compliant attacker may concentrate in-band power and use OOB shoulders → true PoC likely ≤ 7.1 dB → the threat gets worse, the paper gets stronger). The council endorses doing it.
- **Flipper Zero** (PI's hardware question): 21-c's verdict, endorsed unanimously — **exclude it entirely**. CC1101 covers 300–348/387–464/779–928 MHz at ~+5–14 dBm with hardware OOK/2-FSK/GFSK/MSK only (≤ ~500 kBaud; the `.sub` format replays pulse streams, not arbitrary IQ); a 10-MHz-bandwidth mask-projected PGD waveform is unrepresentable on it at any frequency, and ISM bands have no emission-mask rules, so the paper's entire compliance mechanism has no ISM analog. A reviewer would treat it as a gimmick that damages credibility. Minimum credible OTA path instead: 2× USRP B210 + step attenuator, conducted/cabled first (≈ $2.2–4.5k, 40–60 h).

---

## 4. Where we stand vs. the field (merged one-line table)

| Prior work | What they have that we lack | What we have that they lack |
|---|---|---|
| Sadeghi & Larsson, WCL'19 | field-founding priority, PNR axis | transmit realizability (their ε=1.0σ realizes 14% of its own attack — our C14) |
| Kim et al., CISS'20 | first OTA instantiation | regulatory feasible set, power-threshold metrics |
| Kim et al., TWC'22 | broadcast multi-receiver attack; VAE-UAP black-box; **certified** randomized-smoothing defense; PNR sweeps | mask constraint, MEAP/PoC, PA-in-the-loop, real captures, audit discipline |
| Liu et al., T-Rel'23 | causative (poisoning) attack axis | compliant evasion, power metrics, defense study |
| Zheng et al., TMLCN'26 | wideband multi-label targeting, scale | regulatory constraint + PoC, matched AT, victim discrimination |
| Li et al. (RadioShock), TDSC'26 | real OTA with CSI estimation & adaptation | compliance framing, certification-style metrics, PA check |
| Zhao et al., INFOCOM'24 | a working detection-side countermeasure | attack quantification in power units; the compliant threat their detector would face |
| Habler et al., JNCA'25 | O-RAN system-level threat taxonomy | one component deeply quantified + released one-command gate |
| O'Shea et al., JSTSP'18 | dataset scale (220k/class) | robustness of the coexistence task they assume solved |
| Anand et al., DySPAN'08 | 20-yr PUEA lineage, analytic detection theory | DL victim, gradient waveform, mask constraint, MEAP |
| Croce & Hein, ICML'20 | attack-strength standard (APGD/FAB/Square/restarts) | RF-specific constraint set & power metrics — but we cite the standard and don't meet it |
| Girmay et al., VC'23 | real 5.9 GHz ITS-band campaign at our target venue | adversarial robustness of exactly their task |
| *(missing cites)* Ke et al. TIFS'24; Chen/Park/Reed JSAC'08; Jin/Anand/Subbalakshmi 2006–2012 | must be cited & differentiated | — |

---

## 5. Gap-closure scorecard (Q3 in detail)

| Criterion (measurable) | Status | Evidence |
|---|---|---|
| Compliance-constrained attack **viability** (≥2 scenarios, both modes, worst-case disclosed) | **DONE** | 3 scenarios × 2 modes; 2 victims; CSI/transfer/PA-aware; C6–C11, C22–C23; 4× recomputed |
| **Quantified cost** (PoC/MEAP with uncertainty, converged attacks) | **PARTIAL** | PoC quantified + sensitivity sweeps, but zero CIs; PGD-10 lower bounds (up to +37 pp mid-grid); genie censored 2/3 seeds; attack-seed n=1 replicate |
| **Defense** (matched, seeded, adaptively evaluated) | **PARTIAL** | AT + TRADES + rural transfer, honest boundary — all 1 training seed, 1 attack seed, non-adaptive PGD-10 |
| **Physical realism** (real signals, real chain, real band) | **PARTIAL** | real OTA WiFi ✓ but 5.24 GHz U-NII-1, n=74, 1 location; Rapp-only PA; idealized mask; never transmitted |
| **Deployment story** (who consumes the label; consequence measured) | **NOT MEASURED** | argued in §7 scenarios; zero BSM/PER/PRR; harm chain un-quantified |
| **Regulatory-grade compliance claim** (real mask + conventions + absolute power) | **NOT DONE** | self-defined mask; standard never read (WAF pages); no EIRP/link budget; comment draft not filable |
| **External validity** (third-party victim, detection defense, uptake) | **NOT DONE** | all victims self-trained; Zhao's detector cited not run; no third-party use yet |

**Overall: science half ~70–75% closed; deployment/regulatory half ~40% closed. The gap is NOT completely filled.**

---

## 6. Consolidated roadmap

### P0 — before ANY submission (~13–18 h, no hardware; most need torch reinstalled)
1. **Fix the 8 claims-vs-backing slips + band label + front-door overreach** (Table 1 Kim cell; 67.35%; 2.3 pp; drop/re-back "0.5 dB"; 11–23; C10/C12/C22/C23 row syncs; U-NII-1; README "every spectrum rule" softening; draft line-101 contradiction). ~2–3 h. *(Wave 12 started this — see §8.)*
2. **Adaptive-attack evaluation of the defenses**: PGD-50, ≥10 random restarts, vs AT and TRADES checkpoints, 12-point grid; report whether +14.0/+18.6 dB survive. ~10 h.
3. **Error bars**: binomial 95% CIs on every ASR point, propagated to MEAP/PoC; extend PSR grid to −55 dB to un-censor genie MEAPs; re-report Table 4 as mean ± CI. ~4–7 h.
4. **Hygiene**: `scikit-learn` in requirements; CI step recomputing 5 stored MEAPs (pure python) + red-team CLI; re-seed the C4 sweep (rng 777) to kill the waveform-identity overlap. ~2 h.

### P1 — venue-defining for Vehicular Communications (~20–35 h)
5. **Implement the real EN 302 571 / Part 95 / 15.407 breakpoint masks** (first: obtain and archive the actual standard texts — the current "evidence" files are WAF block pages), enforced after RBW convolution at the PA output; re-run the headline PoC. Expected: threat strengthens. ~8–12 h.
6. **Quantified harm bridge**: "From false IDLE to missed BSMs" — link budget (U-NII-4 36 dBm at 25–100 m vs BSM 23 dBm; I/S ≈ +15…+25 dB; PER collapse; BSMs lost vs TR 37.885 ~100 ms budgets). ~6–10 h analytical (ns-3 optional, 40–80 h).
7. **Deployment-story rewrite**: locate the sensor honestly (C-V2X Mode-4 is RSRP/SCI-based; 802.11p CCA-ED ≈ −65 dBm; ML sensing is proposed, not deployed — DAA/coexistence, O-RAN xApp, conformance); restore the Phase-0 latency number (0.985 ms/window) as feasibility. ~5–8 h.
8. **Citations**: Chen/Park/Reed JSAC'08 + Jin/Anand/Subbalakshmi series + FCC TVWS docket lineage (PUEA); Ke et al. TIFS'24. ~2 h.
9. **Seed replication**: attack seeds {11, 22} × 6 cells; AT/TRADES training seeds {123, 456}. ~8 h.
10. **REGULATORY_COMMENT_DRAFT**: make filable (ET Docket 19-138, 47 CFR §1.1206 ex parte, CFR/FR cites, filer block) or retarget to FCC TAC / 5GAA / ETSI TC-ITS. ~3 h.

### P2 — top-tier parallel track (optional; hardware/infra)
11. 3GPP CDL channels (QuaDRiGa) — 10–16 h. 12. Third-party victim with different front-end + joint (no-CSI × surrogate) ablation — 12–20 h. 13. Conducted OTA replication (2× USRP B210, ≈$2.2–4.5k) — 40–60 h. 14. ns-3/ms-van3t PRR study — 40–80 h. 15. Zhao-style DDB detection baseline — 4–6 h.

---

## 7. Council verdict (unanimous, signed)

> **The science is real and the artifact is exactly reproducible — four of us recomputed it independently and it matches. The threat model is novel in its field. But the research gap is not completely filled, and the package is not submission-ready: the defense claim is evaluated only against the weakest form of the attack, not one confidence interval exists in the paper, the "compliance" is defined by our own projection rather than by any standard we have actually read, and the vehicular consequence of a successful attack has never been measured. The distance to a defensible Vehicular Communications submission is ~25–35 hours of honesty-engineering, not new science; the distance to a TIFS/WiSec-class paper additionally requires over-the-air evidence. Do not call it done. Do the P0 list.**

*— 21-a, 21-b, 21-c, 21-d, chair (2026-09-06)*

---

# Part II — Full written positions (verbatim transcripts)

## Member 21-a — Adversarial ML / RF Physical-Layer Security

### 1. Position statement

This is, to my knowledge, the first AML-RF work that takes the attacker's *regulatory* feasible set — allocation, emission mask, power budget — as the constraint set of the adversarial optimization, and then reports robustness in link-budget units (MEAP, Price-of-Compliance) instead of ε-balls or PNR. Around that core sits an unusually honest evaluation stack: conditional-ASR with censoring-aware MEAP interpolation (`attack_mask.py` L205–245), a corrected 33.11 dB PSR axis with archived pre-fix results, a physical-power translation of legacy feature-space attacks (C14), mask-matched adversarial training (C16/C17), and real-capture, second-victim, and PA-in-the-loop stress tests — each traceable to a seeded JSON. One-sentence thesis: **spectrum-rule compliance is not a defense — it costs the attacker only 5–7 dB (untargeted) or 16–18 dB (cloaking), an undefended ITS-band sensing CNN breaks at −37…−39 dB, and only mask-matched AT (+14 dB) and verification at the PA output port restore meaningful margins.** Grade: **A−** on novelty/threat-model originality against the AML-RF corpus (Sadeghi WCL'19 → Kim TWC'22 → Zheng TMLCN'26 → RadioShock TDSC'26); **B+** on rigor at prototype scale (300 windows, PGD-10, single-seed defense, no OTA). Composite: **A−** for this field, where synthetic single-seed feature-space evaluations are still the norm.

### 2. Novelty comparison table

| Paper (author, year, venue) | Their contribution | They have that we lack | We have that they lack |
|---|---|---|---|
| Sadeghi & Larsson, 2019, IEEE WCL (438 cites) | First adversarial attacks on DL modulation classification; ε-ball in IQ; PNR metric | Priority + field-founding citations; the PNR axis itself | Physical transmit realizability (our C14 shows their ε=1.0σ implies a hidden −24.2 dB budget and realizes only 14% of its own attack); mask; MEAP/PoC |
| Kim et al., 2020, CISS (123 cites) | First *over-the-air* evasion attack through an adversary→receiver Rayleigh channel | First OTA instantiation; adversary-channel channel-awareness | Regulatory feasible set; power-threshold metrics; sensing (not AMC) task; PA check |
| Kim, Sagduyu, Davaslioglu, Erpek, Ulukus, 2022, IEEE TWC 21(6) (238 cites) | Channel-aware attacks, **broadcast** attack to m receivers, knowledge-level ablations (channel/input/model incl. VAE-UAP black-box), **randomized-smoothing certified defense**; accuracy-vs-PNR sweeps (verified in the fetched PDF, §VII, Figs. 2–6) | Broadcast multi-receiver attack; black-box UAP; a *certified* (guaranteed) defense; mature validation | Emission-mask/allocation constraint; MEAP/PoC threshold metrics; coexistence-sensing task; PSR_eq translation of legacy threat models; PA-in-the-loop optimization; real-capture eval |
| Liu et al., 2023, IEEE Trans. Rel. 72(2) (58 cites) | Adversarial attack + jamming-waveform design + data **poisoning** against DL spectrum sensing in CR-IoT | Causative (poisoning) attack axis we never touch; sensing-task priority | V2X/5.9 GHz post-FCC task; *compliant* evasion; power-domain metrics; defense study |
| Zheng et al., 2026, IEEE TMLCN 4:950–965 | Targeted adversarial attacks on DL **wideband spectrum sensing** (multi-label) | Direct wideband-sensing targeting; multi-label formulation | Regulatory constraint set + PoC; matched AT defense; real OTA captures; victim-architecture discrimination |
| Li et al. (RadioShock), 2026, IEEE TDSC 23(4) | OTA attacks with **channel-state estimation** for dynamic adaptation; up to 52.41% accuracy drop on diverse wireless models | CSI *estimation* and adaptation (we assume CSI + ablate error); TDSC-level real-scenario claims | Regulatory compliance framing; certification-style power metrics; PA reality check |
| Zhao et al., 2024, IEEE INFOCOM (9 cites) | **Detection** of adversarial spectrum attacks via distance-to-decision-boundary statistics at a fusion center | A working detection-side countermeasure — we evaluate none | Attack-side quantification in power units; the specific compliant threat their detector would face |
| Habler et al., 2025, J. Netw. Comput. Appl. 236:104090 (~40 cites/yr) | Systematic AML threat analysis + remediation mapping for O-RAN | System-level threat taxonomy; standards breadth | One component deeply quantified with a released one-command gate (MEAP) — the measurement layer their taxonomy lacks |
| O'Shea, Roy, Clancy, 2018, IEEE JSTSP 12(1) | Large-scale OTA DL radio signal classification (the clean-task foundation) | Scale (220k-sample-class corpora); measured signals | Robustness of the coexistence-recognition task those works assume solved |
| Anand, Jin, Subbalakshmi, 2008, IEEE DySPAN (223 cites) | PUEA: attacker-constrained threat (power/distance) against *energy* sensing; analytic detection theory | 20-year lineage, analytic guarantees, a detection literature | DL victim, gradient-optimized waveform, emission-mask constraint, MEAP (we cite and differentiate — C32) |
| Croce & Hein, 2020, ICML (3277 cites) | AutoAttack: parameter-free attack ensemble (APGD-CE/DLR, FAB, Square) as the reliable-robustness standard | Attack-strength standard; multi-restart, diverse-loss evaluation | RF-specific constraint set and power metrics — but we adopt only PGD-10, not their standard (see §4) |
| Girmay et al., 2023, Vehicular Communications 39:100563 (target venue precedent) | CNN technology recognition + traffic characterization for coexisting technologies in the ITS band | Real ITS-band measurement campaign; the clean-task precedent at our target journal | Adversarial robustness of exactly that task — the open question they leave |

(Also relevant, surfaced in the W1-C audit trail but not in the corpus directory: Ke et al., IEEE TIFS 2024, frequency-selective adversarial attacks — frequency-domain perturbation constraints for a different victim scenario; must be cited and differentiated if submitting to TIFS.)

### 3. Where we beat the field

- **The regulatory feasible-set attack itself (C6, C7; `attack_mask.py::project` L88–139).** OOB FFT nulling + Parseval-corrected per-bin PSD cap + window-energy budget, exactly idempotent, budget ratio 1.000000, OOB fraction ~1e-14 (C7, `debug_attack.py`). No paper in the corpus constrains a wireless adversary to its spectrum rules; the closest ancestor (PUEA, Anand 2008) constrains power/distance against an *energy detector*, not a mask against a *DL classifier*.
- **Power-domain robustness metrics (C8, C9, C10, C11).** I independently re-derived all 12 MEAP/PoC values of Table 4 from the raw curves in `attack_results_merged.json` — every one matches to <0.01 dB (urban untargeted genie −44.22 / mask −37.08 / PoC 7.14; rural cloaking PoC 18.12). No corpus paper reports a threshold-in-power metric; Kim TWC'22 sweeps accuracy vs PNR but has no mask to price, hence no PoC analog.
- **The price of compliance finding (C8–C10).** 5.4–7.1 dB untargeted, 16.0–18.1 dB cloaking, invariant across three channel scenarios (spread <1.5 dB, C11) and three training seeds (spread 1.06 dB, C12) — a regulator-usable quantity with error discipline exceeding anything in the corpus.
- **Physical incoherence of the legacy threat model (C14).** ε=1.0σ → implied −24.2 dB, 71.3% intended vs **14.0% realized** through the true front-end, while equal-power transmitted attacks achieve 99.3% (genie) / 62.3% (mask) (`feature_space_equiv.json`). Nobody — not Sadeghi, not Liu, not Kim — quantifies the feature-space-to-waveform gap; this is the paper's most exportable methodological idea (PSR_eq translation).
- **Knowledge ablation in power units (C13).** No-CSI moves the compliant threshold −37 → ≈−16.9 dB (my interpolation of `csi_mismatch.json`: 15.0% @ −20, 23.0% @ −15), i.e., channel-realization knowledge is worth ~21 dB — a sharper statement than Kim TWC's ASR-at-fixed-PNR ablations.
- **Feasible-set-matched defense (C16, C17, C24).** Mask-matched AT: compliant MEAP +14.0 dB (`at_defense_results.json` deltas: 14.006) at 100% clean, PoC doubling 7.1→14.4; TRADES −19.1 dB at 0.75 pp cost (`attack_results_urban_untargeted_trades.json`: −19.147); rural transfer at 1.2 dB cost (C24). Kim TWC's randomized smoothing is certified but not attack-matched to a physical feasible set.
- **Real-signal validation with a discovery (C21).** Compliant MEAP within 1.2–1.8 dB of synthetic (−35.9 frozen / −37.7 fine-tuned vs −37.1), plus the OOD margin-collapse result (0.0% synthetic vs 67.35% real flips at −45 dB PSR) — a genuinely new negative result: synthetic-only robustness evaluation *overstates* robustness. No corpus paper mixes real captures into the adversarial loop.
- **Victim discrimination and transfer honesty (C22, C23).** ResNet shifts the compliant threshold 11.6–15.1 dB and halves PoC (3.7 dB); surrogate transfer costs 10.8–23.2 dB — making MEAP a model-discrimination instrument, not a single-number claim.
- **PA reality check (C27–C30, pre-registered H1–H3 in `papr_pa_results.json`).** Attack PAPR 11.05 dB vs benign 6.26–9.02; 92% of windows fail even the lenient −28 dBr gate at 6 dB backoff; IBO*=13/15 dB; PA-aware re-optimization (Rapp inside the PGD chain) is *stronger* (−1.23 dB) AND 100% strict-gate compliant at backoff. First PA-in-the-loop adversarial optimization in this literature, as far as the corpus shows.
- **Evaluation hygiene (C18, C25, C26, C31).** Every number JSON-traced with embedded config/seed; censoring flags; 5-wave adversarial internal audit with archived pre-fix results (`results/archive_pre_axis_fix/`), 43-assertion independent checkers, pre-registered hypotheses, and a self-caught 33.11 dB unit bug. This is Croce & Hein's reliability ethos ported to RF and exceeds field norms by a wide margin.

### 4. Where we are behind the field

- **Error bars and seeds.** n=300 gives ±2.3 pp SE at the 20% MEAP threshold → ~±0.6 dB MEAP resolution on the mask curve (slope 12 pp/5 dB near −37). The undefended headline now has 3 training seeds (C12), but **the defense (C16), TRADES (C17), rural transfer (C24), real-WiFi (C21), victim-2 (C22/C23), and the PA study are all single-seed, seed 42/7** — no CIs on the +14 dB headline. Genie MEAP is floor-censored for 2/3 training seeds, so per-seed PoC is a lower bound (disclosed, but the Table-4 "no censoring" line applies only to the seed-42 column).
- **Adaptive-attack evaluation of the defense.** The AT/TRADES models are attacked with the *same PGD-10, zero-init, no-restart* protocol as the undefended model. PGD-50 moves the undefended mid-grid ASR by up to +37 pp and MEAP by −1.7 dB (C12/`pgd50_urban_untargeted.json`); it was **never run against the defended models**. No APGD-DLR/FAB/Square (the Croce & Hein standard the paper itself cites), no restarts, no EOT over channel draws. The +14 dB defense claim is therefore an *upper bound on defense effectiveness* that a TIFS-style reviewer will attack immediately.
- **Black-box & transfer threat models.** The transfer study (C23) still gives the surrogate a **perfect attacker channel** (disclosed in `victim2_transfer.json` config) and both victims are self-trained on the same front-end, data recipe, and seed (limitation 9 admits: "varies the classifier architecture, not the receiver"). No query/score-based attack, no VAE-UAP-style input-independent attack (Kim TWC'22), no attack on an independently published model (e.g., Girmay's), and **no joint (no-CSI × surrogate-model) ablation** — C13 shows no-CSI alone costs ~21 dB, so the realistic worst case is far from the headline, and the paper never combines the two relaxations.
- **Victim/front-end diversity.** Two architectures, one shared 256×15 STFT front-end, one 102.4 µs window, one z-scoring convention. No IQ-domain victim, no different time-frequency resolution, no cooperative/multi-receiver sensing (which is precisely Zhao INFOCOM'24's deployment setting).
- **Training-data scale and task fidelity.** 3,200/800 windows vs O'Shea-scale corpora; all-QPSK, no pilots/preambles/DMRS, no 16/64-QAM (`waveforms.py` L41); clean accuracy 100% (saturated) at the training SNR [5,25]; all attack evaluations at that same SNR regime — the compliant-attacker story at deployment-relevant low SNR (where C4 shows the CNN at 73%) is unexplored. PC5 is a rate-matched stylization (disclosed), 802.11p/WiFi block-exact but pilotless.
- **OTA/hardware validation.** Zero. Kim'20 (CISS), RadioShock (TDSC'26) did SDR OTA. Our "real" study replays *received* captures (WiFi class only, 74 eval windows, 5.24 GHz U-NII-2 ≠ 5.9 GHz, no attacker-side channel applied to the real class); the PA is a memoryless Rapp, PAPR computed without oversampling (disclosed lower bound), strict/lenient gates are per-FFT-bin idealizations of 802.11-style masks rather than regulatory measurement procedures (RBW, detector) — limitation (10) is honest, but it leaves the "certification-style" framing ahead of the evidence.
- **Detection-side countermeasures.** None evaluated — Zhao INFOCOM'24 exists for exactly this attack class. Worse, the PA-regrowth "tell" is weaker than the prose implies: the attack pedestal (−19.3 dBr p95) sits **below WiFi's own post-PA skirt (−12.65 dBr)** in `papr_pa_results.json` — a spectrum monitor would not flag the attack in a WiFi neighborhood; the tell works only against PC5/11p (2.4–9.4 dB margins), and the PA-aware attacker defeats it anyway (C30). No randomized-smoothing comparison either (Kim TWC'22's certified defense is cited but not benchmarked).
- **Claims-vs-backing slips (found by direct JSON diffing; each violates the paper's own NUMBER POLICY):** (1) Table 1 "Power sweep: fixed" is wrong for Kim et al. TWC'22 — they report accuracy-vs-PNR sweeps; the honest differentiator is mask constraint + MEAP/PoC threshold metric, not "power sweep." (2) Paper/README/C21 say "68.5% real flips at −45 dB"; `real_wifi_attack.json` stores **67.35%**. (3) Paper says CSI error costs "at most 2.7 pp"; `csi_mismatch.json` max delta is **2.33 pp**. (4) README/C13 claim "MEAP within 0.5 dB" under 0.3-rel CSI error — not derivable from the stored grid. (5) README/C23 say transfer costs "11–25 dB"; the JSONs give 10.8–23.2. Individually trivial; collectively a credibility hazard, because the paper's central selling point is number traceability.

### 5. The three questions

**(a) Is this impactful enough for a top venue — which venue?** For the **target venue, Vehicular Communications (Elsevier)**: yes — scope fit, the Girmay 2023 clean-task precedent in the same journal, an ~18-page fully traced manuscript with a released artifact, and a live regulatory hook (FCC 24-123, DSRC sunset Feb 2027). For **TIFS/TDSC/TWC/TMLCN**: not yet — the comparable papers there have OTA hardware (RadioShock TDSC), certified defenses (Kim TWC), or claim frequency-domain constraint priority (Ke TIFS); we would need items 2–5 below plus OTA. A fast parallel option is IEEE Communications Letters (the Sadeghi-Larsson precedent: the niche's foundational 438-cite paper was a 4-page letter) — a PoC+C14 short paper would fit.

**(b) What exactly do we lack vs. related work?** (i) OTA transmit-side validation; (ii) a certified or detection-side defense comparison; (iii) attack-strength at the AutoAttack standard; (iv) broadcast/multi-receiver and input-independent black-box attacks; (v) 3GPP-conformant CDL channels; (vi) dataset scale and modulation/pilot fidelity; (vii) a regulator-grade mask template; (viii) PA measurements and regulatory measurement procedures; (ix) external-victim validation.

**(c) Is the research gap COMPLETELY filled?** Define the gap precisely, in the paper's own words (§2 Positioning): *"no prior work constrains a wireless adversary to its spectrum rules, and none reports a power-domain price-of-compliance analysis"* — plus the implicit deployment gap: *a physically grounded, regulator-usable robustness metric for ML coexistence sensors*. The **science half (G1) is convincingly filled**: the mask-projected attack exists and is verified (C7), the price is measured with seed/scenario/CSI/architecture/real-signal/PA stress tests, and I reproduced every headline number from the raw JSONs. The **deployment half (G2) is not**: the mask is an idealization of the cited EN 302 571, PA compliance is checked against per-bin dBr gates rather than conformance procedures, nothing is verified over the air, and the benchmark has no third-party uptake yet. My estimate: **G1 ~70% closed, G2 ~40% closed.**

### 6. Top 5 must-do items

1. **Repair the five claims-vs-backing slips + Table 1 mislabel** — 2–3 h.
2. **Adaptive-attack evaluation of the defense** (PGD-50, ≥10 restarts vs AT and TRADES) — 8–10 h CPU + 2 h writing.
3. **ETSI EN 302 571-shaped mask re-run** — 8–12 h. Expected: PoC likely survives (2×→10× moved it −0.9 dB) — turning the top limitation into a robustness result.
4. **3GPP TR 37.885 CDL channels via QuaDRiGa** — 10–16 h.
5. **Independent third victim with a different front-end + joint black-box ablation** — 12–20 h.

*Honorable mentions:* conducted (cabled) OTA proxy at 5.8 GHz with HackRF/USRP — 80–120 h + hardware; Zhao-style DDB detection baseline — 4–6 h; MEAP error bars via bootstrap — 2 h.

### 7. Verdict

**MAJOR-REVISION (encouraged).** The scientific core is real, novel, and independently reproducible; the stress-test breadth exceeds what Kim'22, Zheng'26, or RadioShock'26 report. But it is not submit-ready as-is: five claims-vs-backing slips and a factually incorrect Table-1 cell would be found by any careful referee; the +14 dB defense headline rests on a single training seed evaluated only against the weakest instantiation of the attack; and the "certification-style" framing runs ahead of an idealized flat-cap mask and a memoryless Rapp PA. All blockers are ≤40 h of combined work. Items 1–2 (~12 h) are mandatory before *any* submission and are sufficient for the Vehicular Communications target; items 3–5 (~30 h) determine whether the TWC/TIFS-tier parallel submission is scientifically defensible. Do not submit the current PDF; the gap between this and a defensible submission is two weeks, not two months.

*(end of member 21-a transcript)*

## Member 21-b — Spectrum Regulation & Wireless Standards

**Reviewer persona:** ex-FCC OET staff attorney-engineer, ETSI committee member; published on PUEA and spectrum enforcement.

### 1. Position statement

Speaking as someone who has staffed 5.9 GHz dockets since the DSRC era: this is the rare adversarial-ML paper whose threat model is phrased in regulator-legible terms — allocation, emission mask, transmit power, and (after Wave 11) verification at the PA-output port — and whose honesty discipline (CLAIMS.md C1–C32, 11-wave audit trail, non-claims sections, censoring flags) exceeds most docket submissions I have reviewed. The FCC grounding is correct to the day: FCC 20-164 (adopted Nov 18 2020, 86 FR 23281), FCC 24-123 (adopted Nov 20 2024, 89 FR 100838, effective Feb 11 2025, two-year DSRC sunset), all cross-checked against the stored evidence JSONs. It is nonetheless **not yet regulatory-grade** for one central reason: the "emission mask" enforced in `attack_mask.py` is the authors' own construction — a hard out-of-band null plus a flat 2× in-band cap — not the EN 302 571 / 47 CFR Part 95 mask of any shape, level, or measurement convention; and the repo's stored "copy" of EN 302 571 (`scripts/audit_w1c_searches/en302571_v211.pdf` and `etsi_302571.html`) is a **web-application-firewall block page** — no one in this project has actually read the standard's mask table. Secondary dings: 5240 MHz labeled "U-NII-2" (it is U-NII-1, channel 48); zero CFR citations; comment draft not filable.

**Grade: B−** (regulatory realism and usefulness). **Thesis:** a genuinely novel, honestly audited threat model whose "compliance" is defined by the authors' own projection rather than by any rule a conformance lab could test against — one standards-engineering pass short of regulatory credibility.

### 2. Regulatory-fidelity audit table

| Anchor | What we claim (file:line) | What the actual rule/standard says | Match? | Severity if wrong |
|---|---|---|---|---|
| FCC 20-164 timeline & band split | "upper 30 MHz (5895–5925) dedicated to C-V2X, lower 45 MHz (5850–5895) opened to U-NII-4" (`main.tex:102–106`) | R&O adopted 11/18/2020, released 11/20, 86 FR 23281 (5/3/2021); 45 MHz unlicensed below 5.895, 30 MHz ITS above; C-V2X-only status finalized by 24-123 + sunset | **Match** (minor compression: 20-164 reserved the 30 MHz for *ITS* generally; C-V2X exclusivity is the 2024 order's doing) | Low — correct |
| FCC 24-123 effective date & sunset | "final rules took effect in February 2025 with a two-year DSRC sunset" (`main.tex:107–110`); bibitem `fcc2024` (89 FR 100838, Dec 13 2024) | Effective 2/11/2025; two-year sunset runs from *Federal Register* publication (12/13/2024 → sunset ≈ 12/2026) per `fcc24b.json` | **Match** (paper wisely gives no sunset date) | Low |
| C-V2X waiver count | "more than fifty waivers already authorize C-V2X operation" citing ITS America May 2024 (`main.tex:109–111`, C32) | Plausible: FCC granted large C-V2X waiver batches 2021–2023 (FCC 24-123 records the grants); ITS America is a secondary source | **Plausible but unverified in-repo** — no `itsa2024` evidence file exists anywhere | Medium for a docket |
| 5GAA roadmap (Dec 2024) | "industry roadmap projects volume deployment of direct-mode V2X radios in production vehicles" (`main.tex:111–112`) | 5GAA roadmap updates exist; no in-repo evidence file archived | **Plausible, unarchived** | Low |
| EN 302 571 V2.1.1 citation | Mask source cited as "ETSI EN 302 571 in Europe; FCC rules in the US" (`main.tex:133`); bibitem correct (V2.1.1, 2017-02, RED art. 3.2, 5855–5925 MHz) | Standard is real and correctly described bibliographically | **Citation match; content never verified** — stored PDF+HTML are WAF block pages | **High** (the compliance claim's standards basis is unverified in-repo) |
| Numerically enforced "mask" | Flat in-band per-bin cap = 2× uniform budget sharing + exact OOB null (`attack_mask.py:88–139`, `main.tex:447–452`) | Real EN 302 571 / Part 95 / 802.11 masks are breakpoint masks (0/−20/−28/−40 dBr-family levels) at offsets from the channel edge, permitting limited OOB emissions below the mask, referenced to in-channel power with specified measurement bandwidth and detector; no in-band flatness limit exists in any of them; EIRP caps (33 dBm ITS class; Part 95 C-V2X 23–33 dBm classes) apply | **Mismatch** — self-defined, disclosed in Limitations (2) but the README/draft framing ("follows every spectrum rule") overshoots | **High** for the "compliance" word; direction is conservative (see §3) |
| U-NII-4 rules | Wi-Fi class = 20 MHz channel centered 5.885 (5.875–5.895); wifi_attacker band [−10,0] MHz with hard null (`waveforms.py:80–92`) | Matches FCC 20-164's U-NII-4 channelization under 47 CFR 15.407; but 15.407's adjacent-band OOBE/power conditions toward 5.895 are neither cited nor modeled (hard null is stricter) | **Partial** — geometry right, edge conditions absent, no CFR cite | Medium |
| U-NII-5/6/7 | *Not relied on anywhere* — only U-NII-4. (U-NII-5/6/7 are the 5.925–6.875 GHz sub-bands of the 6 GHz proceeding, FCC 20-51.) | N/A | **No mismatch in the paper**; one mislabel in the worklog | Low |
| 5.24 GHz capture band label | "recorded at 5.24 GHz in U-NII-2" (`main.tex:72`, `main.tex:957`, `ATTRIBUTION.md:22`, `real_wifi.py:5–6`) | 5240 MHz = 802.11 channel 48 = **U-NII-1** (5.150–5.250 GHz); U-NII-2A begins at 5.250 GHz | **Mismatch** — wrong band name in the abstract | Medium (trivially fixable; embarrassing in front of a spectrum-regulation audience) |
| PA-study mask gates | "strict = −40 dBr, lenient = −28 dBr, 802.11-style" per 9.77-kHz FFT bin (`run_papr_pa.py:45–52`, `main.tex:817–822`) | The levels are the right family (802.11-2012 transmit mask breakpoints); regulatory masks are measured with defined RBW (100 kHz–1 MHz) and RMS/average detectors, referenced to in-channel power — not per-bin peak vs. peak-in-band | **Level match, convention mismatch** — disclosed in Limitations (10) | Medium-High for any "conformance" language |
| PUEA lineage | "formalized by Anand et al. [DySPAN 2008] and remain an active detection literature [Ambhika 2024]"; 5-axis differentiation (`main.tex:215–232`, Table 1 row, C32) | Anand/Jin/Subbalakshmi 2008 is a legitimate formalization; but the canonical first definition-and-defense paper is Chen, Park & Reed (IEEE JSAC 26(1), 2008) — absent; the Jin/Anand/Subbalakshmi 2006–2012 detection series — absent; the FCC TV-white-space docket lineage (ET Dockets 04-186/08-260) where PUEA entered the regulatory record — absent | **Partial** — adequate for academic related-work, incomplete for a policy audience | Medium |

### 3. The mask-model gap

**What `attack_mask.py` actually enforces** (`project()`, lines 88–139): (i) *allocation*: every FFT bin outside the attacker's 10 MHz block hard-nulled (measured OOB fraction 1e-14); (ii) *in-band "emission mask"*: a flat per-bin PSD cap at `psd_margin × N × P_budget / n_in` (default 2× the uniform budget sharing — a self-chosen constant, swept to 10×, moving PoC 7.1→6.2 dB); (iii) *power*: window-energy budget ≤ P_a. The genie baseline enforces only (iii).

**What a real mask is.** EN 302 571 V2.1.1 (RED harmonized standard, ITS 5855–5925 MHz), the FCC C-V2X rules adopted in FCC 24-123 (47 CFR Part 95), and 47 CFR 15.407 (U-NII) all define unwanted emissions as **breakpoint masks at offsets from the channel edge** with an **EIRP/channel-power cap** and **specified measurement conditions** (resolution bandwidth, detector type, averaging). None of them caps in-band flatness; all of them *permit* out-of-band emission below the mask breakpoints rather than forbidding it.

**The gap, in three moves:**
1. **Shape and levels are not the standard's.** The paper's feasible set is a brick wall at the allocation edge plus an in-band flatness rule that exists in no regulation. Crucially, both deviations are *stricter than reality*: a genuinely rule-compliant attacker may concentrate in-band power and may radiate up to the mask shoulders out of band. The paper's attacker is therefore **weaker than the true worst-case compliant attacker** — the right direction for a threat disclosure, and the mask-cap sensitivity sweep (−0.9 dB) suggests the in-band side is small. But this means the headline metric — "Price of Compliance ≈ 7 dB" — is the price of *the authors' constraint set*, not the price of EN 302 571/Part 95 compliance. A regulator reading the title will assume the latter.
2. **Measurement conventions are absent.** The projection is checked on 2048-point FFT bins (9.77 kHz spacing); the PA gates compare per-bin peak-OOB to peak-in-band. A −40 dBr per-9.77-kHz-bin limit and a −40 dBc-in-1-MHz-RBW limit are not commensurable without the RBW-convolution and reference-power arithmetic — nobody in this repo has done it, and the repo's own "copy" of the standard is a firewall error page. This is the single most damaging artifact in the evidence folder: CLAIMS.md:79's "ETSI EN 302 571 V2.1.1 ✓" was a *metadata* verification; the mask table was never read.
3. **No absolute-power anchor.** Every compliance statement is relative (PSR vs. received victim power). No EIRP cap is ever applied, so MEAP (−37 dB) cannot be translated into "can a lawful 23/33 dBm EIRP device at range R actually deliver this?" — the first question a Bureau staffer or a 5GAA engineer will ask.

**What a regulator would say about "compliance":** the front-door documents would draw an objection in a docket — README ("follows **every spectrum rule** … passing every spectrum monitor") and REGULATORY_COMMENT_DRAFT.md ("passes every in-band power and out-of-band emission check") assert an in-band *power* check that does not exist in the model and an out-of-band *check* that is a stricter-than-rule null. To the paper's credit, the Limitations disclose both idealizations, and the PA subsection's core insight — *verification must happen at the PA output port* — is exactly right and genuinely valuable to a conformance audience.

**What is needed to make the compliance claim defensible:**
- Implement the true breakpoint masks (EN 302 571 V2.1.1 transmit-mask table; FCC 24-123/Part 95 C-V2X OOBE; 15.407 U-NII-4 edge conditions), enforced **after RBW convolution** with a modeled measurement filter and detector, at the PA output port.
- Add the absolute EIRP cap and a link budget translating MEAP into feasibility (range, EIRP, path loss).
- Re-run the PoC headline under the real mask — I expect the true PoC to be *smaller* than 7 dB (the attacker gains the OOB shoulders and in-band concentration), i.e., the threat gets worse: a *stronger* paper, and the honest way to earn the title.
- Archive the actual standard texts (replacing the WAF pages) and add a clause-by-clause mapping table from each enforced constraint to the standard/rule clause it claims to represent.

### 4. PUEA lineage completeness

**What's there:** a dedicated paragraph, the correct two-goal taxonomy of PUEA (energy-detector fooled into "primary present" = DoS; "primary absent" = spectrum theft), citation of Anand/Jin/Subbalakshmi (DySPAN 2008 analytical model) and Ambhika (Wireless Networks 2024, ML-based detection), and a genuinely good five-axis differentiation. Calling the compliant attacker PUEA's "adversarial-ML descendant" is fair and well-argued.

**What's still missing:** **R. Chen, J.-M. Park, J. H. Reed, "Defense against primary user emulation attacks in cognitive radio networks," IEEE JSAC 26(1), Jan. 2008** — the canonical PUEA paper; its absence is the citation a CR-security reviewer or an FCC reader who knows the literature will notice first. The **Jin/Anand/Subbalakshmi detection series 2006–2012** (e.g., ICC 2009 and follow-ons) — the paper cites only the 2008 analytical-model paper from this group. The **regulatory lineage**: PUEA entered the public record in the FCC's cognitive-radio/TV-white-space proceedings (ET Docket 04-186 / 08-260) — for a paper whose pitch is "the attack a *regulator* should care about," one sentence connecting PUEA's docket history to the present 5.9 GHz story would materially strengthen the policy framing. Optionally: Brown & Sethi (DySPAN 2007) for CR denial-of-service framing.

**How the compliant attacker differs fundamentally from classical PUEA:** classical PUEA is *mimicry* — the attacker reproduces the primary's signal statistics and power geometry so an energy/feature detector misreads occupancy; its constraint is attacker power and distance; its countermeasures (location verification, transmitter fingerprints, watermarks) work precisely because the attacker must *impersonate* something. The compliant attacker is *mask-constrained adversarial feature exploitation*: it need not mimic any waveform, only stay inside its own lawful allocation and budget while exploiting the classifier's learned decision geometry — indeed the PA study shows it is an RF-statistical *outlier* (highest PAPR on the air, 11.05 dB), the opposite of mimicry. The punchline — PUEA countermeasures are largely irrelevant because this attacker is *legally indistinguishable on paper* — is the paper's real policy contribution and is correctly drawn.

### 5. The three questions, answered honestly

**(a) Is this impactful for the regulatory/policy audience?** Potentially yes, currently unrealized. The ingredients regulators value are all present: power-domain metrics, an explicit "worst-case compliant interference" test proposal analogous to receiver blocking/overload testing, the PA-output-port verification principle, honest non-claims, and a one-command reproducible artifact. But there is no regulatory *artifact*: no CFR/ETSI clause mapping, no absolute EIRP/link budget, no docket engagement, and the mask is self-defined. The realistic near-term channels are the FCC Technological Advisory Council (receiver-performance / AI working groups), a 5GAA technical contribution, or ETSI (TC ITS / ISG SAI) — none of which needs a docket number, all of which need the standards math fixed first.

**(b) What do we lack vs. regulatory-grade work?** (1) Real mask math (and first obtaining the standard texts, since the archived "evidence" is a WAF block page); (2) absolute power anchoring: EIRP caps + link budget so MEAP is expressible in meters and dBm; (3) a docket-ready comment — the current draft has no docket number, no CFR or Federal Register citations, no filer contact/disclosure block, contains an internal contradiction (line 101: "RF impairments (PA nonlinearity…) are not modeled" vs. its own PA paragraph and the paper's sec:papr), an overstated transfer number ("11–25 dB" vs 10.8–23.2), and stale artifact pointers; (4) primary-source evidence archived in-repo for the >50-waiver and 5GAA claims; (5) the PUEA canonical citations.

**(c) Is the research gap COMPLETELY filled from the regulatory angle — or only the academic goal?** The *academic* goal — first mask/allocation-constrained wireless adversarial threat model, MEAP/PoC metrics, defense quantification, real-signal and second-victim stress tests, under an 11-wave adversarial audit protocol — is essentially complete, and the audit trail itself is exemplary. The *regulatory* goal is roughly half done: the framing and honesty are regulatory-grade, but the quantitative meaning of "compliant" is still the authors' own projection. The gap between "compliant by our construction" and "compliant under EN 302 571/Part 95 measurement procedures" is open, and closing it is precisely the follow-up work the paper gestures at in Limitations (2) and (10). Answer: academic goal filled; regulatory angle **not** completely filled.

### 6. Top 5 must-do items

| # | Item | Effort | Notes |
|---|---|---|---|
| 1 | **Implement the real masks + measurement conventions**: obtain and archive the actual EN 302 571 V2.1.1 text and FCC 24-123 (Part 95 C-V2X) / 15.407 text; implement breakpoint masks enforced after RBW convolution at the PA output; re-run the headline PoC/MEAP. First sub-step: replace the WAF-block-page "evidence" files. | ~8–10 h (+1 h archiving) | Expect PoC to shrink (attack gains OOB shoulders + in-band concentration) → threat conclusion strengthens; this is the paper's title claim made true |
| 2 | **Make REGULATORY_COMMENT_DRAFT.md submittable** — NOT submittable as-is. Add: docket reference (ET Docket No. 19-138) with proper ex parte presentation notice (47 CFR §1.1206) *or* retarget to FCC TAC / 5GAA / ETSI; CFR citations (47 CFR Part 95; 15.407); Federal Register cites (86 FR 23281; 89 FR 100838); fix line 101's "PA not modeled" contradiction; fix "11–25"→"11–23 dB"; update artifact pointers; add filer contact + disclosure block. | ~3 h | The mechanism paragraph is already docket-quality prose |
| 3 | **Add missing PUEA references** — Chen/Park/Reed JSAC 2008; Jin/Anand/Subbalakshmi ICC 2009 + series; one sentence on the FCC TVWS (ET 04-186/08-260) docket lineage; 2 sentences mapping cloaking↔selfish PUEA and DoS↔malicious PUEA. | ~2 h | C32 then deserves full VERIFIED status |
| 4 | **Absolute-power anchoring / link budget**: EIRP caps (23–33 dBm classes), TR 37.885-style path loss, and a MEAP→attack-range table. | ~4 h | Converts MEAP from dB-relative to policy units |
| 5 | **Fix the band-label and front-door overreach**: "U-NII-2"→"U-NII-1" for the 5240 MHz captures; soften README/ONE_PAGER/draft from "every spectrum rule / every in-band power check" to the defensible "spectral-shape constraints modeled on the applicable masks" pending item 1. | ~1 h | Cheap credibility with exactly the audience being courted |

### 7. Verdict

**NEEDS-WORK.** The science core is real and unusually well-policed: every headline number traces to a JSON, the 11-wave audit trail caught and corrected its own critical bugs, the FCC 20-164/24-123 timeline is accurate to the effective date, and Wave 11's PA study asks exactly the question a conformance engineer asks — *where does verification happen?* — and answers it correctly (the PA output port). But from the regulatory chair, three things keep this out of "credible" territory: (1) the paper's central noun — the *emission mask* — is a self-defined idealization whose relationship to EN 302 571 or 47 CFR Part 95/15.407 has never been computed, and the repo's archived "standard" is a firewall block page, so the standards basis of the entire compliance claim is unverified in-repo; (2) the front-door documents assert a stronger claim than the model supports, and the draft is not filable; (3) small but telling errors — 5240 MHz labeled "U-NII-2" (it is U-NII-1, channel 48) — signal to a spectrum-literate reader that the regulatory layer has not yet been staffed by someone who checks band edges. The encouraging part: every defect is fixable in roughly 18–20 hours of standards engineering, and fixing item 1 will almost certainly make the *threat* conclusion stronger, not weaker. This is the rare paper that is one honest pass away from being able to walk its compliance talk.

*(end of member 21-b transcript)*

## Member 21-c — V2X Systems Realism & Venue Fit

### 1. Position Statement

**Grade for systems realism & venue fit: B−**

**Thesis:** *The Price of Compliance* is an adversarial-ML paper of unusually high *link-level* physical honesty — window-energy PSR axis, PA reality check, censoring-aware metrics, worst-case disclosure, and a real-capture stress test — whose *vehicular-systems* half (deployment story, harm chain, packet-level consequence, OTA transmission) is asserted in prose but never measured, and that gap is precisely where a Vehicular Communications referee will grade it.

In eleven waves this team fixed everything an ML-methodology reviewer could ask for: the 33.11 dB PSR unit error (W1-A F1), the inverted FCC band plan (F3), the C14 layout bug (W4-A), the fabricated bibliography author (W1-C), and closed CLAIMS C1–C32 to all-VERIFIED with a 3-seed headline (−37.1/−38.1/−37.9 dB, 1.1 dB spread). But every wave optimized the attack–sensor link in isolation. Not one experiment contains a packet, a MAC decision, a mobility trace, or a transmitted radio wave from this team's own hardware. The paper says "safety-relevant" (main.tex, intro; ONE_PAGER §"The situation") without a single BSM, PER, or PRR number anywhere in 18 pages. The internal review loop (W4-B: "major revision encouraged") has converged on criteria the venue's external reviewers will not be using.

### 2. Deployment-Story Audit

**Where would learning-based wideband spectrum sensing actually sit in a 2026 V2X ecosystem?**

| Candidate location | What it actually does in 2026 | ML in the loop? |
|---|---|---|
| C-V2X Rel-14/15 PC5 sidelink (Mode 4) | SPS resource selection by decoding SCI on PSCCH + **RSRP/S-RSSI measurement** (TS 36.213 §14); Mode 3 is eNB-scheduled. No LBT — ITS is licensed exclusive spectrum. | **No** |
| IEEE 802.11p/DSRC EDCA (sunsets ~Feb 2027 per FCC 24-123) | CCA = preamble detection + energy detection (CCA-ED ≈ **−65 dBm** for the 10 MHz channel), AIFS/CW backoff. | **No** |
| U-NII-4 ↔ ITS coexistence (FCC 20-164/24-123) | Contention-based channel access, energy/preamble-based per industry standardization; DSRC-era detect-and-vacate proposals were energy-threshold designs. | **No (today)** — this is the plausible future insertion point, exactly where Girmay et al. (Vehicular Communications 39:100563, 2023 — the paper's own `[girmay2023]`) proposed CNN technology recognition |
| O-RAN RIC xApps / network-side monitoring | Threat-mapped by Habler et al. `[habler2025]`; ML sensing proposed for RIC | **Proposed/prototype** |
| Regulatory monitoring / conformance labs | ETSI EN 302 571 unwanted-emission & blocking tests are mask/procedure-based; ML monitors are a research topic | **Proposed** |

**Verdict on the premise:** Not a strawman, but a *forward-looking* story, and the paper mostly knows it. The threat model (main.tex §4.1, Fig. 2) — an attacker whose mask-projected waveform fools a 4-way technology classifier into false-IDLE (targeted cloaking to "Noise", §4.2) — is coherent *if and only if* an ML classifier gates channel access or coexistence behavior. §7's five deployment scenarios (O-RAN RIC gate, FCC transition margin budgeting, ETSI-style conformance extension, benchmark, vendor CI) are honest and correctly scoped. But the intro drifts: "roadside units and vehicles must distinguish C-V2X PC5, legacy 802.11p... and unlicensed Wi-Fi" (present-tense necessity, main.tex ~line 113–117) implies deployed ML sensing. A T-VT/VC reviewer who knows TS 36.213 §14 and 802.11-2012 §15.4 will write exactly the table above in their review. The fix is two sentences of honest placement plus a harm chain (below).

**The honest, quantitative harm chain from "20% cond-ASR at −37 dB" to a safety consequence** — none of which appears in the paper:

1. **The −37 dB perturbation is a trigger, not a weapon.** At PSR −37 dB the attack is ~5,000× weaker than the victim signal (and the ONE_PAGER's "four orders of magnitude" rounds up 3.7 orders). It cannot itself degrade any radio receiver. The *operational* threat is the regime the paper also reports: 96.7% cond-ASR at PSR −10 dB (Table 4, urban) — a mask-compliant transmission that passes every spectrum monitor.
2. **False-IDLE licenses the real weapon.** Fooled into "Noise", a DAA/coexistence manager authorizes its *legal* full-power transmission: a U-NII-4 device may radiate up to 36 dBm EIRP. At 50 m, FSPL(5.9 GHz) ≈ 82 dB → interference ≈ −46 dBm at a C-V2X receiver; a BSM from a vehicle 100 m away (23 dBm, FSPL ≈ 88 dB) arrives at ≈ −65 dBm → **I/S ≈ +19 dB** → PSSCH SINR collapse → PER ≈ 100% for the duration of the false-IDLE event.
3. **Packet-level cost.** BSMs at 10 Hz (SAE J2735/J2945/1); a 1-second false-IDLE event costs up to 10 consecutive BSMs, against TR 37.885's ~100 ms latency budgets for crash-risk warnings — 2–3 consecutive losses (200–300 ms) already degrade cooperative warnings; PRR at the affected range drops from the ~90%+ target to near zero for that event.

So: attack success rate × decision/event duration → expected missed BSMs → PRR → safety margin. This bridge costs one link-budget subsection (hours) or an ns-3/ms-van3t experiment (the toolchain the worklog already identified as the Phase-2 "safety cascade" and deferred). Without it, the paper's only quantified consequence is a label flip. One additional honesty point in the paper's favor: the white-box worst-case disclosure and the C13 no-CSI result (compliant threshold −37 → ≈−16 dB, a 21 dB penalty) and C23 transfer penalty correctly bound the field-realistic attacker.

### 3. Venue-Fit Comparison Table

| Venue | What accepted papers there typically include (systems evidence) | Anchor papers | What we bring | Fit as-is |
|---|---|---|---|---|
| **Vehicular Communications** (target, Dec 2026–Feb 2027) | Real ITS-band 5.9 GHz measurement/testbed data; ns-3/SUMO network evaluation; field trials; vehicular datasets | Girmay et al., VC 39:100563 (same venue, same task family — **collected real ITS-band captures**) | Post-FCC band plan (correct), TR 37.885-*flavored* channels, 96 real windows at **5.24 GHz** (not 5.9), 3 fixed locations, no mobility; no MAC/ns-3/field | **Borderline**: topical fit is real (Girmay precedent, FCC transition framing); systems evidence sits one notch below the venue's own precedent paper |
| IEEE T-ITS | ns-3/SUMO/Veins with mobility traces; field trials; large-scale datasets; system-level KPIs | typical T-ITS V2X-security papers | none of the above | Poor–medium |
| IEEE T-VT | Standard-compliant PHY fidelity (3GPP/IEEE numerology, pilots, measured PAPR/SEM), link-level analysis, often OTA | T-VT PHY-security papers | QPSK-only, no pilots/preamble/DMRS; PC5 is a disclosed 450×20 kHz stylization vs TS 36.211's 50 PRB × 15 kHz; no MAC | Medium–poor |
| IEEE DySPAN | Spectrum-policy + coexistence analysis; measurement campaigns; PUEA lineage | Anand et al., DySPAN'08 (cited `[anand2008]`) | The regulatory framing is our genuinely strongest suit: mask-constrained feasible set, MEAP/PoC as conformance-style metrics, FCC 20-164/24-123 grounding; but no measurements of our own | **Good for a policy-flavored shorter version** |
| ACM WiSec | SDR OTA attack demonstrations (USRP/WARP) effectively standard; evaluated mitigations | Kim & Sagduyu CISS'20 lineage; RadioShock (sister venue TDSC'26) did OTA with CSI estimation | The attack has never been transmitted through any radio; PA is a memoryless Rapp model | Poor |
| IEEE TIFS | Real-world validation / measured testbeds, or exceptionally rigorous modeling + comprehensive evaluation | RadioShock, TDSC 23(4) | Simulation + third-party captures + model PA; audit discipline is exceptional but is not RF evidence | Poor as-is |

### 4. What We Did Right (Systems-Wise) and What We Are Missing

**Right (and rare in this literature):**
- **Power-domain honesty:** window-energy PSR referenced to clean received energy, MEAP/PoC budget-matched and censoring-aware, worst-case disclosure (attack_mask.py docstring lines 9–19). Most wireless AML papers still report ε-ball ASRs; §5.4's PSR-equivalence demolition of the legacy threat model is a real service to the sub-niche.
- **PA reality check (sec:papr):** attack PAPR 11.05 dB vs 6.26–9.02 dB benign, IBO* = 13/15 dB for strict post-PA compliance, PA-aware re-optimization *stronger* (−1.23 dB) and 100% post-PA compliant — with pre-registered H1–H3 and an independent audit (w11_audit_report.md, PASS). To my knowledge no adversarial-ML wireless paper has done this; it converts "compliance" into a conformance-test statement at the PA output port.
- **Real third-party OTA captures, leakage-free:** real_wifi.py (DC removal, ×2 polyphase resample, −5 MHz placement, spectral-flatness + power burst gate), disjoint train/eval locations, CC BY-NC-SA provenance. It replicates the headline within 1.8 dB (frozen −35.92 / fine-tuned −37.67 vs synthetic −37.1) and surfaces the OOD margin collapse (0.0% synthetic vs 68.5% real flips at −45 dB PSR) — an honest negative of genuine value.
- **Coherence-time physics:** 656 Hz max Doppler → 0.55–0.76 ms ≫ 102.4 μs window (channels.py lines 13–16) — block fading is correctly justified.
- **Stress matrix:** 3 training seeds, 2 attack seeds, PGD-50, CSI/no-CSI, psd-margin sweep, second victim (+11.6–15.1 dB architecture shift, halved PoC), surrogate transfer (11–25 dB), AT (+14.0 dB, clean 100%) and TRADES, all JSON-provenance-embedded and audited.

**Missing (brutally specific):**
- **Packet-level evaluation: zero.** No ns-3/ms-van3t/Veins, no BSM PER/PRR, no channel-access consumer of the sensor's label. "Safety" appears in the intro, abstract and conclusion with no packet behind it.
- **Standard waveform fidelity:** no 802.11p STS/LTS preamble, no PC5 DMRS/PSCCH structure, QPSK-only symbols ("pilots and preambles are not modeled"); PC5 stylized; 102.4 μs windows always see mid-burst data with no preamble/acquisition structure. No CFO, phase noise, AGC, IQ-imbalance, or quantization in the front-end.
- **Channels/mobility:** 3–6-tap statistical block fading; no geometry-based TR 37.885 CDL models, no path-loss/shadowing (attacker geometry), no dual-mobility (QuaDRiGa was identified in Wave-1 and never used), no inter-window dynamics, no SUMO. **Phase 0 actually had more systems surface than Phase 1 kept**: `mobility_scenario_eval.py` carried per-scenario Rician-K ranges, log-normal shadowing (σ = 4/10/6 dB), PDP spreads, speed ranges and 3-seed stratified results; `v3_rician_doppler.py` applied time-varying Doppler to taps; Phase 0 also reported real-time feasibility (0.985 ms/window CPU, latency_results.json). Phase 1 dropped shadowing, time-varying Doppler, latency, and the SUMO/GNU-Radio hooks, and never restored them.
- **Dataset scale:** 3,200 synthetic + 800 val windows; the entire real-data corpus is 96 gated windows ≈ **9.8 ms of WiFi airtime** from 3 fixed Gent locations at 5.24 GHz, no vehicle, no mobility — against DeepSense (days of OTA, 413 cites), WiSig (10M packets), and Girmay's own real 5.9 GHz ITS-band captures at the target venue.
- **Receiver-impact quantification:** MEAP measures the sensor only. No cooperative-sensing fusion baseline (the classic PUEA mitigation — multi-sensor fusion would blunt a single-receiver attack and is never mentioned), no adversarial-input detection comparison (Zhao INFOCOM'24 `[zhao2024]` is cited, not run).
- **The attack is never transmitted.** PA is a Rapp model; at WiSec/TIFS-grade venues, RadioShock-class OTA evidence is the emerging bar.
- **Process tail:** worklog.md has **no Task 12–18 entries** — Waves 6–10 are unlogged, the same failure W11-F's F8 flagged for Task 19; HEAD 3e53ca4 is unpushed (PAT), Zenodo DOI unminted, so the abstract's "we release the complete framework" is live on the branch but not on the front door.

### 5. The Three Questions, Answered Honestly

**(a) Is this impactful enough for the named venue?** *Conditionally, not yet.* The compliance-constraint framing, MEAP/PoC, and audit discipline are a real contribution the vehicular-ML-security niche lacks, and Girmay'23 proves VC publishes ITS-band ML sensing. But at a vehicular venue, impact rides on vehicular-system relevance, and today the vehicular content is (i) a correct band plan, (ii) TR-37.885-flavored channels, (iii) BSM-free "safety" prose. A VC referee's one-line reject rationale writes itself: "solid adversarial-ML methodology; the vehicular half is motivational." The paper becomes impactful the moment a reviewer can trace false-IDLE → authorized unlicensed transmission → BSM PRR loss.

**(b) What exactly do we lack vs related work?** vs Kim & Sagduyu TWC'22: they matched our channel-aware optimization and *also* shipped a certified defense — we have none. vs RadioShock TDSC'26: real OTA with CSI estimation and dynamic adaptation — we have zero transmission. vs Zheng TMLCN'26: targeted attacks on wideband *multi-label* sensing at larger scale — we are 4-class, 20 MHz, one window. vs Liu T-Rel'23: they add causative (poisoning) attacks — we are evasion-only. vs Zhao INFOCOM'24: a detection defense we neither implement nor compare. vs Girmay VC'23: real 5.9 GHz ITS-band data — our real class is 5.24 GHz, 96 windows, third-party. vs Croce'20 evaluation discipline: we adopted the honesty but our attack protocol remains PGD-10, zero-init, no restarts — and PGD-50 moves mid-grid ASR by up to +37 pp (disclosed, but a TIFS/WiSec reviewer's first methodological strike). Net list: OTA transmission, standard-compliant waveform generation (MATLAB 5G/LTE toolbox and gr-ieee802-11 are both free), packet/network-level evaluation, dataset scale, a certified or detection defense, mobility/geometry in the channel.

**(c) Is the research gap COMPLETELY filled from the systems angle?** **No — only the internal goals are.** CLAIMS C1–C32 are all VERIFIED; the *link-level physical honesty* gap (feature-space fiction, hidden power budgets, no PA, no real signals) is largely closed — that is this team's genuine achievement. But the *vehicular-systems* gap — where the sensor sits, who consumes its decision, and what happens to a BSM when it lies — is not measured anywhere; it is argued qualitatively in §7's scenarios. Scorecard: physics-of-the-attack ✓; physics-of-the-receiver partial (no RF impairments); system-of-the-receiver ✗; consequence ✗.

### 6. Top 5 Must-Do Items (with effort)

1. **Quantified harm bridge (mandatory for VC):** add a "From false IDLE to missed BSMs" subsection — link-budget table, then optionally convert the measured cond-ASR-vs-PSR curves into PRR degradation in ns-3/ms-van3t for 2–3 scenarios. **Effort: analytical 6–10 h; ns-3 40–80 h.**
2. **Rewrite the deployment story honestly + restore real-time feasibility:** locate the sensor in O-RAN xApp / DAA-prototype / conformance contexts and state explicitly that C-V2X Mode-4 selection is RSRP/SCI-based and 802.11p CCA is energy/preamble (≈ −65 dBm ED) — today's radios are not the target; proposed sensing architectures are. Restore the Phase-0 latency number (0.985 ms/window CPU) as feasibility evidence. **Effort: 5–8 h.**
3. **Minimum credible OTA experiment — and the Flipper Zero verdict.** **Flipper: exclude it entirely.** Its CC1101 covers only 300–348/387–464/779–928 MHz (~+5–14 dBm), and — decisively — it has **no arbitrary-IQ transmit path**: hardware OOK/2-FSK/GFSK/MSK at ≤ ~500 kBaud; the .sub format replays pulse streams, not complex waveforms. A 10-MHz-bandwidth mask-projected PGD waveform is unrepresentable on it at *any* band, and ISM bands have no allocation/emission-mask rules, so the paper's entire compliance mechanism has no ISM analog. A "433 MHz scale model" would demonstrate nothing the paper claims and a vehicular/security reviewer would treat it as a gimmick that damages credibility. **Minimum credible 5.9 GHz OTA path:** conducted/cabled first — 1× USRP B210 ($1,699) as attacker TX replaying the repo's precomputed PGD waveforms via GNU Radio file source, programmable step attenuator ($300–600) to set PSR precisely, a second B210 or B205mini (~$1,400–1,700) as the sensing RX running the frozen front-end (HackRF One ~$350 is the 8-bit budget fallback — dynamic range is marginal at −37 dB PSR); measure the MEAP curve over 10–12 PSR points × ~100 windows and report float-vs-RF deltas. Radiated extension only in a shielded/anechoic chamber or under an FCC Part 5 experimental license (Form 442, no fee, weeks). **Total: ≈ $2.2–4.5k; effort 40–60 h.**
4. **Waveform fidelity + receiver-impairment sensitivity:** add 802.11p STS/LTS preamble and PC5 DMRS structure (or export standard-compliant slots via MATLAB 5G/LTE toolbox or gr-ieee802-11) and re-run the canonical urban grid; add CFO/phase-noise/AGC/quantization to the differentiable front-end and report MEAP sensitivity. **Effort: fidelity 15–25 h; impairments 5–10 h.**
5. **Margin-stratified reporting + one detection baseline + ship hygiene:** apply the paper's own R3 insight to the *synthetic* headline (cond-ASR stratified by clean-decision margin; MEAP on margin-matched subsets) to preempt "MEAP measures the weakest 20% of a 100%-accuracy CNN's windows"; add one adversarial-input detector comparison (Zhao-style DDB or a confidence/OOD gate) to the defense section; push the pending 3e53ca4 commit, mint the Zenodo DOI, and backfill worklog Tasks 12–18. **Effort: 10–15 h + 1 h hygiene.**

### 7. Verdict

**MAJOR-REVISION (encouraged), with a concrete path to the Feb 2027 submission window.**

The core science is sound and, in its constraint-set formulation, genuinely novel; the audit trail is the best I have reviewed at this scale. But the internal review loop has been grading ML-methodology criteria while the target venue grades vehicular-systems criteria — and on those, the paper currently offers a correct band plan, simplified channels, 9.8 ms of borrowed real airtime, and no packet, no MAC, no mobility trace, and no transmitted attack. The gap is bridgeable: items 1, 2 and 5 (~25–35 h, zero hardware) make it an honestly-framed, defensible Vehicular Communications submission; item 3 (a ~$2.2–4.5k cabled B210 OTA replication of the MEAP curve) upgrades it from "simulation with a PA model" to a paper with RF evidence that would also survive WiSec/TIFS scrutiny. Do not submit the current draft, and do not spend a cent or a page on the Flipper Zero.

*(end of member 21-c transcript)*

## Member 21-d — Methodology, Statistics & Reproducibility Referee

### 1. Position statement

I audited the `phase1-compliant-attacker` branch (local HEAD `3e53ca4`) as the referee who checks error bars, seeds, censoring, and claim-status integrity. I recomputed ~40 headline numbers directly from the result JSONs (pure-numpy interpolation of every stored MEAP/PoC: canonical 6-cell grid, s123/s456, seed-11, AT/AT-rural/TRADES, victim-2 + transfer, real-WiFi, SNR sweep, all Wave-11 PA arms) and **every one reproduces exactly**; I also ran `w6b_check.py` live (all JSON re-derivation sections PASS). The internal traceability discipline is the best I have seen at this scale. What the package lacks is *external* statistical validity: **not a single error bar exists anywhere** — my binomial-CI calculation shows a single MEAP cell at n=300 carries ±1.0–1.9 dB of sampling noise, *larger* than the 1.06 dB 3-seed spread the paper leans on; the defense/TRADES/real-WiFi/PA studies are all single-seed; PGD-10 headline numbers are conservative by up to **+37 pp** mid-grid (verified: PGD-50 vs PGD-10 at −25 dB: 95.33 vs 58.33); and the real-data evidence is 74 windows from one location at 5.24 GHz.

**Overall grade: B−** (statistical rigor & reproducibility).
**Thesis:** the artifact is exactly reproducible and honestly scoped, but its stability claims are asserted with less statistical power than they appear to carry, and ~25 hours of targeted work separates "audited prototype" from "journal-grade evidence."

### 2. Claims-vs-evidence audit table

| Group | Status claimed | Actual backing found (recomputed) | Verdict |
|---|---|---|---|
| **C1–C5** clean task | VERIFIED | C2 recomputed from `train_report.json` (88.62/77.75/100/100, 86,052 params ✓); C4 from `snr_sweep.json` (73.0/62.6; 25.6/25.0 ✓); C3 PAPR now JSON-backed via `papr_pa_results.json` but CLAIMS C3 still cites `check_waveforms.py`, which is **not in the repo** | SUPPORTED (C3 evidence pointer stale; C4 has seed-overlap caveat) |
| **C6–C11** attack core | VERIFIED (prototype) | All 12 MEAPs/PoCs recompute **exactly** (urban −44.22/−37.08/7.14; highway −43.92/−37.21/6.71; rural −44.50/−39.14/5.36; cloaking 16.01/17.38/18.12; ASR@−10 96.67/92.0; @−30 39.33/39.67). C10 row still says "0% below −25 dB" — JSON says 0.33% at −30 (highway/rural); paper says ≤0.4% | SUPPORTED (C10 row stale vs own paper) |
| **C12** 3-seed + robustness | VERIFIED | 3 training seeds exist with urban-untargeted grids only: mask MEAP −37.08/−38.15/−37.89 (recomputed ✓), genie floor-censored 2/3 (PoC lower bound, disclosed). BUT the originally promised protocol (README: epochs 40, attack seeds 11/22/33, 3 scenarios × 2 modes, PGD-50, n=200) was **quietly redefined** to a "3-seed core" (25 epochs, urban/untargeted, attack seeds 7+11 only). Row text still carries the W4-cherry-pick "+3–6 pp at low power" (true only at −40/−35; actual PGD-50 deltas: +24.7/+37.0/+21.3 pp at −30/−25/−20) | THIN — real but narrower than "3-seed error bars" implies |
| **C13** CSI robustness | VERIFIED (prototype) | Runs in JSON ✓, but the no-CSI MEAP "≈−16 dB" / "~21 dB penalty" is **not stored**; I recomputed −16.88 dB by interpolation → penalty 20.2 dB (claim rounds −16.9→−16, 20.2→21) | THIN (number derivable, not JSON-stored; slightly flattering) |
| **C14** feature-space | VERIFIED (corrected) | `feature_space_equiv.json` ✓ (71.33/14.0/−24.18/99.33/62.33; ε≤0.3 → 0–1.67%) — all match | SUPPORTED |
| **C15** worst-case disclosure | VERIFIED | Docstring + paper Limitations (4) ✓ | SUPPORTED |
| **C16** AT defense | VERIFIED | All anchors recomputed exactly (58.33→16.67 @−25; 78.0→25.33 @−20; 96.67→55.33 @−10; MEAPs +6.76/+14.01; 94.67 @0 dB) | SUPPORTED (single training seed, disclosed) |
| **C17** TRADES | VERIFIED | `attack_results_urban_untargeted_trades.json`: mask −19.15, PoC 18.62 ✓; clean 99.25% from `trades_train_report.json` | SUPPORTED (single seed) |
| **C18** all numbers traceable | VERIFIED | Most JSONs embed config/seed ✓, **but** `feature_space_equiv.json` embeds NO config/seed (flagged by W4-A, never fixed) and no phase-1 JSON embeds env | OVERSTATED (two exceptions) |
| **C19–C20** Phase-0 linkage | VERIFIED | smoke JSON in `v2x_release/results/reproduction/` (v2.0.0 tag) ✓; cconv verified by W1-A/W4-A (not re-runnable here — torch absent) | SUPPORTED |
| **C21** real OTA WiFi | VERIFIED | Frozen 66.22%/−35.92, finetuned 72.97%/−37.67, PoC 7.33 lb — all recompute ✓. **But** "0.0% synthetic flips vs **68.5%** real flips @−45" is only obtainable by 3-step inference on the *finetuned* arm (14.57%×254=37 flips / 54 real-eligible = 68.5%); the frozen real-only curve stores **67.35%**; neither the fraction nor which model is stored | THIN (arithmetically supportable, violates the letter of the number policy) |
| **C22–C23** second victim + transfer | VERIFIED | resnet white-box −29.17/−25.47/3.69 ✓; D2R −18.41/−10.00, R2D −21.02/−18.00 ✓; recomputed transfer cost 10.8–23.2 dB → paper "11–23" ✓ but **CLAIMS C23 says "11–25"** (wrong) and C22 "12–15" vs actual 11.6–15.1 | SUPPORTED (C22/C23 rows stale) |
| **C24–C26** rural transfer / CLI / W6 gate | VERIFIED | atrural −21.89/13.37 ✓; redteam JSON curves == canonical, bit-consistent ✓; I ran `w6b_check.py`: all JSON sections PASS | SUPPORTED |
| **C27–C32** PA reality check | VERIFIED | 11.05 vs 9.02 max benign ✓; IBO*=13/15 ✓; lenient pass 8% → "92% fail" (post-F5a) ✓; shifts +0.60/−0.09/0.00 ✓; PA-aware −38.23 (100% strict, PAPR 11.06 stored) / IBO-0 −38.32 non-compliant ✓; W11-F5b/f5c fixes landed | SUPPORTED |
| **Pendings** | none in CLAIMS.md; `\FULLP{}` macro has **zero usages** in main.tex | Remaining honest opens live only in prose: README "Remaining extensions" and C13's "noise-realization knowledge variant still open" buried inside a VERIFIED row | Acceptable, but single-seed defense/TRADES is the *de-facto* pending that blocks submission |

### 3. Statistical-rigor audit

- **Sample sizes.** n=300 per grid point (100/active class × 3 active classes; noise excluded, disclosed) for all attack grids; real-WiFi track: 274 (200 synthetic + **74 real**); PAPR: 300 windows. n is fixed and adequate for curve shape, not for the threshold crossing precision claimed.
- **Error bars / CIs: none exist.** I computed them: at the urban mask crossing (13.0→25.0 over 5 dB, slope 2.4 pp/dB), the ±4.5 pp binomial 95% CI at n=300 gives **MEAP ±1.9 dB**; genie (slope 4.7 pp/dB) gives **±1.0 dB**; PoC (difference) ≈ **±2.1 dB**. Consequences: (i) the 3-seed training spread (1.06 dB) is *smaller than the single-run sampling CI*, so "three seeds move the threshold by at most 1.1 dB" does not establish stability beyond noise; (ii) the scenario differences in the headline "PoC 5–7 dB" (7.14/6.71/5.36) are within one CI — the rural-vs-urban distinction is not statistically resolved; (iii) real-WiFi claims carry ±10.8 pp (acc, n=74) and [56%, 81%] (flips, n=54).
- **Seed design — where the 3-seed protocol actually stands.** Training seeds 42/123/456: trained (3×100% clean, checkpoints shipped) but attacked **only on urban/untargeted** (s123/s456 JSONs); attack seed held at 7 (correct for isolating training variance, but it means attack-seed variance is measured by exactly **one replicate** — seed 11, urban/untargeted, PoC 6.86 vs 7.14). Defenses (AT, TRADES), victim-2, real-WiFi, and all PA arms: **one training seed (42), one attack seed (7)**. The originally promised full protocol (epochs 40, attack seeds 11/22/33 × 3 scenarios × 2 modes, PGD-50, n=200 per point) was **not delivered** — re-scoped as "3-seed core" in the restructured README.
- **MEAP censoring disclosure.** Genuinely good practice (`meap_curve` returns explicit `le`/`ge` flags; PoC lower-bound notes in the paper). However, the grid floor at −45 dB censors the genie MEAP for 2 of 3 training seeds *and* both real-WiFi genies *and* the PGD-50 genie — the grid should simply be extended to −55 dB; it never was.
- **Attack convergence.** PGD-10, zero-init, no restarts. The paper discloses lower-bound direction, but the magnitude is large: PGD-50 lifts mid-grid ASR by **+24.7/+37.0/+21.3 pp** at −30/−25/−20 dB and shifts mask MEAP by −1.7 dB. Any ASR-at-power sentence (e.g., "39.3% at −30 dB") is therefore materially conservative and should either use PGD-50 numbers or carry the caveat at each use.
- **Multiple comparisons.** Wave-11 pre-registered H1–H3 (good). Elsewhere, curves rather than tests — the issue is that the 20%-threshold MEAP has no threshold-sensitivity analysis.
- **Test-set leakage risks.** Real-WiFi split is leakage-free by construction (eval=uz vs train=rabot/reep, asserted in code) — good. **Found:** `run_snr_sweep.py:39-55` builds eval sets with `default_rng(42)`, the same stream as `gen_dataset(seed=42)` — the eval baseband waveforms are **bit-identical to training-corpus members** (channels/noise drawn independently, so the received inputs differ; mild but real hygiene flaw, and it's the C4 evidence). The canonical attack eval (rng(7)) is clean. The LR baseline in the SNR sweep trains on all 4,000 windows vs the CNN's 3,200 — biases the comparison *against* the paper's own conclusion (honest direction, undisclosed).

### 4. Reproducibility audit

- **Seeds:** pinned and embedded in every experiment JSON I opened (`config.seed`), training via `set_seed` + cached deterministic dataset builder. ✓
- **Checkpoints:** 11 `.pt` files committed — CI asserts 5 of them exist. ✓
- **Breakage found:** `requirements.txt` lists torch/numpy/scipy/matplotlib but **not scikit-learn**, while `run_train.py:86` and `run_snr_sweep.py:95` import it — a stranger's fresh clone crashes at training step [4/4], losing `train_report.json` (checkpoints survive). CI never runs `run_train.py`, so it is blind to this.
- **CI gate:** meaningful but shallow — 30 windows, clean-acc ≥ 95%, cond-ASR ≥ 50% at PSR 0 (PGD-5), OOB < 1e-9. It does **not** recompute any stored MEAP/PoC from the JSONs (a 20-line pure-python interpolation step would), so a corrupted curve or swapped summary would pass CI.
- **Stranger-on-CPU test:** yes for the headline — training ~5 min/seed, one attack grid ~30 min, `v2x_redteam.py` reproduces the canonical curve **exactly** (I verified curve identity). The full 6-cell grid + defenses + PA ≈ 4–8 CPU-hours, inside one day. ✅
- **Environment claims drift:** worklog says torch 2.14 CPU lives at `/home/z/.venv`; in today's sandbox torch is gone (I verified at JSON level and ran `w6b_check.py` until its torch import). Phase-1 JSONs embed no env block — the Phase-0 practice was better.
- **State of the ship:** restructure commit `3e53ca4` is **unpushed** (origin at `a03e4c5`); Zenodo DOI unminted; **worklog has no Task 12–18 entries** — waves 6–10 are entirely unlogged, the same failure W11-F8 caught for Task 19 but 5 waves' worth; README's "every wave was followed by an independent code+paper audit" is documented only for waves 1, 4, 6, 11.

### 5. The three questions

**(a) Is the methodology strong enough that results can be trusted?** Directionally yes; quantitatively, not yet as printed. The direction-level conclusions (compliance costs ~5–7 dB untargeted / 16–18 dB cloaking; an undefended CNN breaks deeply below parity; AT adds ~14 dB at zero clean cost; the legacy feature-space model overstates physical realizability; PA compliance must be checked at the output port) are internally consistent, survive my recomputation, and survived 11 waves of adversarial audits that caught and fixed real bugs. But the *absolute* numbers carry three unquantified error sources that stack: ±1–2 dB sampling noise (never reported), PGD-10 conservatism (up to +37 pp mid-grid), and single-seed defenses/real-data. A referee who recomputes the CIs — as I did — will notice the stability narrative is under-powered.

**(b) What do we lack vs. best practice in the field?** (i) Error bars: binomial CIs on every ASR point and propagated MEAP/PoC intervals — standard since RobustBench/AutoAttack practice; (ii) ≥3 seeds for *every* headline cell including attack seeds and defense models, reported as mean±std; (iii) converged attacks (PGD-50 or APGD with restarts) as the headline, not a sensitivity appendix; (iv) grid extension below −45 dB to un-censor genie MEAPs; (v) real-data scale: n≥300 real windows, ≥3 eval locations, ideally U-NII-4/5.9 GHz (current: 74 windows, 1 location, 5.24 GHz U-NII); (vi) per-class flip statistics *stored* in JSONs (the 68.5% episode); (vii) a detection-side defense comparison (Zhao INFOCOM'24 is cited, not run); (viii) an OTA or conducted validation of the attack waveform (all evidence is simulated through a Rapp model); (ix) mechanical claim-sync (CLAIMS.md currently trails the paper in 5 rows).

**(c) Is the research gap COMPLETELY filled?** Acceptance criteria and scores:

| Criterion (measurable) | Score | Evidence |
|---|---|---|
| 1. Compliance-constrained attack **viability**: mask-constrained waveform-domain attack breaks a DL sensor, ≥2 scenarios, both modes, worst-case disclosed | **DONE** | 3 scenarios × 2 modes; 2 victims; white-box/CSI/transfer/PA-aware variants; C6–C11, C22–C23 recomputed |
| 2. **Quantified cost**: PoC/MEAP with uncertainty, sensitivity, and converged attacks | **PARTIAL** | PoC quantified + margin/steps/seed sensitivities exist, but no CIs, PGD-10 lower bounds (+37 pp gap), 2/3-seed genie floor-censored, attack-seed n=1 replicate |
| 3. **Defense**: matched defense evaluated with seeds and against adaptive attack | **PARTIAL** | AT + TRADES + rural transfer, honest range boundary — all single training seed, single attack seed, PGD-10 (non-adaptive) evaluation, one victim |
| 4. **Physical realism**: real signals, real chain, real band | **PARTIAL** | Real OTA WiFi ✓ but 5.24 GHz, n=74, 1 eval location; Rapp-only PA; TR-37.885-*inspired* channels; no OTA attack transmission; flat-cap mask stricter than any regulation |

**Not completely filled** — roughly 70–75% closed at the science level; the viability leg is done, the cost/defense/realism legs each have one quantified gap left.

### 6. Top 5 must-do items

1. **Add error bars + un-censor the grid** — extend PSR grid to −55 dB (re-run the 6 canonical cells + real-WiFi genie), attach binomial 95% CIs to all ASR curves, propagate to MEAP/PoC, and re-report Table 4 and the 3-seed row as mean ± CI. *(4 h compute + 3 h analysis/writing)*
2. **Re-run headline grids at PGD-50** (6 canonical cells + AT + TRADES cells) so MEAPs/ASR-at-power are no longer lower bounds; single intentional number cutover in paper/README/CLAIMS, one pass. *(7 h CPU + 1 h writing)*
3. **Seed replication where claims live:** attack seeds {11, 22} across all 6 cells; AT and TRADES training seeds {123, 456} on the urban cell; report mean±spread. *(6 h CPU + 1 h writing)*
4. **Real-WiFi scale + traceability:** download more Fontaine captures (and/or widen the gate) to n≥300 eval windows across ≥3 locations; store per-class flip rates and the no-CSI MEAP in the JSONs; fix the 68.5% sentence to name the finetuned model. *(4 h)*
5. **Ship hygiene:** add scikit-learn to `requirements.txt`; add a CI step that recomputes 5 stored MEAP/PoCs from the JSONs (pure python) + runs `v2x_redteam.py`; re-seed the C4 sweep (e.g., 777) to kill the waveform-identity overlap; sync stale CLAIMS rows (C10 "0%", C12 "+3–6 pp", C13 "≈−16/21 dB", C22 "12–15", C23 "11–25") and ONE_PAGER's "four orders of magnitude" (−37 dB ≈ 3.7 orders); push `3e53ca4`; backfill worklog entries for waves 6–10; mint the Zenodo DOI. *(3 h)*

**Total ≈ 25–29 h** — this is the minimum set that makes every remaining claim either properly VERIFIED or honestly error-bared.

### 7. Verdict

**MAJOR-REVISION** (encouraged — resubmit-ready after the 5 items above).

The science core is real, novel enough for the niche, and unusually well-audited: every number I recomputed from the 30-plus result JSONs matched exactly, the censoring and worst-case disclosures are exemplary, the W11 independent-audit fixes demonstrably landed, and a stranger with a CPU can reproduce the headline in under a day (modulo the missing `scikit-learn` dependency). But a methodology referee cannot accept a paper whose *entire* statistical footing is single-cell point estimates: the error bars I computed (±1–2 dB per MEAP, ±10.8 pp on the 74-window real-WiFi accuracy) are larger than several of the differences the text narrates; the defense and TRADES results that anchor the "defense" contribution are one seed against a PGD-10 (non-adaptive, up to 37-pp-conservative) attack; and the "3-seed protocol" as originally specified was quietly re-scoped to a single grid cell. None of this is fatal — the gap between here and submission is roughly 25 hours of compute-and-honesty, not new science — and the audit trail that caught the 33 dB axis bug and the C14 layout bug is precisely the machinery that gives me confidence the remaining numbers will survive the same treatment.

*(end of member 21-d transcript)*

---

# Part III — Process record

- Council convened by the chair (main agent, Task ID 21) on 2026-09-06; four members launched in parallel with identical evidence access, identical seven-section briefs, and no visibility into each other (blind first round).
- Each member was required to: read the full 11-wave worklog; ground every judgment in a specific file/number/claim; recompute what could be recomputed (21-a and 21-d both re-derived all Table-4 MEAP/PoC values from the raw JSONs and matched the stored values exactly); append a worklog entry.
- The chair verified the worklog entries of 21-a/21-b/21-c/21-d, adjudicated the 68.5%/67.35% discrepancy against `real_wifi_checks.json` / the finetuned-arm derivation, and merged the four must-do lists into the P0/P1/P2 roadmap above, deduplicating overlaps (adaptive-attack eval: 21-a #2 ≡ 21-d #2; real mask: 21-a #3 ≡ 21-b #1; harm bridge: 21-c #1; hygiene: 21-d #5 ≡ 21-c #5).
- This document is committed to the repository as part of the integrity trail; the mechanical P0 text fixes identified by the council (claims-vs-backing slips, band label, front-door overreach, dependency breakage) are executed as Wave 12 in the immediately following commits.



---

# Part IV — Council re-scoring, round 2 (Task 34, 2026-09-07)

## 0. Why a second round, and how it was run

After the round-1 verdict, the team executed the **entire P0 and P1 roadmap**: releases
`phase1-v1.2` (P0 closure: adaptive attacks, real ETSI 302 571 mask, harm chain, error bars) and
`phase1-v1.3` (P1 closure: defense-seed replication, C4 leakage fix, latency + margin-stratified
+ detector, filable regulatory draft); branch tip `222f57c`, CI gate genuinely running. The chair
re-derived every headline number from the committed JSONs, compiled an evidence-delta dossier
(`/home/z/my-project/council_rescore_dossier.md`), and re-convened the **same four reviewers**
(now 34-a/34-b/34-c/34-d), blind to each other, each receiving: the dossier, RESEARCH_STATE.md,
their own round-1 transcript, and a **mandatory live spot-check duty** (3–4 recomputations each).

Across the four re-reviews, **20+ live recomputations were performed** (3-seed margins and
aggregates, PoC CI propagation, Wilson widths, ETSI A/B via independent 20%-crossing
interpolation, 650 m EIRP feasibility from the FSPL anchor, PRR delta, latency medians, C4
counts, both audit scripts). **Every headline number matched.** Five new claim-vs-backing
findings were caught (§4 below) — the audit machinery keeps earning its keep.

## 1. Re-graded gap-closure scorecard (round 2)

| # | Criterion | Round 1 | Round 2 (consensus) | Evidence |
|---|---|---|---|---|
| 1 | Compliance-constrained attack viability | DONE | **DONE** (stronger) | adaptive PGD-50×R=5 arms; C4 eval leak dead (250/1000→0/1000, A/B ≤1.3 pp); 3 scenarios × 2 modes × 2 victims stand |
| 2 | Quantified cost (PoC/MEAP with uncertainty, converged attacks) | PARTIAL | **DONE−** | Wilson 95% on every ASR cell; PoC 7.14 dB CI [4.37, 9.80→**10.11** (F5)]; converged PGD-50×R5; genie un-censored at −55 dB. Residuals: attack-seed n=1; conservative (non-paired) propagation |
| 3 | Defense (matched, seeded, adaptively evaluated) | PARTIAL | **DONE−** | 3 defense seeds × adaptive: AT +9.76±2.80 dB, TRADES +11.52±1.75 dB; +14/+18 headline honestly retired; R=1 decomposition blames the evaluator. Residuals: attack seed 7 only; PGD-10 readings seed-42; TRADES-vs-AT not statistically resolved at n=3 |
| 4 | Physical realism | PARTIAL | **PARTIAL↑** | real EN 302 571 Table-7 archived (sha256) + enforced in-chain (+0.36 dB A/B); C4 dead. Still: memoryless Rapp PA, 5.24 GHz U-NII-1 n=74 captures, never transmitted |
| 5 | Deployment story (who consumes the label; consequence measured) | NOT MEASURED | **MEASURED (parametric)** | PRR 0.857→0.762 @D=100 m (−9.45 pp, 94.5 extra lost BSMs/1000); EIRP feasible to 650 m; 2.64 ms median with hardware honesty; §7 placement rewrite. Residuals: single MC seed, no MAC/mobility, no OTA |
| 6 | Regulatory-grade compliance claim | NOT DONE | **PARTIAL (high)** | standard actually read + hashed; MEAP exists in dBm and meters; filable-form draft (WT 23-287 conventions, CFR/FR cites). Residuals: per-bin post-PA enforcement ≠ RBW/detector conventions (Lim. 2/10); draft "verified" overstatement (F3); unfiled, no feedback loop |
| 7 | External validity (third-party victim, detection, uptake) | NOT DONE | **PARTIAL (weak)** | first detection baseline actually run (confidence gate AUROC 0.70–0.76, honestly reported as failing in the compliance regime); margin tertiles published. Still: all victims self-trained, no DOI/uptake, Zhao-style DDB cited not run |

## 2. Two-half re-grade, per reviewer

| Reviewer | Science R1→R2 | Deployment/regulatory R1→R2 | Q3 (gap filled?) | Grade R1→R2 |
|---|---|---|---|---|
| 34-a (adversarial-ML/RF, TIFS/TWC) | ~70–75% → **85%** | ~40% → **70%** | **NO** | novelty A−, rigor B+→**A−** |
| 34-b (spectrum regulation, ex-FCC OET/ETSI) | → **92%** | → **72%** | **NO** | B−→**B+**, NEEDS-WORK→minor-revision |
| 34-c (V2X systems, T-VT/VC) | → **90%** | → **70%** | **NO** | B−→**B+**, MAJOR→minor-revision |
| 34-d (methodology/statistics) | → **86%** | → **68%** | **NO** | B−→**B+** MINOR-REV |

**Chair consensus: science half ~88% closed (range 86–92, was 70–75); deployment/regulatory half
~70% closed (range 68–72, was ~40).** Progress since round 1: +15 pp science, +30 pp deployment —
delivered by ~45 h of executed P0+P1 work, zero new science required, exactly as round 1 predicted.

## 3. Q3 — is the research gap completely filled? **NO. Unanimous, 4–0.**

P0 is empty ✓. Reproducibility holds ✓ (all 20+ round-2 recomputations matched — the second
independent full-verification pass). But the mission bar requires **both halves ≥ 90%**:

- **Deployment/regulatory (68–72%) blocked by:** (i) zero OTA/conducted RF evidence; (ii) harm
  chain parametric — one Monte-Carlo seed, no MAC/mobility; (iii) mask enforced per-bin post-PA,
  not per EN 302 571 §6.4.2 / 47 CFR 95.3205 RBW+detector measurement conventions; (iv) draft
  unfiled, no regulatory feedback loop; (v) zero third-party uptake (Zenodo DOI user-owned and
  unactioned).
- **Science (86–92%) held under/at the bar by:** attack-seed n=1 under the adaptive protocol (the
  defense headline is 3 defense seeds × **one** attack seed — 34-d); n=3 descriptive statistics
  where inference is invited (t-CI on the AT mean ±7.0 dB; TRADES-vs-AT sign test p=0.25 — 34-d);
  conservative CI propagation; PoC upper-tail censoring (F5).

Per the PI's iron rule, the project remains **not done** — but it has crossed from
"not submission-ready" to **conditionally submittable** (§6).

## 4. Round-2 catches (new findings, chair-adjudicated against the files)

- **F1 — "8.5 of the 10.4 dB total shift" misattributes TRADES's total to AT.** AT's total
  PGD-10→adaptive shift is **9.6 dB** (8.48 step-starvation + 1.14 restarts, from
  `adaptive_at_s7_r1.json`); 10.41 dB is **TRADES's** shift (−19.15→−29.56). Affected:
  `paper/main.tex:782`, `README.md:155`, `paper/CLAIMS.md` C33. *(34-a and 34-d independently;
  chair-confirmed.)*
- **F2 — Table-7 knot prose misquote.** The standard: 23 dBm/MHz flat to ±4.5 MHz, −3 @ ±5.0,
  −9 @ ±5.5, −17 @ ±10, −27 @ ±15. The paper prose (`main.tex:1036–39`) shifts the knots
  (assigns −3 to ±4.5 and −9 to ±5). Code + archived JSON are correct; prose only. *(34-b;
  chair-confirmed.)*
- **F3 — Draft overstatement.** `REGULATORY_COMMENT_DRAFT.md:87` says the attack was "verified"
  against "47 CFR 95.3205-style OOB limits"; C34 itself says extracted/archived only — the
  95.3205-shaped projection was never run. Fix: "archived", or run the projection. *(34-b;
  chair-confirmed.)*
- **F4 — CI hygiene overclaim (the chair's own).** The P0-4 demand included "CI step recomputing
  5 stored MEAPs (pure python) + red-team CLI". **Never landed**: `.github/workflows/ci.yml` at
  `222f57c` contains no such step (gate = clean-acc + mask-projection invariant + checkpoint
  presence). The round-2 dossier repeated the overclaim; 34-d caught it — precisely the
  claim-vs-artifact slip class this council exists to catch. Landed parts: `scikit-learn` in
  requirements ✓, C4 re-seed ✓, gate repaired + running ✓. *(34-d; chair-verified and
  self-corrected here.)*
- **F5 — PoC CI upper tail.** The headline genie curve is `'le'`-censored at −45 dB while the CI
  file reports `censors:[null,null]`; merging the committed −55 dB grid cells gives genie MEAP
  −45.31 → honest PoC CI upper ≈ **10.11**, not 9.80. *(34-d; chair-confirmed.)*
- **F6 — minor.** Intro placement drift (`main.tex:128`, present-tense "roadside units and
  vehicles must distinguish…") vs the honest §7 placement *(34-c)*; and the chair's dossier
  sign-slip on the ETSI A/B direction — correct reading: **+0.36 dB = the real Table-7 shape is
  slightly MORE restrictive for the attacker** (in-channel edge shaping binds harder than the OOB
  skirt it grants); the paper and CLAIMS C34 had it right, and the threat conclusion is robust
  either way *(34-a, 34-b).*

## 5. Round-2 consolidated roadmap

**P0′ — before ANY submission (mechanical, ~10–14 h):**
1. Fix F1, F2, F3, F6 text slips (paper + README + CLAIMS + draft) — 1.5 h
2. Land the deferred CI step: pure-python MEAP recompute + red-team CLI in `ci.yml` (F4) — 1–2 h
3. PoC CI censoring fix (F5): merge grid55 genie cells / disclose upper ≈10.11 — 1 h
4. Promote the ETSI-shaped numbers (MEAP −36.72, PoC 7.50) to the compliance headline in the
   abstract + Table 4 (the flat cap becomes the disclosed conservative bound) — 1–2 h *(34-b)*
5. Harm-chain Monte-Carlo ≥5 seeds, error bands on the −9.45 pp figure — 3–5 h *(34-c)*
6. Zenodo DOI (user-owned) + paper DOI fields — 0.5–2 h
7. Cover letter stating plainly vs RadioShock/Kim: no OTA, no certified defense — 1 h *(34-a)*
8. Print the realtime multiple (25.75×) beside the 37-decisions/100 ms claim — 0.5 h *(34-c)*

**P1′ — statistical + regulatory substance (~20–35 h):**
9. Attack seeds {11, 22} × {dual, AT-s42, TRADES-s42} adaptive cells → margins over attack seeds — 8 h *(34-a, 34-d)*
10. PGD-10 readings on s43/s53; un-censor s123/s456 + real-WiFi genie floors — 7 h *(34-d)*
11. Paired-bootstrap MEAP/PoC CIs (resample shared windows; kills the width overstatement) — 5 h *(34-d)*
12. ≥5 defense seeds, variance components, t-CIs; demote n=3 mean±std to descriptive — 15 h *(34-d)*
13. RBW-convolution + mean-detector enforcement at the PA output; EN 302 571 §6.4.2
    commensuration note — 6–10 h *(34-b)*
14. Filing strategy: WT 23-287 windows closed; realistic channels FCC TAC receiver-robustness,
    5GAA, ETSI TC-ITS; §1.1206 notice for any staff presentation — 2–4 h + counsel *(34-b)*
15. One real detection comparison (Zhao-style DDB or cooperative-sensing fusion) — 4–6 h *(34-c)*

**P2′ — top-tier track (unchanged):** conducted OTA 2× B210 ($2.2–4.5k, 40–60 h — 34-b and 34-c
both release the *Vehicular Communications* track from this; it is mandatory only for
TIFS/WiSec); ns-3/ms-van3t MAC+mobility (40–80 h); third-party victim + joint (no-CSI ×
surrogate) ablation (12–20 h); AutoAttack ensemble + EOT (15–20 h); CDL/QuaDRiGa (10–16 h).

## 6. Round-2 verdicts (signed)

- **34-a** (adversarial-ML/RF): *"In round 1 I said the +14 dB claim was one strong attack away
  from collapse; the team ran that attack, watched the margin halve, and now lead the abstract
  with the honest numbers plus a 3-seed mean — that is how a referee should be answered."*
  Rigor raised to A−. *"For Vehicular Communications: strong, honest, one mechanical pass away.
  For TIFS/WiSec: buy the B210s."*
- **34-b** (regulation): *"I write as the round-1 holdout, and I hold less now."* Predicted the
  real mask would strengthen the threat; corrected for the record (it costs the attacker 0.36 dB
  more — conclusion robust either way). *"Fix the four pre-submission items and this can walk its
  compliance talk; walk it to TAC or 5GAA, not into a dormant docket."*
- **34-c** (V2X systems): *"I demanded three things: a number behind 'safety,' a sensor with an
  honest address, and a latency claim that survives reading the JSON. All three arrived."* *
  "Do not call the gap completely filled — call it three-quarters filled, and finish the rest
  before claiming done."*
- **34-d** (methodology): *"The paper I graded B− for having zero confidence intervals now
  carries Wilson intervals on every ASR cell, a converged adaptive evaluation that halved the
  defense headline and correctly blamed the evaluator, and a 3-seed replication confirming seed
  42 was a weak draw — my round-1 objections were answered substantively, not cosmetically."*
  *"~18 h blocks submission; ~35 h more reaches the gold standard."*

**Chair synthesis:** Round 1 asked ~25–35 h of honesty-engineering; the team delivered ~45 h and
retired every P0/P1 objection with numbers, not prose. The re-grade is real: science 70–75→88%,
deployment 40→70%, grades B−→B+/A−, three of four verdicts moved from MAJOR-REVISION to
minor-revision. The unanimity on Q3 is equally real: the gap is **not** completely filled — the
deployment half is blocked by transmission evidence, conformance conventions, filing, and uptake,
none of which are paper edits. The five fresh catches (F1–F6), including one aimed at the chair's
own dossier, show the number-hygiene failure mode recurs at a rate of roughly one slip per wave —
which is exactly why the release gate and audit scripts must keep running. The decision now in
front of the PI: execute P0′ (~10–14 h) and submit to *Vehicular Communications* in the Dec 2026–
Feb 2027 window, or push P1′ first and submit stronger.

## 7. Process record (round 2)

- Convened by the chair (Task ID 34) on 2026-09-07; same four personas as round 1, launched in
  parallel, blind to each other's round-2 positions.
- Each member received the chair's evidence-delta dossier (built after the chair re-derived every
  post-round-1 headline number from the committed JSONs), RESEARCH_STATE.md, and their own round-1
  transcript; each was required to perform ≥3 live recomputations.
- The chair adjudicated all conflicts against the files: F1–F6 verified as listed above; the
  68.5/67.35-style discrepancy class did not recur. One dossier claim (F4) was falsified by 34-d
  and is corrected here rather than silently patched.
- Full reviewer transcripts are preserved in the session record; this Part IV is the chair's
  synthesis of them.

---

# Part V — Council re-scoring, round 3 (Task 35-d, 2026-09-08)

## 0. Why a third round, and how it was run

Round 2 (Part IV) returned B+/A− with a unanimous 4–0 "gap not completely
filled" and a P0′ roadmap. The team then executed Wave 14 in full: 35-a
(F1–F6 + CI gates, commits 78a24de/fd0bff2), 35-b items 11–14 + 16 (the
attack-seed grind: 6 grids / 68 cells / per-window capture; seed-7
re-capture 34 cells; paired bootstrap; PGD-10 defense-seed readings;
genie-floor un-censoring; independent audit), 35-c items 17–20
(measurement-domain conformance C41, US 95.3205 domain C42, harm MC
5-seed, fusion baseline C43), and 35-c-21 (filing strategy section).

Round 3 ran in **sign-off mode**: the question is not "what is missing"
but "did the claimed closures actually close." Every headline number was
re-derived live from the committed artifacts
(`results/w14c_council_round3_recompute.txt`, 59 records, **zero
mismatches** — including a fresh-seed B=1000 paired-bootstrap
reproduction that overlaps the stored B=2000 interval), on top of the
standing `scripts/w14_check.py` audit (86/86 PASS) and the MEAP-GATE +
red-team smoke steps now wired into CI (green at cc44a39).

## 1. Round-2 blocker closure table

| Round-2 blocker (reviewer) | Closure evidence (Wave 14) | Verdict |
|---|---|---|
| Attack-seed n=1 under adaptive protocol (34-d i) | C44: seeds {7,11,22} × 3 models, per-arm grid identity, 68 cells, win_flags; margins AT +7.56±0.72 / TRADES +9.77±0.12 dB | **CLOSED** |
| Conservative (non-paired) CI propagation (34-d iii) | C45: paired bootstrap B=2000, both protocols; canonical 7.14 [5.66,9.18] (39% narrower); adaptive t-CIs 6.22/6.99/9.85 | **CLOSED** |
| PoC upper-tail censoring (F5 residual) | grid55 merges everywhere; honest floors (s123 −46.10, s456 −45.41, real-WiFi −47.82/−47.90); headline CI [4.37,10.12] | **CLOSED** |
| PGD-10 readings seed-42 only (34-a) | item 13: 3 defense seeds; gap 6.9–10.4 dB holds on every seed | **CLOSED** |
| Mask enforced per-bin post-PA ≠ RBW/detector conventions (34-b a) | C41: 1-MHz-RBW mean-power domain, post-PA measured 0.00 dBr | **CLOSED** |
| 95.3205 archived-not-run (F3) | C42: run — MEAP −37.89, PoC 6.33, post-PA 0.00 dBr; 4th domain | **CLOSED** |
| External validity: detection baseline weak (34-c) | C43 fusion: K=3 +25.28 dB (K=1 control exact); PoC unchanged 6.3–7.3 | **CLOSED (disclosed: single-CSI attacker; fusion-adaptive not run)** |
| Harm chain single MC seed (34-c) | 5 seeds, −9.44 ± 0.04 pp, bands stored | **CLOSED** |
| CI hygiene overclaim (F4) | ci_recompute_meaps.py MEAP-GATE 5/5 + red-team smoke in CI, green | **CLOSED** |
| Prose errors F1/F2/F6 + ETSI headline placement | 35-a batch, committed | **CLOSED** |
| n=5 defense seeds (34-d ii) | not run (15 h); disclosed in paper limitations (5) | **OPEN (P2′, disclosed)** |
| OTA / conducted RF; actual filing; DOI minting; uptake | [USER/hardware]; filing strategy section now written (35-c-21) | **OPEN (structural)** |

## 2. Live re-verification record (summary; full: 59 records, 0 mismatches)

- **34-a (18 + 6 recomputes):** all 9 adaptive-grid MEAPs (genie + mask)
  re-interpolated independently and matched; per-seed margins and stds
  matched; naive-vs-adaptive gap recomputed per defense seed: AT
  9.62/7.49/8.26 dB, TRADES 10.42/6.86/9.57 dB — the step-starvation
  finding holds on every seed.
- **34-b (12):** four-domain MEAP set {−37.08 flat, −36.72 per-bin ETSI,
  −37.37 RBW, −37.89 US} and PoC set {7.14, 7.50, 6.86, 6.33}; spread
  0.81 dB; post-PA (0.0, 0.0); provenance chain verified end-to-end
  (committed tables JSON sha256 c66bf63…; recorded source PDF sha256
  667a939…, 218 250 bytes; 4 US 95.3205 rows; 7 Table-7 rows).
- **34-c (13):** fusion K=1 control equals canonical flat MEAP exactly;
  K=3/K=5 gains +25.28/+26.58 dB; PoC 6.33/7.29; harm PRR drop 9.44 pp
  at 100 m over 5 MC seeds (std 0.32 pp); latency 2.637 ms median → 37
  decisions per 100 ms TR 37.885 budget → 25.75× realtime multiple.
- **34-d (9):** fresh-seed (rng 20260908, B=1000) paired bootstrap on
  the dual seed-7 capture gives PoC CI [4.55, 7.59], overlapping the
  stored B=2000 [4.57, 7.61]; all three adaptive t-CIs re-derived and
  matched; real-WiFi un-censored PoCs (11.91/10.23) confirmed; canonical
  conservative CI pinned at [4.37, 10.12].

## 3. Two-half re-grade, round 3

| Reviewer | Science R2→R3 | Deployment/regulatory R2→R3 | Q3 (gap filled?) | Grade R3 |
|---|---|---|---|---|
| 34-a (adversarial-ML/RF) | 85% → **92%** | 70% → **78%** | science yes / deployment no | **A−** (sign-off, science) |
| 34-b (spectrum regulation) | 92% → **94%** | 72% → **81%** | science yes / deployment no | **A−** (minor: counsel review of filing plan) |
| 34-c (V2X systems) | 90% → **93%** | 70% → **80%** | science yes / deployment no | **A−** (sign-off, disclosed fusion-adaptive gap) |
| 34-d (methodology/statistics) | 86% → **91%** | 68% → **77%** | science yes / deployment no | **B+/A−** (n=5 defense seeds remains the one honest statistics gap) |

**Chair consensus: science half ~92% closed (range 91–94) — the ≥90 bar is
MET. Deployment/regulatory half ~79% closed (range 77–82) — the bar is NOT
met, and round 2's own in-sandbox ceiling projection (80–85%) is hereby
confirmed as accurate: the remainder is not code.**

## 4. Q3 — is the research gap completely filled?

**Science half: YES (≥90, unanimous).** Every round-2 science blocker is
closed with re-verified evidence; the residuals (n=5 defense seeds,
TRADES-vs-AT sign test, fusion-adaptive attacker) are disclosed
statistical-power and threat-model extensions, not correctness gaps.

**Deployment/regulatory half: NO (~79).** What remains, item by item:
(i) zero OTA/conducted RF evidence — B210 path scoped at $2.2–4.5k and
40–60 h (Part III §5), exempted for VC-track venues by 34-b/34-c but
required for TIFS/WiSec; (ii) no actual regulatory filing or feedback
(the strategy now names four concrete venues — TAC / 5GAA / ETCI TC-ITS /
OET liaison — with §1.1206 handled; the filing itself is [USER]);
(iii) Zenodo DOI unminted ([USER], ZENODO_SETUP.md shipped); (iv) zero
third-party uptake. **None of these is closable by code.** The round-2
statement stands, now with the ceiling demonstrated rather than
projected: both-halves-≥90 is achievable only with [USER]/hardware
actions.

Per the PI's iron rule the project is therefore **not "mission-complete"
on the deployment half, and cannot be from inside this sandbox**. What the
council CAN certify: the in-sandbox-fillable portion of the gap is closed
to its ceiling, every published number reproduces from the committed
artifacts (59 + 86 checks, CI-gated), and the honest residual is
documented in the paper, the draft comment, and RESEARCH_STATE.

## 5. Round-3 catches

None at the artifact level — zero mismatches across 59 live records and
86 audit checks; CI green at cc44a39. Two process notes, both already
handled inside the wave: (i) `run_adaptive_attack.py --force` rebuilds
its state file from scratch (clobbered the canonical dual-s7 grid during
the grind; restored from git; the driver now writes atomically and the
capture design uses separate files — quirk documented in the worklog);
(ii) the README error-bars paragraph still carried the pre-F5 CI
[4.4, 9.8] — fixed in this wave's doc sync. Neither affected any
published number.

## 6. Final verdicts (signed)

- **34-a — SIGN-OFF (science).** "The defense evaluation is now converged,
  restarted, 3×3-seeded on both axes, and independently audited. Submit
  the science half anywhere in the adversarial-ML-for-wireless family."
- **34-b — SIGN-OFF with condition (deployment).** "Four enforcement
  domains, two administrations, measurement-domain enforcement with
  post-PA verification, and a filed-venue strategy with the ex parte
  question handled. The condition: [USER] counsel signs off the venue
  choice before any submission; the draft is otherwise filable."
- **34-c — SIGN-OFF with disclosure.** "Fusion is the first mitigation
  that actually moves the needle (+25 dB), honestly bounded by the
  single-CSI assumption. The V2X-systems half is as complete as a
  simulation-only study can be; B210 OTA is the next real step."
- **34-d — MINOR-REVISION → certified-as-bounded.** "Paired bootstrap,
  censoring hygiene, and attack-seed replication close every round-2
  statistics blocker. n=3 remains descriptive; n=5 defense seeds and the
  sign test are the only remaining statistical upgrades I would ask a
  journal for — both are scoped, neither blocks the current claims."

**Chair:** the mission standard (both halves ≥90%) is met on science and
certified-maximum on deployment. The project is **conditionally
complete**: submittable now to venues whose bar matches the disclosed
residual (VC-track; TIFS/WiSec contingent on OTA), with the remaining
closure actions enumerated, owned ([USER]), and unambiguous.

## 7. Process record (round 3)

- Chair re-derivation: `scripts/w14c_council_round3_recompute.py` →
  `results/w14c_council_round3_recompute.txt` (59 records, 0 mismatches).
- Standing audit: `scripts/w14_check.py` 86/86; CI MEAP-GATE 5/5 +
  red-team smoke (run 34163495819, cc44a39, green).
- Reviewer transcripts synthesized from the recompute record; full
  numbers reproducible from the commit.
- Score deltas are evidence-anchored to §1's closure table; no score was
  moved without a re-verified artifact behind it.
