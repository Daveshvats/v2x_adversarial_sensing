# Wave-11 Independent Audit Report — PA Reality Check (Task 19-F)

**Auditor:** W11 auditor (independent code+paper auditor, Task ID 19-F)
**Date:** 2026-09-06
**Scope:** Wave-11 additions — `scripts/run_papr_pa.py`, `results/papr_pa_results.json`,
paper subsection `\label{sec:papr}` + abstract/Conclusion/Limitations/Related-Work/bibliography
edits in `paper/main.tex`, CLAIMS.md C27–C32.

**Method:** all numbers verified by INDEPENDENT re-derivation in fresh checker scripts
`scripts/w11_check1_papr.py`, `w11_check2_meap.py`, `w11_check3_regrowth.py`,
`w11_check3b_aware.py`, `w11_check5_claims.py` (own numpy/torch implementations of
Rapp / PSD / PAPR / MEAP-interpolation; only `src/` modules and the canonical
`build_eval_set` from `scripts/run_attack.py` imported; NO PA function of
`run_papr_pa.py` imported). All runs: `/home/z/.venv/bin/python`, `torch.set_num_threads(2)`.

---

## Executive summary

The Wave-11 experiment is **scientifically sound and its stored numbers are real**:
every PAPR figure, every stored MEAP, every MEAP-shift, the regrowth order of magnitude,
the IBO\* search, and the power-control energy convention reproduce under independent
re-implementation. The paper-to-JSON trace is almost perfect — **one factual mismatch**
(the "0 % pass lenient" sentence), **one untraced number** (post-PA PAPR 4.9 dB — which I
re-derived as 4.93 dB, so it is correct but should be stored), and **one broken process
citation** (CLAIMS C31 cites "worklog Task 19", which was never written; Wave-11 is also
not yet committed to the repo).

**Overall verdict: PASS — conditional on fixing F5a (MAJOR) and the three MINOR items
before ship.**

---

## Findings table

| ID | Check | Severity | Status | Finding |
|----|-------|----------|--------|---------|
| F1 | 1 (PAPR benign) | INFORMATIONAL | PASS | All 4 benign classes reproduce with own numpy PAPR on mirrored rng(123)/200-window generation: PC5 6.255 (JSON 6.26), 11p 8.766 (8.77), WiFi 8.693 (8.69), Noise 9.019 (9.02); max dev 0.005 dB « 0.15 dB tolerance. p95s match too (6.95/9.89/9.67/10.32). |
| F2 | 2 (MEAP arms) | INFORMATIONAL | PASS | All 9 stored MEAPs recompute EXACTLY with own interpolation (max dev 0.00000 dB): control −37.0000, pa_p3_ibo0 −36.4012, pa_p3_ibo6 −37.0899, pa_p3_ibo12/13 −37.0000, pa_p2_ibo6/15 −37.0000, aware ibo13 −38.2318, aware ibo0 −38.3215. All `shift_vs_control_db` = round(meap−control, 2) (±0.002 from rounding). Curves monotone, lengths = 19 = psr grid. PA-aware stats match paper (100 % strict, −60.24 dBr p95; ibo0 non-compliant −12.79 dBr). |
| F2b | 2 (control sanity) | INFORMATIONAL | PASS | Control curve bit-identical to the canonical `attack_results_urban_untargeted.json` mask curve at all 3 shared PSRs (−30: 39.33, −15: 88.33, 0: 100.00); control MEAP −37.00 vs canonical −37.0833 → 0.083 dB, supporting the paper's "0.1 dB of the coarser canonical grid". Internal IBO\* consistency holds (pa_p3_ibo13 pass 0.9533 ≥ 0.95 at IBO\*=13; pa_p2_ibo15 pass 0.97 at IBO\*=15; pa_p3_ibo12 0.8267 < 0.95 below it). |
| F3 | 3 (fresh regrowth point) | INFORMATIONAL | PASS | Fresh 30-window run (urban, n_eval=10, seed 7, PGD-10 mask-constrained, PSR −36, own Rapp p=3 IBO 6 dB on per-window mean power, own 2048-pt-FFT peak-OOB-minus-peak-in-band dBr): **p95 = −19.92 dBr** (p50 −22.44) vs JSON 300-window −19.32 and task window [−22, −17]. Pre-PA mask excess ≈ −135 dBr (projection nulls OOB); pre-PA PAPR 11.11 dB (JSON 11.05). dBr regrowth is scale-invariant under per-window IBO as claimed (max |Δ| ≤ 5.7e−6 dBr = float32 noise). |
| F4 | 4 (energy audit) | INFORMATIONAL | PASS | After renormalizing the Rapp output to the per-window budget: 10·log10(E_out/p_budget) = **0.000000 dB** (mean; max |·| = 3.7e−7 dB) — power-control convention implemented exactly as documented in the JSON `config.power_control` string; the dBr mask-excess metric is unchanged by renormalization (scale-invariant ratio). |
| F5a | 5 (paper vs JSON) | **MAJOR** | **FAIL** | **Paper says 0 %, JSON says 8 %.** `main.tex` line 836 (and CLAIMS C28): "0\,\% of windows pass even the lenient $-28$\,dBr shoulder" at the ~6 dB backoff (p=3). `papr_pa_results.json` `regrowth["p=3/ibo=6"].pass_lenient = 0.08` → **8 % of windows DO pass the lenient gate (92 % fail)**; p=2 is 3.67 %. The 0 % figure matches the *strict* gate (`pass_strict = 0.0`), i.e. the sentence conflates strict with lenient. The qualitative conclusion (regrowth breaks compliance at operating backoff) survives, but the printed fraction is wrong; fix wording to "92 % of windows fail even the lenient shoulder" or cite the strict gate. |
| F5b | 5 (paper vs JSON) | MINOR | FAIL (traceability) | **"post-PA PAPR 4.9 dB" (paper line 862, CLAIMS C30) is not stored anywhere in `papr_pa_results.json`** (no PAPR field in any `pa_aware_*__meap` entry or regrowth row). Independent re-derivation on a fresh 30-window PA-aware IBO-0 run (`w11_check3b_aware.py`, own chain: project → own Rapp → own renorm) gives **4.93 dB** — the number is correct, but it violates the paper's stated NUMBER POLICY ("every number in this file traces to a results JSON"). Fix: add `post_tx_papr_mean_db` to the `run_arm` stats and re-merge, or drop the figure. |
| F5c | 5 (paper wording) | MINOR | WARN | "the regrowth pedestal rises ∼9 dB above the **strongest** legitimate neighbor's post-PA skirt (PC5: −28.7 dBr)" (line 838) / C28 "above **loudest** benign neighbor": PC5 has the **quietest** post-PA skirt of the three benign classes (−28.67 vs 11p −21.73 vs WiFi −12.65 dBr); the 9.35 dB margin holds only vs PC5, is only **2.4 dB vs 11p**, and the attack pedestal is **6.7 dB below WiFi's** skirt. "Strongest/loudest neighbor" reads backwards — should be "the cleanest (lowest-skirt) benign neighbor (PC5)". |
| F5d | 5 (abstract) | INFORMATIONAL | PASS (note) | Abstract cites "inside a −40 dBr mask at 13 dB backoff" — that is the p=3 IBO\*; p=2 needs 15 dB (body discloses both 13/15). Acceptable rounding-to-best-case in the abstract; optionally "13–15 dB". |
| F5e | 5 (setup wording) | INFORMATIONAL | PASS (note) | Paper says IBO "swept over 0–16 dB"; the stored config sweep is [0,2,…,12] — the 0–16 range is the script's fine 1 dB IBO\* search grid (`arange(0,16.1,1)`, not stored in the JSON config). |
| F5f | 5 (Sec. 3 legacy) | INFORMATIONAL | PASS (note) | Sec. 3 still says "measured: 6.1 dB PC5 vs 8.8/8.6 dB for the OFDM classes" — the older, untraced measurement (W9-B already flagged `check_waveforms.py` as not in the package). Wave-11 now provides a committed benign-PAPR measurement (6.26/8.77/8.69) agreeing to <0.2 dB; the Sec. 3 sentence should now cite it. |
| F5g | 5 (paper vs JSON, rest) | INFORMATIONAL | PASS | All remaining sec:papr / abstract / Conclusion / Limitations numbers trace exactly: 11.05 (p95 12.8), benign 6.26/8.77/8.69/9.02, −19.3 dBr (−19.32), +1.0 dB (1.02), −28.7 dBr (−28.67), IBO\*=13/15, MEAP shifts +0.6/−0.1/0.0 (0.60/−0.09/0.0/−0.0), PA-aware −38.2 (−38.2318), −1.2 shift (−1.23), 100 % strict (1.0), −60 dBr (−60.24), control −37.0 (−37.000…), 300 windows / seed 7 / PGD-10 / psd_margin 2 (config), genie −44.2 canonical (−44.2248), "11.1 vs 6.3–9.0" (11.05 vs 6.26–9.02), H1 confirmed flag, H2-falsified / H3-not-triggered characterizations consistent with stored shifts (≤ +0.6 dB) and IBO\* ≤ 16. |
| F6 | 5 (bibliography) | INFORMATIONAL | PASS | `anand2008` (cited line 220 + table row 259), `ambhika2024` (line 221), `itsa2024` (line 110), `5gaa2024` (line 112) all appear BOTH as `\cite` in text and as `\bibitem` in the bibliography; PUEA paragraph and threat-table row present; no orphan `\bibitem` entries anywhere. |
| F7 | 6 (CLAIMS C27–C32) | INFORMATIONAL | PASS (with caveats) | C27 ✓ (all numbers), C29 ✓ (shifts +0.60/−0.09/0.00, control −37.00, canonical −37.08 verified vs `attack_results_urban_untargeted.json`), C32 ✓ (citations + band-status facts). C30 ✓ except the untraced 4.9 dB (F5b). C28 carries the 0 %-lenient error (F5a) and the "loudest neighbor" wording (F5c). C31's technical content matches the code: both bug descriptions are in `run_papr_pa.py`'s `chain()` docstring (0/0 NaN-at-zero-init → argmax-of-NaN "13 dB improvement"; asat sized to window energy → +33.11 dB phantom backoff, "same unit class as the Wave-3 PSR bug") and the merge-on-write clobber fix is in `flush()`'s NOTE. |
| F8 | 6 (process) | MINOR | FAIL (process) | **`worklog.md` has NO "Task ID: 19" entry** — the Wave-11 main agent never logged its work, yet CLAIMS C31 cites "worklog Task 19" as a backing artifact (broken citation; the "66.7 % = argmax-of-NaN signature, 20/30" detail and the "D2 physical-sanity gate" narrative trace only to that missing entry). Additionally Wave-11 artifacts (`run_papr_pa.py`, `papr_pa_results.json`, paper/CLAIMS edits) are **not committed to `v2x_repo`** (HEAD `f1e7f8c` predates Wave 11; no `run_papr_pa.py`/`papr_pa_results.json` anywhere in the repo tree) — they exist only in `download/v2x_phase1/`. |
| F9 | 7 (regression) | INFORMATIONAL | PASS | Canonical price-of-compliance headline intact: `-37.1` appears at lines 580/588/644/696/751/776/790, `-44.2` at 588/695/776/789/825; Table 2 row "urban & untargeted & −44.2 dB & −37.1 dB & 7.1 dB" verbatim; canonical JSON genie −44.2248 / mask −37.0833 / PoC 7.1414. No accidental edits to the price-of-compliance section. |

---

## What was independently re-derived (evidence)

| Quantity | Own value | Stored/paper value | Verdict |
|---|---|---|---|
| PC5 benign PAPR mean (200 win, rng 123) | 6.255 dB | 6.26 | PASS (0.005 dB) |
| 802.11p benign PAPR mean | 8.766 dB | 8.77 | PASS |
| WiFi benign PAPR mean | 8.693 dB | 8.69 | PASS (0.003 dB) |
| Noise benign PAPR mean | 9.019 dB | 9.02 | PASS |
| Control MEAP (own interp, threshold 20 %) | −37.0000 dB | −37.0000 | PASS (exact) |
| pa_p3_ibo0 / ibo6 MEAP | −36.4012 / −37.0899 | −36.4012 / −37.0899 | PASS (exact) |
| pa_aware ibo13 / ibo0 MEAP | −38.2318 / −38.3215 | −38.2318 / −38.3215 | PASS (exact) |
| Control vs canonical curve (shared PSRs) | 39.33/88.33/100.00 | 39.33/88.33/100.00 | PASS (bit-identical) |
| Fresh regrowth p95 (30 win, own Rapp p=3 IBO 6) | **−19.92 dBr** | −19.32 (300 win) | PASS (order/sign; in [−22,−17]) |
| Regrowth dBr scale-invariance (×0.5/×0.123/×3) | ≤ 5.7e−6 dBr | claim: invariant | PASS |
| Energy ratio after budget renorm | 0.000000 dB (max 3.7e−7) | 0.00 ± 0.01 dB | PASS |
| PA-aware IBO-0 post-PA PAPR (30 win, own chain) | 4.93 dB | paper: 4.9 (NOT in JSON) | number correct, traceability FAIL |
| Pre-PA mask delta PAPR (30 win) | 11.11 dB | 11.05 (300 win) | PASS (supporting) |

## Number mismatches found (paper/CLAIMS vs JSON)

1. **Paper (sec:papr, line 836) and CLAIMS C28 say "0 % of windows pass even the lenient −28 dBr shoulder" at IBO 6 / p=3. JSON says `pass_lenient = 0.08`, i.e. 8 % pass / 92 % fail.** (The 0 % belongs to the *strict* gate.) — MAJOR.
2. Paper (line 862) and CLAIMS C30 say "post-PA PAPR 4.9 dB" for the PA-aware IBO-0 arm; **no such field exists in the JSON** (independently re-derived: 4.93 dB — correct but untraced). — MINOR.
3. CLAIMS C31 cites "worklog Task 19" as a backing artifact; **worklog.md contains no Task ID 19 entry**. — MINOR (process).
4. Wording (paper line 838 + C28): "strongest/loudest legitimate neighbor's post-PA skirt (PC5)" — PC5 is the *cleanest* (−28.67 dBr); margins are +2.4 dB vs 11p and −6.7 dB vs WiFi. — MINOR.

## Required actions (pre-ship)

1. **(MAJOR, F5a)** Fix the lenient-gate sentence in `main.tex` + CLAIMS C28: replace "0 % of windows pass even the lenient −28 dBr shoulder" with "92 % of windows fail even the lenient −28 dBr shoulder (0 % pass the strict −40 dBr gate)".
2. **(MINOR, F5b)** Store `post_tx_papr_mean_db` in `run_arm`'s stats (and re-merge the arm, or quote the re-derived 4.9 dB with a checker citation) so the paper's 4.9 dB traces to the JSON.
3. **(MINOR, F5c)** Reword "strongest legitimate neighbor" → "the cleanest (lowest-skirt) benign neighbor (PC5)" in paper + C28.
4. **(MINOR, F8)** Append the Wave-11 main-agent worklog entry (Task ID 19) and commit the Wave-11 artifacts to the repo branch; update C31's citation if the entry number changes.
5. (Optional, F5f) Align Sec. 3's PAPR sentence with the committed Wave-11 measurement (6.26/8.77/8.69).

## Checker scripts (added by this audit)

- `scripts/w11_check1_papr.py` — CHECK 1 (benign PAPR, own numpy, mirrored rng).
- `scripts/w11_check2_meap.py` — CHECK 2 (all MEAP/shift/censor recompute, own interpolation).
- `scripts/w11_check3_regrowth.py` — CHECK 3 + 4 (fresh PGD-10 30-window regrowth point, own
  Rapp/PSD, scale-invariance, budget-energy audit).
- `scripts/w11_check3b_aware.py` — supplementary: fresh PA-aware IBO-0 chain (own Rapp/renorm)
  reproducing the untraced 4.9 dB post-PA PAPR claim.
- `scripts/w11_check5_claims.py` — CHECK 5 + 6 + 7 (automated paper/CLAIMS/bibliography/
  regression cross-check; 43 assertions).
