#!/usr/bin/env python3
"""w14_docs_sync.py — Wave-14 additions to paper/CLAIMS.md (C42–C45) and
README.md (attack-seed/bootstrap/US-domain/fusion blocks), plus a stale
README CI-number fix ([4.4, 9.8] -> [4.4, 10.1] + paired bootstrap)."""

# ---------------- CLAIMS.md ----------------
PATH = "paper/CLAIMS.md"
src = open(PATH).read()

ANCHOR = "| VERIFIED (protocol-identical to etsi_mask_results.json; resumable/idempotent runner) |"
assert src.count(ANCHOR) == 1

NEW_ROWS = ANCHOR + """
| C42 | US 47 CFR 95.3205-shaped conformance domain (Wave 14, item 18): the C-V2X OBU unwanted-emissions limits (−16 dBm/100 kHz within 1 MHz of the channel edge, −13 dBm/MHz to 5 MHz, −16 dBm/MHz to 30 MHz; archived eCFR text, 89 FR 100855 + 90 FR 5724 correction) enforced in the mean-power measurement domain (100-kHz RBW near the edge, 1-MHz RBW elsewhere, overlapping centres): compliant MEAP −37.89 / PoC 6.33 — the most permissive of the FOUR enforcement domains (0.81 dB below flat; the region-flat −13 dBm/MHz middle permits in-band concentration the knot-shaped ETSI template forbids). Post-PA worst excess 0.00 dBr on every window/cell. PoC headline robust across all four domains 6.33–7.50 dB (within ±0.9 dB of flat) | results/conformance_us_results.json; scripts/run_conformance_us.py; data/standards/en302571_tables.json (fcc_47cfr_95_3205 rows) | VERIFIED (same protocol as conformance_results.json; w14_check.py G-block) |
| C43 | Cooperative-sensing fusion baseline (Wave 14, item 20, council 34-c external validity): K-receiver majority vote over independent receiver chains, same compliant delta through each receiver's attacker channel; attacker CSI single-receiver (fusion-adaptive attacker NOT evaluated — disclosed). K=1 control reproduces the canonical MEAPs exactly (−37.08/−44.22 dB). K=3 raises the compliant MEAP −37.08 → −11.80 (+25.28 dB detection gain), K=5 → −10.50 (+26.58 dB); PoC stays 7.14/6.33/7.29 dB across K=1/3/5 — fusion defeats the absolute power advantage but leaves the relative mask penalty (and every compliance conclusion) unchanged | results/w14b_fusion_baseline.json; scripts/run_fusion_baseline.py | VERIFIED (K=1 control exact; w14_check.py F-block) |
| C44 | Attack-seed replication of the adaptive margins (Wave 14, item 12 — closes council 34-d blocker i): attack seeds {11,22} on the canonical seed-42 trio with PER-ARM grid identity to the s7 files (dual genie [−50..−30]/mask [−45..−20]; AT genie [−45..−25]/mask [−35..−10]; TRADES mask [−35..−5]), 6 grids, 68 cells, all with per-window capture, all MEAPs uncensored. Margins over attack seeds: AT +7.56 ± 0.72 dB (range 1.33), TRADES +9.77 ± 0.12 dB (range 0.24) — 4–23× SMALLER than the defense-seed spreads (±2.80/±1.75): the certification-relevant seed lottery is the defense training seed. Adaptive PoC 3-seed means: undefended 6.22 (range 0.67), AT 6.99 (0.16), TRADES 9.85 (1.39) | results/w14_attack_seed_margins.json (grid-identity checks all pass); results/adaptive_{dual,at,trades}_s{11,22}_r5.json (win_flags 300/cell); scripts/w14_attack_seed_margins.py | VERIFIED (independent audit w14_check.py A/C blocks, 86 checks ALL PASS) |
| C45 | Per-window capture + paired-bootstrap CIs (Wave 14, items 11+14 — closes council 34-d iii): (a) canonical PGD-10 captures for model seeds {7,123,456} reproduce every stored cond-ASR exactly; (b) seed-7 canonical ADAPTIVE cells re-captured in separate files (34 cells, continuity all-match, originals untouched); (c) genie floors un-censored by extending grids to −55 dB (s123 −46.10, s456 −45.41, real-WiFi frozen −47.82 / finetuned −47.90 → real-signal PoC 11.91 / 10.23 dB, larger than the synthetic 7.1 — synthetic evaluation understates what compliance buys on real captures); (d) paired-bootstrap CIs (B=2000, windows resampled with replacement, same indices both arms): canonical PoC 7.14 CI [5.66, 9.18] vs conservative [4.37, 10.12]; 3-seed model-mean 7.03 t-CI [6.64, 7.43]; adaptive 3-attack-seed PoC means 6.22/6.99/9.85 dB with t-CIs [5.30, 7.13]/[6.79, 7.19]/[8.09, 11.61] | results/w14b_bootstrap_cis.json; results/w14b_perwindow_s{7,123,456}.json; results/w14b_perwindow_adaptive_{dual,at,trades}_s7.json; scripts/w14b_bootstrap_cis.py + w14b_adaptive_capture.py | VERIFIED (independent audit w14_check.py B/D/E blocks) |"""

src = src.replace(ANCHOR, NEW_ROWS)
open(PATH, "w").write(src)
print("CLAIMS.md: C42-C45 appended (4 rows)")

# ---------------- README.md ----------------
PATH = "README.md"
src = open(PATH).read()

# stale CI number fix (missed in 35-a)
OLD_CI = """**Error bars (Wave 12, C36)**: Wilson 95% binomial CIs on every conditional-ASR
cell across 15 result files; headline PoC 7.1 dB carries CI [4.4, 9.8] (wider
than the 1.06 dB 3-seed spread — the stability argument needed this
context). `results/w12_confidence_intervals.json`."""
NEW_CI = """**Error bars (Wave 12, C36 + Wave 14, C45)**: Wilson 95% binomial CIs on
every conditional-ASR cell across 15 result files; headline PoC 7.1 dB
carries conservative CI [4.4, 10.1] (genie floor un-censored via the
−55 dB grid extension) and a **paired-bootstrap CI [5.66, 9.18]**
(B=2000, genie/mask arms resampled on shared windows — 39% narrower than
the conservative bound propagation; 3-seed model-mean 7.03 dB, t-CI
[6.64, 7.43]). `results/w12_confidence_intervals.json` +
`results/w14b_bootstrap_cis.json`."""
assert src.count(OLD_CI) == 1
src = src.replace(OLD_CI, NEW_CI)
print("README.md: C36 error-bar paragraph updated")

ANCHOR2 = """window of every PSR cell. The PoC headline is robust across all three
enforcement domains within ±0.5 dB. `results/conformance_results.json` +
`src/conformance.py`."""
assert src.count(ANCHOR2) == 1
NEW_BLOCK = ANCHOR2 + """

**US 95.3205-shaped domain — the fourth enforcement regime (Wave 14, C42)**:
the C-V2X OBU unwanted-emissions limits (47 CFR 95.3205, archived eCFR
text) enforced in the mean-power measurement domain (100-kHz RBW near the
band edge, 1-MHz elsewhere): compliant MEAP **−37.89 dB / PoC 6.33** —
the most permissive of the four domains for the attacker (0.81 dB below
flat; the region-flat −13 dBm/MHz middle region permits in-band
concentration). Post-PA worst excess 0.00 dBr on every window/cell.
**PoC headline robust across all four enforcement domains: 6.33–7.50 dB.**
`results/conformance_us_results.json`.

**Attack-seed replication (Wave 14, C44 — council 34-d blocker i)**: the
adaptive margins are now replicated across attack seeds {7, 11, 22} on the
canonical seed-42 trio with per-arm grid identity and per-window capture
(6 grids, 68 cells, all uncensored; plus a 34-cell seed-7 re-capture that
reproduces every stored cond-ASR exactly). **Margins over attack seeds:
AT +7.56 ± 0.72 dB, TRADES +9.77 ± 0.12 dB — 4–23× smaller than the
defense-seed spreads (±2.80 / ±1.75)**: the certification-relevant seed
lottery is the defense training seed, not the attack seed. Adaptive PoC
3-seed means: undefended 6.22 / AT 6.99 / TRADES 9.85 dB, with
paired-bootstrap + t CIs. `results/w14_attack_seed_margins.json` +
`results/w14b_bootstrap_cis.json` + the six
`adaptive_{dual,at,trades}_s{11,22}_r5.json` grids.

**Cooperative-sensing fusion baseline (Wave 14, C43 — council 34-c)**:
K-receiver majority vote (independent chains, same compliant delta;
attacker CSI single-receiver, fusion-adaptive attacker disclosed as NOT
evaluated). **K=1 control reproduces the canonical MEAPs exactly; K=3
raises the compliant MEAP by +25.3 dB (−37.08 → −11.80), K=5 +26.6 dB,
while the PoC stays 6.3–7.3 dB across K** — fusion buys 25 dB of
detection headroom but leaves the price of compliance (and every
compliance conclusion) unchanged. `results/w14b_fusion_baseline.json`.

**PGD-10 defense readings now 3-seed (Wave 14, item 13)**: PGD-10 on
AT/TRADES defense seeds 43/53 — AT mask MEAP −23.1/−19.7/−20.9 dB,
TRADES −19.1/−19.2/−18.5 dB across seeds {42,43,53}; the
naive-vs-adaptive gap (6.9–10.4 dB) holds on every seed.
`results/{at,trades}_defense_results_s{43,53}.json`.

**Honest real-WiFi floors (Wave 14, item 14)**: genie grids extended to
−55 dB resolve the floor censoring — real-signal PoC is 11.91 dB (frozen)
/ 10.23 dB (finetuned) with genie floors at −47.8/−47.9 dB (3.6 dB deeper
than synthetic): **synthetic evaluation understates what compliance buys
against real captures**. `results/w12_confidence_intervals.json`
(C21 pairs, un-censored)."""
src = src.replace(ANCHOR2, NEW_BLOCK)
open(PATH, "w").write(src)
print("README.md: Wave-14 blocks added")
