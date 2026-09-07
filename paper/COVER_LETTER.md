# Cover Letter — Vehicular Communications submission (draft)

**Manuscript:** *The Price of Compliance: Emission-Mask-Constrained Adversarial Attacks on
Deep Learning Spectrum Sensing in the 5.9 GHz ITS Band*
**Target window:** December 2026 – February 2027 (per research-council round 2, 2026-09-07)
**Status:** DRAFT — not submitted anywhere (decision log D4). Do not send until the Wave-14
P0′ items and the PI's go/no-go are complete.

---

Dear Editors,

We submit "The Price of Compliance," a study of whether an adversary can defeat
learning-based spectrum sensing in the 5.9 GHz ITS band while remaining fully inside its
regulatory emission constraints — and what that capability implies for the coexistence
architecture now being standardized as the DSRC sunset runs out (89 FR 100838; two-year
timeline from February 2025).

**Why this fits Vehicular Communications.** The paper is, to our knowledge, the first to make
the attacker's *regulatory feasible set* — frequency allocation, emission mask, and power
budget — the constraint set of the adversarial optimization itself, and to report robustness
in link-budget units (MEAP, the minimum received power at which a 20% conditional-ASR
threshold is crossed, and the Price-of-Compliance, its delta against an unconstrained
attacker). This venue published the clean-task baseline our attack breaks (Girmay et al.,
2023), the FCC's 2024 Second Report & Order has made 5.9 GHz coexistence a live standards
question, and our harm chain lands on the metric this community cares about: BSM packet
reception ratio (−9.5 pp at 100 m inter-vehicle distance under the attacked sensing, 94.5
additional lost BSMs per 1000).

**Key results (all Wilson-CI-carrying, all recomputed by four independent reviewers):**
- An undefended sensing CNN is broken by a rule-abiding transmitter tens of dB below the
  victim signal level; the emission mask costs the attacker only 5–7 dB (untargeted,
  7.5 dB under the real ETSI EN 302 571 Table-7 template) and 16–18 dB for targeted cloaking.
- Mask-matched adversarial training and TRADES retain +9.8 ± 2.8 dB and +11.5 ± 1.8 dB of
  adaptive-attack margin (3 defense seeds; PGD-50 × 5 restarts) — roughly half of what a
  non-adaptive PGD-10 evaluation suggests, a protocol finding of independent interest.
- The defense-seed spread (3.5–5.5 dB) exceeds the undefended training-seed spread 3–5×,
  disqualifying single-seed defense evaluation for certification-style claims.

**What we do NOT claim (scope, stated plainly).** No over-the-air transmission of the attack
(a conducted B210 replication is the planned follow-up); no certified defenses; the harm-chain
link-budget layer is parametric and disclosed (not an ns-3 study); the real-signal stress test
uses 5.24 GHz U-NII-1 captures (n=74 eval windows, one class, honestly labeled — not 5.9 GHz
spectrum). Every number in the manuscript is traced to a committed result JSON through the
released claim-verification matrix, and a CI gate recomputes the headline MEAPs from the
artifact on every push.

**Relation to prior work.** Kim & Sagduyu (TWC 2022) and RadioShock (2026) attack wireless
classifiers over the air but under generic power budgets, not regulatory masks; Zheng et al.
(2026) attack wideband sensing without compliance constraints; Liu et al. (2022) use jamming
and poisoning rather than in-mask evasion. None constrain the adversary to its spectrum rules
or report a power-domain price of compliance; that is our contribution, together with the
finding that the price is set by the *allocation*, not the mask's fine shape (real ETSI
Table-7 template shifts the headline by +0.36 dB).

The full artifact (code, checkpoints, standards extracts, 30+ result JSONs, 11-wave audit
trail, and the four-reviewer research-council record) is released for referee reproduction.

Sincerely,
The authors
