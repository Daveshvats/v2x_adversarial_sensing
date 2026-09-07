# DRAFT TECHNICAL COMMENT — receiver-side adversarial robustness of
# coexistence sensing in the 5.9 GHz transition

**Status: DRAFT v2 (2026-09-07). Not submitted. Filable-form: the structure,
filer block, citation apparatus, and scope language below follow standard
ex parte / technical-comment conventions; the bracketed fields must be
completed by the filer before submission.**

---

## Filer block

> FILER: [Name of filer / organization]
> CONTACT: [Name, title]
> ADDRESS: [Street, city, state, ZIP, country]
> EMAIL: [email address]
> DATE: [filing date]
> FILING CHANNEL: [fill at filing — see "Where to file" note below]
> DOCKET: [fill at filing]

**Where to file (guidance, not legal advice):** the 5.9 GHz transition
proceeding (WT Docket No. 23-287, FCC 24-123, adopted Nov. 20, 2024,
published 89 FR 100838, Dec. 13, 2024) closed its principal comment windows
when the rules took effect Feb. 11, 2025. Plausible channels for this
contribution are: (a) an ex parte presentation to the Commission or staff in
a relevant active docket; (b) a technical contribution to 5GAA or the
C-V2X/DSRC ecosystem working groups; (c) an input to ETSI work items on
receiver-side resilience or to the FCC's Technological Advisory Council
receiver-robustness workstream; (d) a comment in any future Further Notice
addressing 5.9 GHz coexistence or machine-learning-based spectrum access.
The authors should confirm the channel and docket with counsel.

## Regulatory grounding (citations used in this comment)

1. FCC 20-164, Report and Order, adopted Nov. 18, 2020, published at
   86 FR 23281 (May 3, 2021) — upper 30 MHz (5895–5925 MHz) reserved for
   ITS; lower 45 MHz (5850–5895 MHz) opened to U-NII-4.
2. FCC 24-123, Second Report and Order, adopted Nov. 20, 2024, published at
   89 FR 100838 (Dec. 13, 2024), effective Feb. 11, 2025 — C-V2X-only
   operation in the upper 30 MHz; two-year DSRC sunset running from Federal
   Register publication.
3. 47 CFR 95.3205 (Unwanted emissions limits, C-V2X OBUs), as amended at
   89 FR 100855 (Dec. 13, 2024), corrected at 90 FR 5724 (Jan. 17, 2025).
4. 47 CFR 15.407 (U-NII-4 band-edge and out-of-band emission conditions for
   the lower 45 MHz).
5. ETSI EN 302 571 V1.2.1 (2013-09), Table 7 (in-band unwanted emissions,
   ITS 10 MHz channels); official ETSI deliver, archived with sha256
   provenance in the released artifact.
6. IEEE 802.11 / 3GPP C-V2X transmit-mask conventions referenced for
   the power-amplifier analysis.

## Summary

We report a receiver-side robustness consideration for the 5.9 GHz
transition that, to our knowledge, is not reflected in current coexistence
testing practice: a transmitter that is **fully compliant with its
allocation and emission mask** — verified against the actual EN 302 571
Table 7 template and the 47 CFR 95.3205 limits, not a stylized mask — can
induce misclassification in deep-learning-based coexistence sensing at
received powers tens of dB below the victim signal. We suggest that learned
coexistence receivers adopted during or after the transition be evaluated
against rule-compliant worst-case interference, not only against
conventional blocking, adjacent-channel, and jamming profiles. An open,
reproducible evaluation suite for exactly this test is released (GitHub
release `phase1-v1.2`; archival DOI pending); every number below traces to a
committed result file.

## The mechanism, stated for a regulatory audience

Adversarial-example research has shown that machine-learning classifiers can
be misclassified by small, purpose-crafted perturbations. The wireless
lineage of this threat is the primary-user-emulation attack (PUEA) studied
since 2008 (Chen, Park, and Reed, IEEE JSAC 26(1), Jan. 2008; Anand, Jin,
and Subbalakshmi, IEEE DySPAN 2008; Jin, Anand, and Subbalakshmi, IEEE
Trans. Commun. 60(8), Aug. 2012). PUEA work assumed a cognitive radio
deciding "is the primary present?" from energy features; the modern version
is a deep-learning spectrum sensor deciding "is this band occupied, and by
whom?" from wideband IQ. Prior adversarial-ML work in wireless either
perturbs data inside the receiver (physically unrealizable) or transmits
without regard for spectrum rules (illegal — already addressed by
enforcement). The case of regulatory interest is different: a
**rule-compliant** transmitter that adds a purpose-crafted, low-power
component to its own lawful transmission. Such a device:

* passes its emission constraints by construction — the perturbation is
  projected onto the transmitter's mask (we verified the attack survives
  the real EN 302 571 Table 7 shape and power-amplifier regrowth at
  appropriate backoff; the 47 CFR 95.3205 C-V2X OOB limits are extracted,
  archived, and the 95.3205-shaped projection has been RUN: compliant
  MEAP -37.9 dB, PoC 6.3 dB, post-PA worst excess 0.00 dBr — the attack
  survives all four enforcement domains we model);
* creates no enforcement signature — there is nothing to detect or fine
  under mask-shaped monitoring;
* yet shifts the output of a learned coexistence classifier.

## Measured effect (prototype scale; every figure carries a Wilson 95%
binomial confidence interval; TR 37.885-style urban/highway/rural channel
models; conditional attack success measured only on windows the sensor
initially classified correctly)

* Compliant attacker reaches a 20% conditional misclassification threshold
  at approximately **−37.1 dB** received power relative to the victim
  signal (unconstrained attacker: −44.2 dB). The price of compliance is
  7.1 dB untargeted (95% CI [4.4, 9.8]) and 16–18 dB for targeted
  "cloaking" of a safety transmitter.
* The price of compliance is **not an artifact of a self-defined mask**:
  re-running with the real EN 302 571 V1.2.1 Table 7 template moves the
  compliant MEAP by only +0.36 dB (PoC 7.50 vs 7.14) — the real template is
  slightly more restrictive for the attacker.
* At −10 dB (one-tenth of the victim's received power): **96.7%**
  conditional misclassification.
* Inside ITS power rules the attack is **fleet-relevant**: at the 33 dBm
  EIRP cap, a cloaked attacker can reach up to ~650 m (dense-blocker
  geometry) at the 20% false-IDLE point, and a synchronized worst-case
  blocker costs **9.5 percentage points of BSM delivery (PRR) at 100 m**
  (~95 additional lost basic safety messages per 1000); random-timing
  average case 0.2 pp. (Parametric link budget anchored to the measured
  attack curve; all parameters disclosed and swept.)
* Mask-matched adversarial training raises the compliant threshold by
  **+6.7 dB (canonical training seed) / +9.8 ± 2.8 dB across three
  independently trained defenses** at zero clean-accuracy cost (TRADES
  variant: +9.9 / +11.5 ± 1.8 dB at 0.75 pp clean cost), under a converged
  adaptive attack (PGD-50, 5 restarts). The 3.5–5.5 dB seed-to-seed spread
  in the defense margin is itself a finding: single-seed defense evaluation
  is not certification-grade.
* The attack is **not merely a weak-window phenomenon**: stratifying by
  clean-decision margin, the strongest-margin third of windows crosses the
  20% threshold 10 dB above the headline (−27.0 dB vs −37.1 dB), and at
  −20 dB nearly half of the strongest-margin windows flip.
* A max-softmax confidence gate — the simplest deployable adversarial-input
  detector — **fails in the regime that matters**: 16–36% true-positive
  rate at 1% false-positive rate for attack powers at or below −30 dB,
  near-blind at −15 dB. Simple detectors are not a defense; worst-case
  receiver testing is required.
* Real over-the-air 802.11 captures in the victim-signal role reproduce the
  compliant threshold within 1.8 dB; synthetic-only evaluation was found to
  overstate robustness by hiding decision-margin fragility on real signals.
* The result is architecture-dependent by 11.6–15.1 dB across two tested
  classifiers — robustness is a property of the specific deployed model,
  which is the reason per-model testing is needed.
* Feasibility: the sensing decision costs ~2.6 ms per window (median, CPU,
  front-end included) — roughly 37 decisions per 100 ms TR 37.885 latency
  budget — so computational latency is not the obstacle; robustness is.

## Suggested consideration

Learned coexistence-sensing receivers (including any ML-based sensing in
O-RAN RIC xApps, chip-level coexistence managers, or C-V2X receiver
enhancements) that are relied upon for transition-period coexistence
decisions could be evaluated, before deployment, against worst-case
**rule-compliant** interference in power-domain terms: the minimum
compliant-interferer power at which misclassification crosses a policy
threshold (our suite calls this MEAP), and the margin between that point
and the powers such interferers can lawfully deliver in practice (our
link-budget layer translates this into meters and dBm under disclosed
parameters). This is analogous in spirit to existing receiver
blocking/overload testing, but adversarial in the perturbation and
worst-case in the waveform.

One additional, verifiable property: our power-amplifier study (Rapp
nonlinearity, pre-registered protocol) shows the naive mask-compliant
attack waveform is an RF-statistical outlier — the highest-PAPR signal in
the band (11.1 dB vs 6.3–9.0 dB for C-V2X/802.11/normal classes) whose
post-amplifier spectrum violates mask-shoulder limits at ordinary operating
backoffs (it requires roughly twice the backoff of a compliant OFDM
transmitter to stay inside a −40 dBr mask). A receiver-robustness test at
the PA-output port would therefore catch unsophisticated compliant
attackers with conventional measurement gear — while our PA-aware
re-optimization result (stronger attack, fully compliant post-PA) shows
such RF-side screening is a complement to, not a substitute for,
worst-case receiver testing in power units.

## Scope and non-claims (stated plainly)

* Single-receiver, differentiable-model scope; no network-level or
  multi-cell claims. The harm-chain numbers are a disclosed-parameter
  link-budget translation, not a system simulation.
* Prototype scale: 300 evaluation windows per condition, three training
  seeds for the undefended headline (spread 1.1 dB), three defense seeds
  for the adaptive margins (spread 3.5–5.5 dB), converged adaptive attack
  (PGD-50, 5 restarts) for the defense numbers, Wilson 95% binomial
  confidence intervals on every conditional-ASR cell. Worst-case attacker
  knowledge (exact model and received realization — the disclosed upper
  bound; surrogate-model attackers measured 10.8–23.2 dB weaker).
* Synthetic channel and waveform models (TR 37.885-inspired;
  standard-parameterized PC5/802.11p), with real OTA 802.11 captures for
  the Wi-Fi class (74 evaluation windows, one capture location, 5.24 GHz
  U-NII-1 — the OFDM family adjacent to U-NII-4, labeled as such); no
  over-the-air validation of the attack transmission itself (conducted/
  cabled OTA is the next planned step; hardware budget disclosed in the
  artifact).
* RF impairments beyond a memoryless Rapp power-amplifier model (phase
  noise, CFO, AGC, quantization) are not modeled; PA spectral regrowth at
  the amplifier output port IS modeled and independently verified.
* Nothing here licenses non-compliant behavior; the attack *is* compliant
  by construction, which is precisely why it is a policy consideration
  rather than an enforcement matter.

## Artifact

Open-source evaluation suite (generators, channels, attack, metrics,
checkpoints, all results with embedded configs; independent audit scripts
re-derive every headline number from the stored data):

* Repository: github.com/Daveshvats/v2x_adversarial_sensing, branch
  `phase1-compliant-attacker`, release `phase1-v1.2` (2026-09-07).
* Archival DOI: pending (Zenodo integration; the filer should not cite a
  DOI until minted).
* One-command red-team: `python scripts/v2x_redteam.py evaluate --victim
  dual --scenario urban` then `python scripts/v2x_redteam.py report --json
  results/redteam_dual_cv2x_attacker_urban.json` (CPU-only).

## Precedent the comment builds on

* Primary-user-emulation attacks: R. Chen, J. T. J. Park, J. H. Reed,
  "Defense against primary user emulation attacks in cognitive radio
  networks," IEEE J. Sel. Areas Commun., vol. 26, no. 1, Jan. 2008;
  S. Anand, S. Jin, K. Subbalakshmi, IEEE DySPAN 2008; S. Jin, D. Anand,
  K. Subbalakshmi, IEEE Trans. Commun., vol. 60, no. 8, Aug. 2012 (and the
  2009–2012 series).
* Adversarial ML on wireless signal classification: O'Shea et al. (2018);
  Sadeghi and Larsson (IEEE Commun. Lett. 2019); Kim and Sagduyu (CISS
  2020); Kim et al. (IEEE TWC 2022); Girmay et al. (Vehicular
  Communications 2023 — the clean-task paper whose sensing model this
  suite breaks); RadioShock et al. (IEEE TDSC 2026).
* Receiver-robustness policy lineage: FCC's long-standing receiver
  interference-resilience workstream (TAC) and the 5.9 GHz transition
  orders cited above.


## Filing strategy (35-c-21; where this goes — [USER] to file, counsel to
review)

WT 23-287 — the docket whose conventions this draft follows — closed with
the 2024 Second Report & Order; a new comment there is moot. The finding
has four live routes, in order of fit:

1. **FCC Technological Advisory Council (TAC).** The receiver-robustness
   angle (a compliant transmitter can shift a learned sensor) belongs in
   a TAC working-session presentation, not a docket: TAC is advisory,
   meetings are public and transcribed, and the submission cost is a
   slide deck + speaker request. No ex parte issues arise (advisory
   bodies are outside 47 CFR 1.1206's decisional-personnel scope). Ask:
   receiver-robustness language in any coexistence-guidance product.
2. **5GAA technical working groups.** The C-V2X deployment constituency.
   Contribution via a member organization under NDA-free technical WG
   conventions; the artifact maps to security/safety work items. The
   harm-chain numbers (BSM PRR -9.4 pp at 100 m) are the hook.
3. **ETSI TC-ITS.** The measurement-domain finding (per-FFT-bin mask
   projections are NOT measurement-equivalent to clause-6.4.2 RBW/mean
   power; the difference is attacker-visible at +0.65 dB) is directly
   relevant to EN 302 571 maintenance and conformance-method discussion;
   route via a national member body.
4. **FCC OET laboratory liaison.** For the measurement-convention
   questions only (RBW/detector treatment of projected emissions), as a
   technical inquiry rather than a filing.

**Ex parte note (47 CFR 1.1206):** if the [USER] chooses a
permit-but-disclose route that involves commission decisional personnel,
oral ex parte presentations require notice in the docket (written notice
for off-the-record communications; the TAC route avoids the question
entirely). Counsel should confirm current applicability — the
transition-era dockets are closed and the default route above is
non-docketed.

**What each venue receives:** this draft (venue-adapted), the paper PDF,
the repository (Zenodo DOI once minted — [USER]), and the four-domain
compliance table (flat / per-bin ETSI / RBW mean-power / US 95.3205:
PoC 6.3-7.5 dB, all post-PA-verified at 0.00 dBr excess).

**Plainly not done here:** no filing of any kind has been made; no
regulatory feedback exists; the DOI is not minted. These are [USER]
actions tracked in RESEARCH_STATE.
