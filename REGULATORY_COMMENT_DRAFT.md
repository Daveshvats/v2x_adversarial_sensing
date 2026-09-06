# DRAFT technical comment — receiver-side adversarial robustness of
# coexistence sensing in the 5.9 GHz transition

**Status: DRAFT prepared 2026-09-05. Not submitted. For the authors to
review, adapt, and file through the appropriate channel** (e.g., a
technical ex parte / comment in relevant FCC dockets on 5.9 GHz transition
implementation, a 5GAA technical contribution, or an ETSI ISG SAI-style
input). All numbers are from the released artifact and trace to result
JSONs; citations point to the accompanying paper.

---

## Summary

We report a receiver-side robustness consideration for the 5.9 GHz
transition that, to our knowledge, is not reflected in current coexistence
testing practice: a transmitter that is **fully compliant with its U-NII-4
allocation and emission mask** can induce misclassification in
deep-learning-based coexistence sensing at received powers tens of dB below
the victim signal. We suggest that learned coexistence receivers adopted
during or after the transition be evaluated against rule-compliant
worst-case interference, not only against conventional blocking, adjacent
channel, and jamming profiles. An open, reproducible evaluation suite for
exactly this test is available (DOI on release; CPU-only; one command).

## The mechanism, stated for a regulatory audience

Adversarial-example research has shown that machine-learning classifiers can
be misclassified by small, purpose-crafted perturbations. In wireless, prior
work either perturbs data inside the receiver (physically unrealizable) or
transmits without regard for spectrum rules (illegal, i.e., already
addressed by enforcement). The case of regulatory interest is different:
a **rule-compliant** transmitter that adds a purpose-crafted, low-power
component to its own lawful transmission. Such a device:

* passes the modeled spectral-emission constraints (allocation plus an
  emission mask modeled on the applicable limits — the perturbation is
  projected onto the attacker's mask by construction);
* creates no enforcement signature — there is nothing to detect or fine
  under mask-shaped monitoring;
* yet shifts the output of a learned coexistence classifier.

## Measured effect (prototype scale; three training seeds; TR 37.885-style
urban/highway/rural channel models; conditional attack-success measured
only on windows the sensor initially classified correctly)

* Compliant attacker reaches a 20% conditional misclassification threshold
  at approximately **−37 to −39 dB** received power relative to the victim
  signal (unconstrained attacker: −44 dB; compliance costs the attacker
  only ~5–7 dB).
* At −10 dB (one-tenth of the victim's received power): **96.7%**
  conditional misclassification; the sensor can also be driven to report an
  active safety transmitter as empty channel ("cloaking") for ~16–18 dB
  additional attacker cost.
* Mask-matched adversarial training raises the compliant threshold by
  **+14 dB at zero clean-accuracy cost** (+18.6 dB for a TRADES-style
  variant at 0.75 pp cost), within the trained power range.
* Real over-the-air 802.11 captures in the victim-signal role reproduce the
  compliant threshold within 1.8 dB; synthetic-only evaluation was found to
  overstate robustness by hiding decision-margin fragility on real signals.
* The result is architecture-dependent by 12–15 dB across two tested
  classifiers — i.e., robustness is a property of the specific deployed
  model, which is the reason per-model testing is needed.

## Suggested consideration

Learned coexistence-sensing receivers (including any ML-based sensing in
O-RAN RIC xApps, chip-level coexistence managers, or C-V2X receiver
enhancements) that are relied upon for transition-period coexistence
decisions could be evaluated, before deployment, against worst-case
**rule-compliant** interference in power-domain terms: the minimum
compliant-interferer power at which misclassification crosses a policy
threshold (our suite calls this MEAP), and the margin between that point and
the powers such interferers can lawfully deliver in practice. This is
analogous in spirit to existing receiver blocking/overload testing, but
adversarial in the perturbation and worst-case in the waveform.

One additional, verifiable property: our power-amplifier study (Rapp
nonlinearity, pre-registered protocol) shows the naive mask-compliant
attack waveform is an RF-statistical outlier — the highest-PAPR signal
in the band (11.1 dB vs 6.3–9.0 dB for C-V2X/802.11/normal classes) whose
post-amplifier spectrum violates mask-shoulder limits at ordinary
operating backoffs (it requires roughly twice the backoff of a compliant
OFDM transmitter to stay inside a −40 dBr mask). A receiver-robustness
test at the PA-output port would therefore catch unsophisticated
compliant attackers with conventional measurement gear — while our
PA-aware re-optimization result (stronger attack, fully compliant
post-PA) shows such RF-side screening is a complement to, not a
substitute for, worst-case receiver testing in power units.

## Scope and non-claims (stated plainly)

* Single-receiver, differentiable-model scope; no network-level or
  multi-cell claims.
* Prototype scale: 300 evaluation windows per condition, PGD-10 (lower
  bound on attack strength), three training seeds (spread 1.1 dB), worst
  case attacker knowledge (exact model and received realization — the
  disclosed upper bound; surrogate-model attackers measured 11–23 dB
  weaker).
* Synthetic channel and waveform models (TR 37.885-inspired;
  standard-parameterized PC5/802.11p), with real OTA 802.11 captures for
  the Wi-Fi class; no over-the-air validation of the attack itself.
* RF impairments beyond a memoryless Rapp power-amplifier model (phase
  noise, CFO, AGC, quantization) are not modeled; PA spectral regrowth at
  the amplifier output port IS modeled and independently verified (see the
  "power-amplifier reality check" section of the accompanying paper).
* Nothing here licenses non-compliant behavior; the attack *is* compliant
  by construction, which is precisely why it is a policy consideration
  rather than an enforcement matter.

## Artifact

Open-source evaluation suite (generators, channels, attack, metrics,
checkpoints, all results with embedded configs):
github.com/Daveshvats/v2x_adversarial_sensing, branch
`phase1-compliant-attacker` (DOI minted at release).
