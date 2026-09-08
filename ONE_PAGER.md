# Can a rule-abiding transmitter break your AI radio sensor?

**A plain-language summary for engineers, regulators, and standards people.**
(One page. No math. The technical backing is the accompanying paper and the
open-source test suite.)

---

## The situation

After the FCC's 2020–2024 redesign of the 5.9 GHz band, unlicensed Wi-Fi-type
devices (U-NII-4) and vehicle safety radios (C-V2X / 802.11p) are neighbors:
Wi-Fi-like transmissions live just below the 5895 MHz boundary, vehicle
safety lives just above it. During the transition, radios must *sense* which
technology is transmitting and behave accordingly — and increasingly, that
sensing is done by a deep-learning classifier, because legacy
energy-detection cannot tell co-channel technologies apart.

Machine-learning classifiers are known to be attackable with tiny, carefully
crafted perturbations ("adversarial examples"). The research literature has
two problems, though. Most attacks are mathematical fictions — they modify
the data *inside the receiver*, which no radio transmitter can physically
do. And the few transmitted attacks ignore spectrum rules entirely: a
transmitter that squashes its power across the whole band is a jammer, and
jammers are already illegal, already detected, and not interesting.

## What we asked

**What if the attacker follows the rules?** A real U-NII-4 device must
transmit only inside its allocation, below the boundary, within its emission
mask. Can such a device — perfectly rule-compliant, passing every spectrum
monitor — still break a deep-learning coexistence sensor? And how much
attack power does compliance cost, compared to an unconstrained one?

## What we found

1. **Yes.** A rule-compliant transmitter breaks an undefended deep-learning
   sensor at a received power **tens of thousands of times below the signal
   it is attacking** (−37 dB in our units — nearly four orders of magnitude). At a
   more modest one-tenth of the victim's power, it fools the sensor **96.7%
   of the time**.

2. **The rules cost the attacker surprisingly little.** A rule-breaking
   (unconstrained) attacker would succeed at about 7 dB less power for
   generic mischief — and about 16–18 dB less for the nastier goal of
   making an active transmitter *look like empty air* ("cloaking").
   Compliance is a speed bump, not a wall.

3. **"Compliant" attacks are invisible to enforcement — almost.** By
   construction, the attack waveform is a legitimate transmission: right
   allocation, right mask, normal-looking power. There is nothing to fine.
   This is a *policy* gap, not an enforcement gap — the rules permit the
   attacker to exist. One practical caveat from our power-amplifier study:
   the naive attack waveform is unusually peaky (the highest-PAPR signal
   on the air, 11 dB vs 6–9 dB for everything else), so transmitting it
   through a normal radio's amplifier produces out-of-band regrowth that
   a spectrum analyzer *can* see — unless the attacker re-optimizes with
   the amplifier in the loop, which also fixes that tell (and makes the
   attack slightly stronger). Detection helps against lazy attackers; it
   is not a defense against the worst case.

4. **The standard defense works, with an honest boundary.** Training the
   sensor against these attacks (adversarial training with the same mask
   constraints) shifts the break-point by +14 dB at zero cost to normal
   accuracy, and a variant buys +18.6 dB for a 0.75-point accuracy cost —
   but those numbers are from the same 10-step attack used in evaluation,
   and a patient adversary running a converged 50-step attack with random
   restarts claws most of it back: the honest margins are +7.6 dB and
   +9.8 dB (verified across three independent attack runs, 95% confidence
   ±1.8 dB or less). But the protection only covers the power range it
   was trained for.

5. **This is not about our specific neural network.** We attacked a second,
   architecturally different network: its break-point shifted by 11.6–15.1 dB,
   showing the test discriminates between models — model choice matters
   more than the emission mask. Attacks transferred *between* models only
   at 11–23 dB extra cost: knowing the exact victim model is a big part of
   the threat.

6. **Real signals confirm it.** Replacing our simulated Wi-Fi class with
   real over-the-air Wi-Fi captures (real channels, real hardware), the
   compliant-attack threshold lands within 1.8 dB of the simulated number.
   And the real data exposed something the simulation hid: near the
   boundary of its competence, the model flips under perturbations
   essentially of *zero* power — synthetic-only testing would have
   overstated robustness.

## What you can do about it (today)

We release the test suite (open source, CPU, one command):

```
python scripts/v2x_redteam.py evaluate --victim <your-exported-model>
```

It reports, in link-budget units, the minimum power at which a
rule-compliant transmitter breaks your sensor, verifies the attack truly
respected the mask during the test, and emits a certification-style report.
Suggested uses: a pre-deployment robustness gate for ML-based sensing
components (O-RAN RIC xApps, coexistence chips), a regression gate in CI,
and worst-case input to threshold/guard-budget design during the 5.9 GHz
transition.

## The one-line takeaway

**"Compliant" and "harmless" are not the same thing: a transmitter your
rules are happy with can still blind a deep-learning coexistence sensor —
and the only honest defense is to test for it, in power units, before
deployment.**
