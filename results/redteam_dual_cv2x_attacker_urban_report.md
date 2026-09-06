# V2X Red-Team Report — victim `dual`

- **Eval set:** synthetic, urban, 100/active class, seed 7
- **Attacker allocation:** cv2x_attacker (mask-compliant (OOB nulled, in-band PSD capped))
- **Clean accuracy (active tx):** 100.00%
- **Worst-case disclosure:** attacker knows the victim model and the exact received realization incl. noise — ASRs are upper bounds
- **Config:** PGD-10, attack seed 7, PSD margin 2.0, alpha 0.25
- **PSR reference:** clean received-signal window energy (sum |r|^2, pre-noise)

## Results

MEAP (PSR at which conditional-ASR crosses 20.0%) and Price of Compliance (MEAP_compliant − MEAP_genie):

| Setting | MEAP (dB) | censored |
|---|---|---|
| genie | -44.22475707646814 | None |
| mask_compliant | -37.08333333333358 | None |
| **Price of Compliance** | **7.14142374313456 dB** | None |

Conditional-ASR sweep (success over windows the victim classified correctly before the attack):

| PSR (dB) | genie ASR % | compliant ASR % | robust acc % |
|---|---|---|---|
| -45 | 16.33 | 3.0 | 97.0 |
| -40 | 40.0 | 13.0 | 87.0 |
| -35 | 71.0 | 25.0 | 75.0 |
| -30 | 88.0 | 39.33 | 60.67 |
| -25 | 98.67 | 58.33 | 41.67 |
| -20 | 100.0 | 78.0 | 22.0 |
| -15 | 100.0 | 88.33 | 11.67 |
| -10 | 100.0 | 96.67 | 3.33 |
| -5 | 100.0 | 99.67 | 0.33 |
| +0 | 100.0 | 100.0 | 0.0 |
| +5 | 100.0 | 100.0 | 0.0 |
| +10 | 100.0 | 100.0 | 0.0 |

## Projection physics (verified during the run)

- post-projection budget ratio (median): `0.16995442` (must be ≤ 1.0000001)
- perturbation out-of-band fraction: `1.033e-14` (mask: ~0)

## Reading

The mask-compliant attacker — a transmitter that respects its regulatory allocation and emission mask — reaches the 20.0% conditional-ASR threshold at **-37.08333333333358 dB** PSR (received attack power relative to the victim signal). The unconstrained genie needs -44.22475707646814 dB; the rules cost the attacker **7.14142374313456 dB**. Lower (more negative) MEAP = weaker victim. Every number above is an upper bound (worst-case attacker knowledge; see disclosure).