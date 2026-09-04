# Archived pre-2026-09-05 attack results

These JSONs were produced before the Wave-1 audit fix of the PSR power-axis
unit error (mean-power budget vs window-energy constraint, 10*log10(2048) =
33.11 dB shift) and before the genie PSD-cap exemption fix. Their ABSOLUTE
PSR/MEAP labels are wrong; their PoC difference metric is superseded by the
corrected runs (the flat PSD cap was silently applied to the genie, inflating
PoC from ~7 to ~13 dB). Kept for provenance only — do not cite.
Corrected results: attack_results_{scenario}_{mode}.json + merged.
