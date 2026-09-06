# Real over-the-air WiFi captures — attribution and license

These are unmodified raw capture files from:

> J. Fontaine, E. Fonseca, A. Shahid, M. Kist, L. A. DaSilva, I. Moerman,
> E. De Poorter, "Towards low-complexity wireless technology classification
> across multiple environments," *Ad Hoc Networks*, vol. 91, art. 101881,
> 2019. doi:10.1016/j.adhoc.2019.101881
>
> Dataset: "Technology-Recognition dataset of real-life LTE, Wi-Fi and
> DVB-T" — https://cloud.ilabt.imec.be/index.php/s/qrJCWgzQaGPfHPr
> (imec / Ghent University, captures in Gent, Belgium)

**License: CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/)
— redistributed here unmodified for non-commercial research use, with
attribution. Derivative works (e.g., the gated/normalized segments and
results derived in this repository) inherit the ShareALike terms.

## Files

`wf10Msps_<gain>_<location>_f5240MHz_r<N>.bin` — IEEE 802.11 WiFi captured
over-the-air at 5240 MHz (U-NII-2 band; same OFDM PHY family as U-NII-4)
with a USRP at 10 Msps, float32 interleaved I/Q, 8.8 MB (1.1 M complex
samples, 0.11 s) each. Locations: UZ (university hospital), Rabot, Reep,
Gentbrugge — distinct real propagation environments.

Only 8 of the 9 files yield verified OFDM bursts under our spectral-flatness
gate; `wf10Msps_g30_*` files are retained for transparency (their content is
below the gate: noise-floor level at gain 30) and are excluded by the loader.
`wf10Msps_g76_gentbrugge_*` passes the power gate but fails the flatness
gate (non-OFDM interference) and is likewise excluded.

## Processing applied downstream (see src/real_wifi.py)

DC removal (USRP LO leakage) → ×2 polyphase resample to 20 Msps → frequency
placement at [−10, 0] MHz relative to the 5895 MHz boundary (the in-window
half of a U-NII-4 channel) → burst gating (spectral flatness ≤ 0.45 AND
power ≥ 3× the flat-window floor) → unit-power normalization per window.
Physics gates (occupancy 99.7%, PAPR 9.95 vs synthetic 8.71 dB) are checked
by `scripts/w6_check.py`.
