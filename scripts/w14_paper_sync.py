#!/usr/bin/env python3
"""w14_paper_sync.py — Wave-14 doc sync for paper/main.tex.

Adds: US 95.3205 fourth-domain paragraph (C42), attack-seed replication
paragraph (C44), paired-bootstrap CI numbers (C45), cooperative-fusion
baseline (C43), honest real-WiFi PoC floors (item 14), PGD-10 3-seed
defense readings (item 13), and updates the protocol/limitations/abstract/
conclusion to match. Every replacement asserts a unique anchor.
"""
import sys

PATH = "paper/main.tex"
src = open(PATH).read()
n_applied = 0


def rep(old, new, tag):
    global src, n_applied
    assert src.count(old) == 1, f"[{tag}] anchor not unique: {src.count(old)}"
    src = src.replace(old, new)
    n_applied += 1
    print(f"  [{tag}] applied")


# ---- E1: protocol CI sentence (Wilson + paired bootstrap pointer) ----
rep(r"""attack, and every MEAP carries an explicit censoring flag when the grid
does not resolve it. Every conditional-ASR cell is stored with its
success count and eligibility, and headline MEAP/PoC numbers carry
Wilson binomial 95\% confidence intervals (sampling noise at fixed
model/protocol; model-seed spread is reported separately).""",
r"""attack, and every MEAP carries an explicit censoring flag when the grid
does not resolve it. Every conditional-ASR cell is stored with its
success count and eligibility, and headline MEAP/PoC numbers carry
Wilson binomial 95\% confidence intervals (sampling noise at fixed
model/protocol; model-seed spread is reported separately); because the
genie and mask arms attack the \emph{same} windows, headline PoCs
additionally carry paired-bootstrap 95\% CIs
(Section~\ref{sec:adaptive}) that propagate the shared sampling noise
instead of worst-casing each arm separately.""", "E1")

# ---- E2: protocol honesty-box confidence row ----
rep(r"""Confidence & Wilson 95\% binomial CIs on every ASR cell (results/w12\_confidence\_intervals.json) \\""",
r"""Confidence & Wilson 95\% binomial CIs on every ASR cell; paired-bootstrap MEAP/PoC CIs (results/w12\_confidence\_intervals.json, results/w14b\_bootstrap\_cis.json) \\""",
    "E2")

# ---- E4: tab:adaptive caption attack-seed note ----
rep(r"""\caption{Adaptive-attack evaluation (urban, untargeted, attack seed 7,
300 active windows; PGD-50, 5 restarts, best-of-R per sample).""",
r"""\caption{Adaptive-attack evaluation (urban, untargeted, attack seed 7
(replicated on attack seeds 11/22 with grid identity: margin spreads
$\pm$0.1--0.7\,dB, Section~\ref{sec:adaptive}),
300 active windows; PGD-50, 5 restarts, best-of-R per sample).""", "E4")

# ---- E3: attack-seed replication + bootstrap paragraph ----
rep(r"""replication adds results/adaptive\_\{at,trades\}\_\{d43,d53\}\_s7\_r5.json
plus the re-derivation and independent audit in
results/w13\_defense\_seed\_replication.json (checked by
scripts/w13\_check.py, F1--F8).""",
r"""replication adds results/adaptive\_\{at,trades\}\_\{d43,d53\}\_s7\_r5.json
plus the re-derivation and independent audit in
results/w13\_defense\_seed\_replication.json (checked by
scripts/w13\_check.py, F1--F8).

\paragraph{Attack-seed replication and paired-bootstrap CIs}
\label{par:attackseed}
The orthogonal axis---\emph{attack} seeds, fixed at 7 so far---is closed
with the same grid-identity discipline: two further attack seeds
$\{11,22\}$ on the canonical seed-42 trio, per-arm PSR grids identical
to the seed-7 files, 300 windows, and per-window outcome capture (six
new grids, 68 replication cells, all MEAPs uncensored; plus a 34-cell
per-window re-capture of the seed-7 canonical cells that reproduces
every stored cond-ASR exactly---the attack is deterministic given the
seed, so the capture doubles as a continuity check; independent audit
scripts/w14\_check.py, 86 checks). The margins are far more stable
across attack seeds than across defense seeds: AT $+7.6\pm0.7$\,dB and
TRADES $+9.8\pm0.1$\,dB (per-seed ranges 1.3 and 0.24\,dB) versus the
defense-seed spreads of $\pm2.8$ and $\pm1.8$\,dB above---a
$4$--$23\times$ smaller spread, i.e.\ the seed lottery that matters for
certification is the \emph{defense} training seed, not the attack seed.
Adaptive PoCs over three attack seeds: undefended $6.2$ (range 0.67),
AT $7.0$ (0.16), TRADES $9.9$ (1.39)\,dB. Because the capture stores
the genie/mask window pairing, these PoCs carry paired-bootstrap 95\%
CIs ($B{=}2000$ percentile, windows resampled with replacement, same
indices for both arms): the canonical PGD-10 PoC $7.14$\,dB moves from
the conservative bound-propagation interval $[4.4,10.1]$\,dB to
$[5.7,9.2]$\,dB (39\% narrower, 0/2000 resamples dropped), the
three-seed model-mean is $7.0$\,dB with $t$-interval $[6.6,7.4]$, and
the adaptive protocol gives undefended $6.2$ $[5.3,7.1]$, AT $7.0$
$[6.8,7.2]$, and TRADES $9.9$ $[8.1,11.6]$\,dB ($t$-intervals over
three attack seeds). The two model-seed grid floors that the
12-point PSR grid left censored (seeds 123/456 genie arms) are resolved
by extending those grids to $-55$\,dB, which moves their point PoCs to
$7.96$ and $7.52$\,dB (was: floor-censored lower bounds; the headline
interval above is recomputed with the merged cells). Artifacts:
results/w14\_attack\_seed\_margins.json,
results/w14b\_bootstrap\_cis.json, the six
results/adaptive\_\{dual,at,trades\}\_s\{11,22\}\_r5.json grids
(win\_flags per cell), and
results/w14b\_perwindow\_adaptive\_\{dual,at,trades\}\_s7.json.""",
    "E3")

# ---- E5: real-WiFi R2 honest floors ----
rep(r"""\textbf{(R2) The canonical
compliant-attacker conclusion replicates on real signals.} With real WiFi
as the victim signal, the compliant MEAP is $-35.9$\,dB (frozen) and
$-37.7$\,dB after the leakage-free fine-tune (73\% real-WiFi
accuracy)---bracketing the synthetic-task value of $-37.1$\,dB and its
three-seed spread. The price of compliance is 7.3\,dB (lower bound),
matching the synthetic 7.1\,dB.""",
r"""\textbf{(R2) The canonical compliant-attacker conclusion replicates on
real signals---at a \emph{higher} price.} With real WiFi as the victim
signal, the compliant MEAP is $-35.9$\,dB (frozen) and $-37.7$\,dB after
the leakage-free fine-tune (73\% real-WiFi accuracy)---bracketing the
synthetic-task value of $-37.1$\,dB and its three-seed spread. Extending
the genie arm to $-55$\,dB resolves the censoring that earlier bounded
the real-signal price from below: the true genie floors sit at
$-47.8/-47.9$\,dB (frozen/finetuned), $3.6$\,dB deeper than the
synthetic $-44.2$\,dB---consistent with the near-zero-margin pathology
of (R3)---so the real-signal price of compliance is $11.9$\,dB (frozen)
and $10.2$\,dB (finetuned), \emph{larger} than the synthetic $7.1$\,dB
(Wilson CIs $[4.2,16.6]$ and $[2.0,16.3]$: with 74/22 real windows the
honest statement is the point spread, not a tight match). Synthetic
evaluation understates what compliance buys against real captures.""",
    "E5")

# ---- E6: fusion baseline paragraph after the detector repro line ----
rep(r"""Repro: results/w13\_margin\_stratified.json (per-PSR per-tertile ASR +
both detector directions).""",
r"""Repro: results/w13\_margin\_stratified.json (per-PSR per-tertile ASR +
both detector directions).

\paragraph{Cooperative-sensing fusion baseline}
\label{par:fusion}
The detector above is per-receiver; deployments can also vote. We
evaluate $K$-receiver majority fusion (independent receiver chains, the
\emph{same} compliant delta reaching all $K$ through their own attacker
channels; attacker CSI remains single-receiver---a fusion-adaptive
attacker is disclosed future work). The control is exact: at $K{=}1$ the
fusion run reproduces the canonical MEAPs to the digit
($-37.1/-44.2$\,dB). At $K{=}3$, majority voting raises the compliant
MEAP from $-37.1$ to $-11.8$\,dB ($+25.3$\,dB of detection gain;
$K{=}5$: $-10.5$\,dB), while the price of compliance stays at
$6.3$--$7.3$\,dB across $K$. Spatially distributed sensing therefore
defeats the attacker's \emph{absolute} power advantage---roughly 25\,dB
must be bought to flip three voters---but leaves the \emph{relative}
mask penalty untouched, so no compliance conclusion in this paper
changes. For regulators the reading is double-edged: receiver fusion is
the cheapest real mitigation found so far, but it is demonstrated only
against a single-receiver-CSI attacker; the honest threat model for a
fused roadside unit remains the fusion-adaptive one. Artifacts:
results/w14b\_fusion\_baseline.json (per-$K$ per-PSR fused ASR,
per-receiver effective PSR, disclosed single-CSI limitation).""",
    "E6")

# ---- E7a: three-domain sentence makes room for the fourth ----
rep(r"""The price-of-compliance headline is
thus robust across all three enforcement domains (flat idealization,
per-bin real template, standard's RBW/mean-power domain) within
$\pm$0.5\,dB, which is the strongest statement about mask modeling this
simulation-only study can make.""",
r"""The price-of-compliance headline is
thus robust across the three ETSI-side enforcement domains (flat
idealization, per-bin real template, standard's RBW/mean-power domain)
within $\pm$0.5\,dB.""", "E7a")

# ---- E7b: US 95.3205 fourth-domain paragraph ----
rep(r"""Artifact: \texttt{results/conformance\_results.json} and
\texttt{src/conformance.py}.

\subsection{From sensing error to safety harm: BSM delivery}""",
r"""Artifact: \texttt{results/conformance\_results.json} and
\texttt{src/conformance.py}.

\paragraph{A fourth enforcement domain: the US C-V2X unwanted-emissions
limits}
\label{par:usdomain}
The European template is not the only live regime: 47 CFR 95.3205
imposes region-flat OBU unwanted-emissions limits ($-16$\,dBm/100\,kHz
within 1\,MHz of the band edge, $-13$\,dBm/MHz to 5\,MHz,
$-16$\,dBm/MHz to 30\,MHz; archived eCFR text, 89~FR~100855 with the
90~FR~5724 correction). Projecting into the same mean-power measurement
domain (100-kHz RBW for centres within 1\,MHz of the channel edge,
1-MHz RBW elsewhere, overlapping centres), the compliant MEAP is
$-37.9$\,dB and the PoC $6.3$\,dB: of the four enforcement domains now
evaluated (flat idealization, per-bin ETSI, ETSI RBW/mean-power, US
95.3205-shaped) this is the most permissive for the attacker ($0.8$\,dB
below flat), because the region-flat $-13$\,dBm/MHz middle region
permits in-band spectral concentration that the knot-shaped ETSI
template forbids---the mirror image of the edge-shaping effect above.
Post-amplifier compliance is again verified by measurement (worst
excess $0.00$\,dBr on every window of every cell). The
price-of-compliance headline is robust across all four enforcement
domains spanning two administrations, within $\pm$0.9\,dB of the flat
idealization (PoC 6.3--7.5\,dB)---which is the strongest mask-modeling
statement this simulation-only study can make. Artifact:
\texttt{results/conformance\_us\_results.json}.

\subsection{From sensing error to safety harm: BSM delivery}""",
    "E7b")

# ---- E8: limitations rewrite ----
rep(r"""(5)~Prototype scale: undefended headline now carries three training seeds
(mask MEAP spread 1.1\,dB), two attack seeds, 300 windows, PGD-10 (PGD-50
and seed stability checked on key conditions: shifts $\le$1.7\,dB); the
adaptive margins are replicated across three defense training seeds
(spread 3.5--5.5\,dB, larger than the undefended training-seed spread),
while the PGD-10 defense readings and the TRADES-vs-AT comparison at
PGD-10 remain seed-42 single-seed; the adaptive evaluation fixes the attack
seed at 7 (5 restarts), so attack-seed variance is covered only on the
undefended model. Confidence
intervals are Wilson binomial on sampling noise only; model-seed variance
(1.1\,dB undefended, 3.5--5.5\,dB defended) is a separate, additive source.""",
r"""(5)~Prototype scale: the undefended headline carries three training seeds
(mask MEAP spread 1.1\,dB) and the adaptive protocol three attack seeds
$\{7,11,22\}$ (grid-identical, per-window captured; the canonical PGD-10
pair covers attack seeds $\{7,11\}$); the adaptive margins are replicated
across three defense training seeds (spread 3.5--5.5\,dB, larger than the
undefended training-seed spread) \emph{and} three attack seeds (spread
0.1--1.3\,dB, $4$--$23\times$ smaller than the defense-seed spread). The
PGD-10 defense readings now span three defense seeds (AT mask MEAP
$-23.1/-19.7/-20.9$\,dB, TRADES $-19.1/-19.2/-18.5$\,dB; the
naive-vs-adaptive gap of $6.9$--$10.4$\,dB holds on every seed).
Confidence intervals are Wilson binomial on per-cell sampling noise plus
paired-bootstrap MEAP/PoC CIs; model-seed variance (1.1\,dB undefended,
3.5--5.5\,dB defended) remains a separate, additive source. Two floors
remain: five defense seeds and a fusion-adaptive attacker are not run.""",
    "E8")

# ---- E8b: limitation (8) real-capture PoC update ----
rep(r"""(8)~The real-capture study uses 96 gated bursts from three capture
locations (74 eval / 22 train windows);""",
r"""(8)~The real-capture study uses 96 gated bursts from three capture
locations (74 eval / 22 train windows; genie floors resolved to
$-55$\,dB, real-signal PoC $10.2$--$11.9$\,dB with wide Wilson CIs);""",
    "E8b")

# ---- E9a: abstract conformance phrase ----
rep(r"""enforced in the standard's 1-MHz-RBW mean-power measurement domain): an \emph{emission-mask price of compliance of
5--7\,dB} (7.5\,dB under the real template, 6.9\,dB in the measurement
domain; untargeted) and 16--18\,dB for targeted ``cloaking'' of an active
transmitter as noise.""",
r"""enforced in the standard's 1-MHz-RBW mean-power measurement domain, and
in the US 47~CFR~95.3205-shaped limits): an \emph{emission-mask price of
compliance of 5--7\,dB} (6.3--7.5\,dB across all four enforcement
domains; untargeted) and 16--18\,dB for targeted ``cloaking'' of an
active transmitter as noise.""", "E9a")

# ---- E9b: abstract seed/CI clause ----
rep(r"""robustness on the canonical defense seed (replicated across three defense
training seeds: $+9.8\pm2.8$\,dB for AT, $+11.5\pm1.8$\,dB for TRADES---the
$3.5$--$5.5$\,dB seed spread is itself a certification finding), and the""",
r"""robustness on the canonical defense seed (replicated across three defense
training seeds: $+9.8\pm2.8$\,dB for AT, $+11.5\pm1.8$\,dB for TRADES---the
$3.5$--$5.5$\,dB seed spread is itself a certification finding---and
stable across three attack seeds, $\pm0.1$--$0.7$\,dB; every headline
PoC carries a paired-bootstrap 95\% CI), and the""", "E9b")

# ---- E9c: abstract stress-test sentence gains fusion ----
rep(r"""These conclusions are stress-tested beyond the prototype protocol: three
training seeds move the compliant threshold by at most 1.1\,dB; replacing""",
r"""These conclusions are stress-tested beyond the prototype protocol: three
training seeds move the compliant threshold by at most 1.1\,dB;
three-receiver majority fusion buys $+25$\,dB of detection headroom while
leaving the price of compliance unchanged ($6.3$--$7.3$\,dB across $K$);
replacing""", "E9c")

# ---- E10: conclusion four-domain phrase ----
rep(r"""attackers, not safety against the worst case. The mask idealization
survives contact with the real ETSI EN 302 571 template (+0.4\,dB), and
the harm chain closes the gap from sensing error to fleet consequence:""",
r"""attackers, not safety against the worst case. The mask idealization
survives contact with four enforcement domains spanning two
administrations (real ETSI EN 302 571 template $+0.4$\,dB; its
RBW/mean-power measurement domain and the US 47~CFR~95.3205-shaped
limits $-0.3$ to $-0.8$\,dB), receiver fusion raises the attack's power
floor by 25\,dB without changing the price of compliance, and
the harm chain closes the gap from sensing error to fleet consequence:""",
    "E10")

open(PATH, "w").write(src)
print(f"\n{n_applied}/14 edits applied to {PATH}")
