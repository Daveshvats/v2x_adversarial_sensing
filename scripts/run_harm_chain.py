#!/usr/bin/env python3
"""run_harm_chain.py — Wave 12 part 2: the quantified harm chain
(council P0, reviewer 21-c: "harm chain (false-IDLE -> unlicensed TX -> BSM
PRR loss) absent; link-budget analysis 6-10 h").

The paper's sensing-level results are RELATIVE (PSR = attacker window energy
/ clean received window energy, channel normalized to unit power). This
script adds the ABSOLUTE link-budget layer that turns those numbers into
deployment-scale statements, with every parameter disclosed and swept:

  Stage 1  ATTACKER FEASIBILITY (absolute EIRP)
           Given the MEASURED targeted-noise mask-ASR curve (urban,
           attack_results_merged.json), the PSR needed for a chosen
           false-IDLE rate, a victim<->cloaked-transmitter distance d_B and
           an attacker<->victim distance d_A:
             P_B_rx  = P_B - PL(d_B)                (victim hears TX_B)
             P_att_rx = P_B_rx + PSR                (attack budget at RX)
             P_att    = P_att_rx + PL(d_A)          (attacker EIRP)
           Feasible iff P_att <= EIRP_CAP (33 dBm ITS class, EN 302 571 /
           47 CFR 95; 36 dBm U-NII-4 also shown for reference).

  Stage 2  HARM AT A THIRD RECEIVER (BSM PRR loss)
           The fooled victim (false idle) transmits its own BSM at full
           power, colliding with the legitimate packet it failed to sense.
           Legit link TX_C -> RX_D at distance D; colliding victim at
           uniform 1-D road position; lognormal shadowing both links.
             baseline : PRR(D) = P(S >= N + gamma_dec)
             attacked : PRR(D) = (1-p_int)*PRR_base
                              + p_int * P(survive collision)
             p_int    = ASR(PSR) * p_overlap   (attacker-synchronized
             worst case: the attacker keys its waveform to the transmission
             it wants cloaked - trivially achievable with energy detection;
             the random-timing average case is also reported)
             survive  : [no-capture]  SINR = S - 10log10(I+N) >= gamma_dec
                        [capture]     S - I >= gamma_cap  (802.11p-style)
  Stage 3  FLEET METRICS
           dPRR at the safety-relevant distances (100/150/200 m) and extra
           lost BSMs per 1000 transmitted.

All RF parameters are textbook/standard values, stated in the JSON and
swept in the sensitivity block: FSPL anchor 47.85 dB @ 1 m (5.9 GHz),
path-loss exponents {urban 3.0, highway 2.0, rural 2.5}, victim/cloaked TX
23 dBm (OBU class), EIRP cap 33 dBm, noise -95 dBm (10 MHz, NF 9 dB),
gamma_dec 6 dB (QPSK 1/2, PER 10%), gamma_cap 8 dB, shadowing sigma 4 dB
urban, BSM 0.5 ms airtime, overlap probability 0.6.

Outputs: results/harm_chain.json + paper/figs/harm_chain.png (2-panel).
Deterministic (fixed seeds); pure numpy + matplotlib.
"""
import os, sys, json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np

OUT = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "paper", "figs")

# ----------------------------------------------------------------------------
# parameters (all disclosed in the output JSON)
# ----------------------------------------------------------------------------
FSPL0 = 20 * np.log10(4 * np.pi / (3e8 / 5.9e9))    # 47.85 dB @ 1 m
P_TX_VICTIM = 23.0        # dBm fooled victim OBU (also TX_C, TX_B class)
P_TX_CLOAKED = 23.0       # dBm the cloaked transmitter (TX_B / TX_C)
EIRP_CAP_ITS = 33.0       # dBm EN 302 571 / 47 CFR 95 ITS class cap
EIRP_CAP_UNII4 = 36.0     # dBm U-NII-4 cap (reference)
NOISE_DBM = -174 + 10 * np.log10(10e6) + 9.0    # 10 MHz, NF 9 dB -> -95 dBm
GAMMA_DEC = 6.0           # dB QPSK 1/2 10 MHz, PER 10% (C-V2X BSM baseline)
GAMMA_CAP = 8.0           # dB capture margin
SHADOW_URBAN = 4.0        # dB lognormal sigma (urban)
P_OVERLAP = 0.6           # sensing-latency overlap fraction (swept 0.4-0.8)

SCENARIO_N = {"urban": 3.0, "highway": 2.0, "rural": 2.5}


def pl_db(d_m, n):
    return FSPL0 + 10.0 * n * np.log10(np.maximum(d_m, 1.0))


def load_targeted_curve(scenario="urban"):
    """The MEASURED targeted-noise mask-attacker cond-ASR curve (percent)
    that the harm chain consumes — direct provenance from the paper's runs."""
    d = json.load(open(os.path.join(OUT, "attack_results_merged.json")))
    runs = d["per_scenario"][scenario]["runs"]["targeted_noise/cv2x_mask"]
    psr, asr = [], []
    for k in sorted(runs, key=lambda s: float(s[4:-2])):
        psr.append(float(k[4:-2]))
        asr.append(runs[k]["cond_asr"])
    meap = d["per_scenario"][scenario]["summary"]["targeted_noise"][
        "meap_cv2x_mask_db"]
    return np.array(psr), np.array(asr), meap


def asr_at(psr_grid, asr_grid, psr):
    """Piecewise-linear interpolation of the measured ASR curve (clamped)."""
    return float(np.interp(psr, psr_grid, asr_grid,
                           left=asr_grid[0], right=asr_grid[-1]))


def stage1_feasibility(psr_grid, asr_grid, meap_db, n, scenario):
    """Required attacker EIRP (dBm) vs (d_A, d_B) at the MEAP PSR and at a
    high-success PSR; max feasible d_A for each d_B (EIRP cap)."""
    psr_points = {
        "meap_20pct": meap_db,
        "asr38_psr": -10.0,
        "asr90_psr": 0.0,
    }
    d_B = np.array([20.0, 50.0, 100.0, 200.0])
    d_A = np.arange(25.0, 1000.0, 25.0)
    out = {"d_B_m": d_B.tolist(), "psr_points": {}, "max_feasible_dA": {}}
    for tag, psr in psr_points.items():
        asr = asr_at(psr_grid, asr_grid, psr)
        eirp = {}
        for db in d_B:
            p_b_rx = P_TX_CLOAKED - pl_db(db, n)
            p_att_rx = p_b_rx + psr
            eirp[f"dB={db:.0f}"] = {
                "psr_db": round(psr, 2),
                "false_idle_pct": round(asr, 2),
                "required_eirp_dbm": [round(p_att_rx + pl_db(da, n), 1)
                                      for da in d_A],
            }
        out["psr_points"][tag] = eirp
        # max feasible d_A per d_B at this PSR
        mf = {}
        for db in d_B:
            p_b_rx = P_TX_CLOAKED - pl_db(db, n)
            need = p_b_rx + psr
            req = need + pl_db(d_A, n)
            ok = np.where(req <= EIRP_CAP_ITS)[0]
            mf[f"dB={db:.0f}"] = (round(float(d_A[ok[-1]]), 0)
                                  if len(ok) else 0.0)
        out["max_feasible_dA"][tag] = mf
    return out


def prr_curves(psr_grid, asr_grid, psr_db, scenario, n, sigma,
               p_overlap=P_OVERLAP, d_grid=None, n_mc=4000, seed=123,
               keep_round=True):
    """Monte-Carlo PRR(D) for the baseline and the attacked condition.
    Returns dict with curves for both collision models + the random-timing
    average case."""
    if d_grid is None:
        d_grid = np.unique(np.concatenate([
            np.arange(20.0, 520.0, 20.0), np.array([100.0, 150.0, 200.0])]))
    rng = np.random.default_rng(seed)
    asr = asr_at(psr_grid, asr_grid, psr_db) / 100.0

    base, att_nocap, att_cap, att_avg = [], [], [], []
    for D in d_grid:
        # --- baseline: noise-limited, lognormal shadowing ---
        x_s = rng.normal(0, sigma, n_mc)
        s = P_TX_VICTIM - pl_db(D, n) + x_s
        base.append(float(np.mean(s >= NOISE_DBM + GAMMA_DEC)))

        # --- attacked: colliding victim at uniform 1-D road position ---
        u = rng.uniform(-0.5, 1.5, n_mc)          # relative position vs D
        d_v = np.abs(u) * D
        d_v = np.maximum(d_v, 5.0)                # not on top of RX_D
        x_i = rng.normal(0, sigma, n_mc)
        i = P_TX_VICTIM - pl_db(d_v, n) + x_i

        # no-capture model: SINR threshold
        sinr = s - 10 * np.log10(10 ** (i / 10.0) + 10 ** (NOISE_DBM / 10.0))
        surv_nc = float(np.mean(sinr >= GAMMA_DEC))
        att_nocap.append((1 - asr * p_overlap) * base[-1]
                         + asr * p_overlap * surv_nc)

        # capture model: legit survives if S - I >= gamma_cap
        surv_c = float(np.mean((s - i >= GAMMA_CAP) &
                               (s >= NOISE_DBM + GAMMA_DEC)))
        att_cap.append((1 - asr * p_overlap) * base[-1]
                       + asr * p_overlap * surv_c)

        # random-timing average case (attacker not synchronized):
        # sensing window falls inside a 0.5 ms packet at 10 Hz -> ~1% duty
        p_rand = 0.01
        p_eff = asr * p_rand
        att_avg.append((1 - p_eff) * base[-1] + p_eff * surv_c)

    out = {
        "d_m": d_grid.tolist(),
        "psr_db": round(psr_db, 2),
        "false_idle_rate_pct": round(100 * asr, 2),
        "prr_baseline": [round(v, 4) for v in base],
        "prr_attacked_nocapture": [round(v, 4) for v in att_nocap],
        "prr_attacked_capture": [round(v, 4) for v in att_cap],
        "prr_attacked_random_timing": [round(v, 4) for v in att_avg],
    }
    if not keep_round:
        out["prr_baseline"] = [float(v) for v in base]
        out["prr_attacked_nocapture"] = [float(v) for v in att_nocap]
        out["prr_attacked_capture"] = [float(v) for v in att_cap]
        out["prr_attacked_random_timing"] = [float(v) for v in att_avg]
    return out


def prr_curves_multi(psr_grid, asr_grid, psr_db, scenario, n, sigma,
                     seeds, p_overlap=P_OVERLAP, n_mc=4000):
    """Run prr_curves per seed; return (mean-std-rounded dict, per-seed dict).
    Wave 14 / council round-2 34-c item: Monte-Carlo error bands on the
    harm-chain PRR deltas (previously a single MC seed, n=4000)."""
    per_seed = {}
    for s in seeds:
        c = prr_curves(psr_grid, asr_grid, psr_db, scenario, n, sigma,
                       p_overlap=p_overlap, n_mc=n_mc, seed=s,
                       keep_round=False)
        per_seed[str(s)] = c
    keys = ["prr_baseline", "prr_attacked_nocapture",
            "prr_attacked_capture", "prr_attacked_random_timing"]
    agg = {"d_m": per_seed[str(seeds[0])]["d_m"],
           "psr_db": per_seed[str(seeds[0])]["psr_db"],
           "false_idle_rate_pct": per_seed[str(seeds[0])]["false_idle_rate_pct"]}
    for k in keys:
        arr = np.array([per_seed[str(s)][k] for s in seeds])  # (n_seed, n_D)
        agg[k] = [round(float(v), 4) for v in arr.mean(0)]
        agg[k + "_std"] = [round(float(v), 5) for v in arr.std(0)]
    agg["mc_seeds"] = list(seeds)
    return agg, per_seed


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-seeds", type=int, default=5,
                    help="number of Monte-Carlo seeds (Wave 14 / 34-c item; "
                         "default 5; seeds 123/101/202/303/404)")
    args = ap.parse_args()
    MC_SEEDS = [123, 101, 202, 303, 404][:max(1, args.mc_seeds)]

    psr_grid, asr_grid, meap = load_targeted_curve("urban")
    n = SCENARIO_N["urban"]

    report = {
        "experiment": "Harm chain: false-IDLE -> unlicensed BSM TX -> PRR "
                      "loss at a third receiver (council P0, 21-c)",
        "provenance": {
            "asr_curve_source": "results/attack_results_merged.json "
                                "urban targeted_noise/cv2x_mask",
            "meap_targeted_mask_db": round(meap, 2),
            "note": "ASR(false-IDLE) values are MEASURED (interpolated from "
                    "the paper's attack runs); the link-budget layer below "
                    "is parametric and fully disclosed.",
        },
        "parameters": {
            "fspl_anchor_db_1m": round(FSPL0, 2),
            "path_loss_exponent": {"urban": 3.0, "highway": 2.0,
                                   "rural": 2.5},
            "victim_tx_dbm": P_TX_VICTIM,
            "cloaked_tx_dbm": P_TX_CLOAKED,
            "eirp_cap_its_dbm": EIRP_CAP_ITS,
            "eirp_cap_unii4_dbm": EIRP_CAP_UNII4,
            "noise_dbm_10mhz_nf9": NOISE_DBM,
            "gamma_dec_db": GAMMA_DEC,
            "gamma_capture_db": GAMMA_CAP,
            "shadowing_sigma_db": SHADOW_URBAN,
            "p_overlap": P_OVERLAP,
            "antenna_gain_dbi": 0.0,
        },
        "stage1_attacker_feasibility": stage1_feasibility(
            psr_grid, asr_grid, meap, n, "urban"),
        "stage2_prr": {},
        "stage2_prr_per_seed": {},
        "stage3_fleet_metrics": {},
        "mc_error_bands": {},
        "sensitivity": {},
        "mc_seeds": MC_SEEDS,
    }

    # stage 2: PRR curves at three attack operating points
    # (Wave 14: seed-ensemble mean curves; per-seed runs stored alongside)
    for tag, psr in [("at_meap_20pct", meap),
                     ("at_psr_minus10", -10.0),
                     ("at_psr_0", 0.0)]:
        agg, per_seed = prr_curves_multi(
            psr_grid, asr_grid, psr, "urban", n, SHADOW_URBAN, MC_SEEDS)
        report["stage2_prr"][tag] = agg
        report["stage2_prr_per_seed"][tag] = per_seed

        # MC error bands on the headline deltas (no-capture model)
        d = np.array(agg["d_m"])
        for D0 in (100.0, 150.0, 200.0):
            idx = int(np.argmin(np.abs(d - D0)))
            dps = []
            for s in MC_SEEDS:
                c = per_seed[str(s)]
                dps.append(100 * (c["prr_baseline"][idx]
                                  - c["prr_attacked_nocapture"][idx]))
            dps = np.array(dps)
            report["mc_error_bands"].setdefault(tag, {})[f"D={D0:.0f}m"] = {
                "dprr_pp_mean": round(float(dps.mean()), 2),
                "dprr_pp_std": round(float(dps.std()), 3),
                "dprr_pp_min": round(float(dps.min()), 2),
                "dprr_pp_max": round(float(dps.max()), 2),
                "n_mc_seeds": len(MC_SEEDS),
            }

    # stage 3: fleet deltas at safety-relevant distances
    for tag in ("at_meap_20pct", "at_psr_minus10"):
        c = report["stage2_prr"][tag]
        d = np.array(c["d_m"])
        b = np.array(c["prr_baseline"])
        for model_key, model_name in [
                ("prr_attacked_nocapture", "no-capture"),
                ("prr_attacked_capture", "capture")]:
            a = np.array(c[model_key])
            deltas = {}
            for D0 in (100.0, 150.0, 200.0):
                idx = int(np.argmin(np.abs(d - D0)))
                deltas[f"D={D0:.0f}m"] = {
                    "prr_base": round(float(b[idx]), 3),
                    "prr_attacked": round(float(a[idx]), 3),
                    "dprr_pp": round(100 * float(b[idx] - a[idx]), 2),
                    "extra_lost_bsms_per_1000": round(
                        1000 * float(b[idx] - a[idx]), 1),
                }
            report["stage3_fleet_metrics"].setdefault(tag, {})[
                model_name] = deltas

    # sensitivity: path-loss exponent and overlap sweeps at the MEAP point
    for n_s in (2.5, 3.5):
        report["sensitivity"][f"n={n_s}"] = prr_curves(
            psr_grid, asr_grid, meap, "urban", n_s, SHADOW_URBAN)[
                "prr_attacked_nocapture"]
    for ov in (0.4, 0.8):
        report["sensitivity"][f"overlap={ov}"] = prr_curves(
            psr_grid, asr_grid, meap, "urban", n, SHADOW_URBAN,
            p_overlap=ov)["prr_attacked_nocapture"]

    with open(os.path.join(OUT, "harm_chain.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("wrote results/harm_chain.json")

    # ---- compact console summary ----
    s1 = report["stage1_attacker_feasibility"]
    print("\n== Stage 1: attacker EIRP feasibility (urban n=3.0) ==")
    for tag, mf in s1["max_feasible_dA"].items():
        psr = s1["psr_points"][tag]["dB=100"]["psr_db"]
        asr = s1["psr_points"][tag]["dB=100"]["false_idle_pct"]
        print(f"  PSR {psr:6.1f} dB (false-IDLE {asr:5.1f}%): "
              f"max d_A @ cap 33 dBm  " +
              "  ".join(f"dB={k.split('=')[1]}: {v:.0f} m"
                        for k, v in mf.items()))
    print("\n== Stage 3: PRR loss (no-capture / capture) ==")
    for tag in ("at_meap_20pct", "at_psr_minus10"):
        for mn in ("no-capture", "capture"):
            m = report["stage3_fleet_metrics"][tag][mn]["D=150m"]
            print(f"  {tag:16s} {mn:10s} D=150 m: PRR {m['prr_base']:.3f} "
                  f"-> {m['prr_attacked']:.3f}  (dPRR {m['dprr_pp']:+.1f} pp, "
                  f"+{m['extra_lost_bsms_per_1000']:.0f} lost BSM/1000)")


if __name__ == "__main__":
    main()
