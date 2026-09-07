#!/usr/bin/env python3
"""run_fusion_baseline.py — Wave 14 / 35-c-4 (council 34-c external validity:
"cooperative sensing with multiple receivers is the deployment-realistic
detection floor; the weak single-receiver confidence gate may not represent
it").

K-receiver cooperative fusion baseline against the compliant attacker:
  * K receivers of the SAME transmission (same underlying signal s, class y):
    each receiver gets an INDEPENDENT victim channel, noise draw, and
    attacker->receiver channel — exactly the canonical per-window draw
    process, K times. Receiver 1's draw order preserves the canonical
    build_eval_set stream, so receiver-1 windows are IDENTICAL to
    attack_results_merged.json's eval set (continuity-checkable).
  * The attacker optimizes delta on receiver 1's chain only (single-receiver
    CSI — the canonical PGD-10 protocol, budget = receiver-1 clean window
    energy). The SAME delta is then applied through every receiver's
    attacker channel: r_k = x_rx_k + conv(delta, h_a_k). This is the
    realistic unknown-receiver-geometry case; an attacker with full CSI of
    all K receivers could do better (disclosed residual, not adaptive here).
  * Fusion decision: majority vote over the K per-receiver predictions.
    Eligible window = fused CLEAN decision correct; fusion attack success =
    fused ADVERSARIAL decision wrong.
  * K in {1, 3, 5}: K=1 reproduces the canonical single-receiver numbers
    (control); K=3/5 quantify the spatial-diversity detection gain (or its
    absence) in MEAP dB.

Arms: genie (power-only) and cv2x_mask (flat-cap compliant), urban,
untargeted, PGD-10, alpha 0.25, psd_margin 2.0, PSR grid -45..0, seed 7.

Output: results/w14b_fusion_baseline.json   (chunked/idempotent)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import sys, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch
torch.set_num_threads(2)

from waveforms import gen_signal, ATTACK_BANDS, NOISE_CLASS
import channels as CH
from receiver import FrontEnd, DualStreamModel
from attack_mask import waveform_pgd, meap_curve

OUT = os.path.join(ROOT, "results")
K_LIST = [1, 3, 5]
PSR_GRID = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
            -10.0, -5.0, 0.0]
L = 2048
RESULT_JSON = os.path.join(OUT, "w14b_fusion_baseline.json")


def build_k_receiver_eval_set(n_per_class, seed, k, scenario="urban"):
    """K receivers of the same transmissions.

    Receiver 1 consumes the CANONICAL stream (rng order identical to
    build_eval_set, so ALL receiver-1 windows are bit-identical to the
    canonical eval set); receivers 2..K use independent per-receiver rng
    streams (seed*7919 + j) with the same draw procedure — statistically
    independent channels/noise/attacker links, zero interference with the
    canonical stream.
    """
    rng = np.random.default_rng(seed)
    rngs_extra = [np.random.default_rng(seed * 7919 + j)
                  for j in range(1, k)]
    X, Y, H, P = [], [], [], []
    for cls in (0, 1, 2):
        for _ in range(n_per_class):
            s = gen_signal(cls, rng)
            # ---- receiver 1: canonical stream ----
            h_v = CH.draw_channel(scenario, rng)
            r = np.convolve(s, h_v)[(len(h_v) - 1) // 2:
                                    (len(h_v) - 1) // 2 + L]
            p_clean1 = np.sum(np.abs(r) ** 2)
            p = np.mean(np.abs(r) ** 2)
            snr = rng.uniform(5, 25)
            ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
            r = r + ns * (rng.standard_normal(L) +
                          1j * rng.standard_normal(L))
            h_a = CH.draw_channel(scenario, rng)
            per_rx = [(r.astype(np.complex64), h_a)]
            # ---- receivers 2..K: independent streams ----
            for j, rk in enumerate(rngs_extra, start=1):
                h_vj = CH.draw_channel(scenario, rk)
                rj = np.convolve(s, h_vj)[(len(h_vj) - 1) // 2:
                                          (len(h_vj) - 1) // 2 + L]
                pj = np.mean(np.abs(rj) ** 2)
                snrj = rk.uniform(5, 25)
                nsj = np.sqrt(pj / (10 ** (snrj / 10.0)) / 2.0)
                rj = rj + nsj * (rk.standard_normal(L) +
                                 1j * rk.standard_normal(L))
                h_aj = CH.draw_channel(scenario, rk)
                per_rx.append((rj.astype(np.complex64), h_aj))
            X.append(per_rx)
            Y.append(cls)
            P.append(p_clean1)
    X_t = torch.from_numpy(np.array([[x[0] for x in win] for win in X]))
    H_t = torch.from_numpy(np.array([[x[1] for x in win] for win in X]))
    return (X_t, torch.tensor(Y, dtype=torch.long),
            H_t, torch.tensor(P, dtype=torch.float32))


def majority(preds_k):
    """preds_k: (K, B) -> fused majority prediction (B,)."""
    stacked = torch.stack(preds_k, dim=0)              # (K, B)
    fused = []
    for c in range(stacked.size(1)):
        vals, counts = torch.unique(stacked[:, c], return_counts=True)
        fused.append(vals[counts.argmax()])
    return torch.stack(fused)


def forward_all(model, frontend, x_rx, h_a, delta=None):
    """Per-receiver predictions; optionally add conv(delta, h_a) first.

    x_rx: (B, L); h_a: (B, Lh) — the SAME B windows (chunked).
    """
    preds = []
    for i in range(0, x_rx.size(0), 300):
        r = x_rx[i:i + 300]
        if delta is not None:
            from attack_mask import cconv
            h = torch.as_tensor(h_a[i:i + 300], dtype=r.dtype)
            r = r + cconv(delta[i:i + 300], h)
        with torch.no_grad():
            preds.append(model.forward_wave(r, frontend).argmax(1))
    return torch.cat(preds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--max-time-sec", type=float, default=520.0)
    args = ap.parse_args()
    t0 = time.time()
    torch.manual_seed(7)

    ck = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                    map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ck["model"])
    model.eval()
    frontend = FrontEnd()
    frontend.set_stats(ck["mag_mean"], ck["mag_std"])

    report = json.load(open(RESULT_JSON)) if os.path.exists(RESULT_JSON) else {
        "experiment": "K-receiver cooperative-sensing fusion baseline vs the "
                      "compliant attacker (35-c-4, council 34-c external "
                      "validity)",
        "config": {
            "scenario": "urban", "mode": "untargeted", "steps": 10,
            "attack_eval_seed": 7, "n_eval_per_class": 100,
            "attack_pgd_alpha_frac": 0.25, "psd_cap_margin": 2.0,
            "psr_db": list(PSR_GRID),
            "k_receivers": K_LIST,
            "fusion": "majority vote over K per-receiver predictions; "
                      "eligible = fused clean decision correct; success = "
                      "fused adversarial decision wrong",
            "attacker_csi": "single-receiver (receiver-1 chain only, "
                            "canonical protocol); same delta applied through "
                            "every receiver's attacker channel — NOT "
                            "fusion-adaptive (disclosed residual)",
            "budget_reference": "receiver-1 clean window energy (PSR "
                                "nominal); per-receiver effective PSR "
                                "disclosed per cell",
            "receiver1_continuity": "receiver-1 draw order preserves "
                                    "build_eval_set(urban,100,7), so K=1 "
                                    "reproduces the canonical numbers",
        },
        "runs": {},
        "elapsed_s": 0.0,
    }

    k_max = max(K_LIST)
    X, Y, H, P = build_k_receiver_eval_set(100, 7, k_max)
    # clean predictions per receiver (K_max receivers)
    clean_preds_k = [forward_all(model, frontend, X[:, j], H[:, j])
                     for j in range(k_max)]
    clean_acc1 = (clean_preds_k[0] == Y).float().mean().item()
    report["clean_acc_receiver1"] = round(clean_acc1, 4)
    print(f"[fusion] receiver-1 clean acc: {clean_acc1*100:.2f}%", flush=True)

    runs = report["runs"]
    for setting in ("genie", "cv2x_mask"):
        key = "untargeted/" + setting
        runs.setdefault(key, {})
        sband = None if setting == "genie" else ATTACK_BANDS["cv2x_attacker"]
        for psr in PSR_GRID:
            cell = f"psr={psr:+.0f}dB"
            if cell in runs[key]:
                continue
            if time.time() - t0 > args.max_time_sec:
                print("[fusion] time budget — flush and exit", flush=True)
                json.dump(report, open(RESULT_JSON, "w"), indent=2)
                return
            p_budget = P * (10 ** (psr / 10.0))
            # attack on receiver-1 chain (canonical protocol, PGD-10)
            delta = None
            preds1 = []
            for i in range(0, X.size(0), 300):
                out = waveform_pgd(model, frontend, X[:, 0][i:i + 300],
                                   H[:, 0][i:i + 300], Y[i:i + 300],
                                   p_budget[i:i + 300], steps=10, band=sband,
                                   targeted=False, target_class=NOISE_CLASS,
                                   return_delta=True)
                preds1.append(out["preds"])
                if delta is None:
                    delta = out["delta"]
                else:
                    delta = torch.cat([delta, out["delta"]])
            preds1 = torch.cat(preds1)

            # per-K fusion (receivers 1..K of the k_max built)
            per_k = {}
            for k in K_LIST:
                adv_preds_k = [preds1 if j == 0 else forward_all(
                    model, frontend, X[:, j], H[:, j], delta)
                    for j in range(k)]
                fused_clean = majority([clean_preds_k[j] for j in range(k)])
                fused_adv = majority(adv_preds_k)
                elig = (fused_clean == Y)
                n_elig = int(elig.sum())
                succ = int(((fused_adv != Y) & elig).sum())
                per_k[f"K={k}"] = {
                    "cond_asr": round(100 * succ / max(n_elig, 1), 2),
                    "n_eligible": n_elig,
                }
            # per-receiver ASR (transfer view) for k_max receivers
            per_rx_asr = []
            for j in range(k_max):
                pr = preds1 if j == 0 else forward_all(
                    model, frontend, X[:, j], H[:, j], delta)
                e = (clean_preds_k[j] == Y)
                per_rx_asr.append(round(
                    100 * float(((pr != Y) & e).float().sum() /
                                max(float(e.float().sum()), 1)), 2))
            # effective PSR per receiver (mean over windows)
            from attack_mask import cconv
            eff = []
            for j in range(k_max):
                e_all = []
                for i in range(0, X.size(0), 300):
                    h = torch.as_tensor(H[:, j][i:i + 300],
                                        dtype=X.dtype)
                    d = delta[i:i + 300]
                    p_d = cconv(d, h).abs().pow(2).sum(dim=1)
                    # receiver-k clean window energy
                    p_ck = X[:, j][i:i + 300].abs().pow(2).sum(dim=1)
                    e_all.append(10 * torch.log10(
                        p_d / (p_ck + 1e-30) + 1e-30))
                eff.append(round(float(torch.cat(e_all).mean()), 2))
            runs[key][cell] = {
                "per_k": per_k,
                "per_receiver_asr_pct": per_rx_asr,
                "per_receiver_effective_psr_db_mean": eff,
            }
            report["elapsed_s"] = round(report.get("elapsed_s", 0.0) +
                                        (time.time() - t0), 1)
            json.dump(report, open(RESULT_JSON, "w"), indent=2)
            print(f"[fusion] {key} {cell}: K1 {per_k['K=1']['cond_asr']}%  "
                  f"K3 {per_k['K=3']['cond_asr']}%  "
                  f"K5 {per_k['K=5']['cond_asr']}%", flush=True)
            t0 = time.time()

    # ---- MEAP summaries per K ----
    summary = {}
    for setting in ("genie", "cv2x_mask"):
        key = "untargeted/" + setting
        for k in K_LIST:
            asrs = [runs[key][f"psr={p:+.0f}dB"]["per_k"][f"K={k}"]
                    ["cond_asr"] for p in PSR_GRID]
            mg, cg = meap_curve(PSR_GRID, asrs, 20.0)
            summary[f"{setting}_K{k}_meap_db"] = mg
            summary[f"{setting}_K{k}_censor"] = cg
    for k in K_LIST:
        gm = summary.get(f"genie_K{k}_meap_db")
        mm = summary.get(f"cv2x_mask_K{k}_meap_db")
        if gm is not None and mm is not None:
            summary[f"poc_K{k}_db"] = round(mm - gm, 3)
    report["summary"] = summary
    report["summary_note"] = {
        "diversity_gain_db": "cv2x_mask K1 MEAP - K5 MEAP (positive = "
                             "fusion raises the attack's minimum power)",
        "disclosed_limit": "single-receiver-CSI attack; a fusion-adaptive "
                           "attacker (full CSI of all K receivers) is NOT "
                           "evaluated here",
    }
    json.dump(report, open(RESULT_JSON, "w"), indent=2)
    print(json.dumps(summary, indent=1), flush=True)
    print("FUSION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
