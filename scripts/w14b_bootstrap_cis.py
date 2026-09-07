#!/usr/bin/env python3
"""w14b_bootstrap_cis.py — Wave 14 / 35-b-11 (council 34-d iii): paired
bootstrap CIs for MEAP and PoC from per-window outcomes, replacing the
conservative-width bound propagation for every curve that has win_flags.

Why paired: the genie and mask arms attack the SAME evaluation windows, so
the PoC (mask MEAP - genie MEAP) difference is a PAIRED quantity — resampling
windows with replacement (same indices for both arms) propagates the shared
sampling noise instead of worst-casing each arm separately. This is the fix
for 34-d iii ("conservative CI propagation ... overstates width").

Covered:
  1. canonical PGD-10 dual model, model-seeds {7,123,456}:
     per-seed PoC bootstrap CI + 3-seed-mean bootstrap CI + point estimates.
  2. adaptive PGD-50xR5 grids, attack seeds {7, 11, 22} per defense
     (dual/at/trades): seed 7 via the per-window capture files
     (w14b_perwindow_adaptive_*_s7.json, continuity-checked against the
     canonical grids), seeds 11/22 via the item-12 replication grids.
     Per-seed paired PoC bootstrap CIs + a 3-seed-mean bootstrap and
     t-interval per defense.

Method: B=2000 percentile bootstrap; meap_curve (20% threshold,
censoring-aware) per resample; reps where a curve fails to cross are counted
and dropped (disclosed). Reproducible: default_rng(20260907).

Output: results/w14b_bootstrap_cis.json
"""
import os, sys, json, glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
from attack_mask import meap_curve

OUT = os.path.join(ROOT, "results")
B = 2000
RNG_SEED = 20260907


def load_capture(path):
    d = json.load(open(path))
    arms = {}
    for key, cells in d["runs"].items():
        setting = key.split("/")[-1]
        psrs, flags = [], []
        for k in sorted(cells, key=lambda s: float(s[4:-2])):
            c = cells[k]
            if "win_flags" not in c:
                return None          # capture-free file
            psrs.append(float(k[4:-2]))
            flags.append(np.asarray(c["win_flags"], dtype=np.int64))
        arms[setting] = (np.asarray(psrs), np.asarray(flags))
    return arms, d["config"]


def curve_from_flags(flags, idx):
    """cond-ASR (%) per psr for resampled windows idx (paired across arms).

    flags: (n_psr, n_windows) with values 1/0/-1; eligible = flags >= 0.
    """
    f = flags[:, idx]
    elig = f >= 0
    n_el = elig.sum(axis=1)
    k = ((f == 1) & elig).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        asr = np.where(n_el > 0, 100.0 * k / np.maximum(n_el, 1), 0.0)
    return asr, n_el


def bootstrap_poc(arms, rng):
    """Percentile bootstrap of (meap_genie, meap_mask, poc). Returns dict."""
    psrs_g, flags_g = arms["genie"]
    psrs_m, flags_m = arms["cv2x_mask"]
    n = flags_g.shape[1]
    mg_s, mm_s, poc_s = [], [], []
    dropped = 0
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        asr_g, _ = curve_from_flags(flags_g, idx)
        asr_m, _ = curve_from_flags(flags_m, idx)
        mg, cg = meap_curve(list(psrs_g), list(asr_g), 20.0)
        mm, cm = meap_curve(list(psrs_m), list(asr_m), 20.0)
        if mg is None or mm is None:
            dropped += 1
            continue
        mg_s.append(mg)
        mm_s.append(mm)
        poc_s.append(mm - mg)
    out = {
        "n_reps": len(poc_s), "dropped_no_crossing": dropped,
    }
    if poc_s:
        out["meap_genie_boot_mean_db"] = round(float(np.mean(mg_s)), 3)
        out["meap_mask_boot_mean_db"] = round(float(np.mean(mm_s)), 3)
        out["meap_genie_ci95_db"] = [round(float(np.percentile(mg_s, 2.5)), 2),
                                     round(float(np.percentile(mg_s, 97.5)), 2)]
        out["meap_mask_ci95_db"] = [round(float(np.percentile(mm_s, 2.5)), 2),
                                    round(float(np.percentile(mm_s, 97.5)), 2)]
        out["poc_ci95_db"] = [round(float(np.percentile(poc_s, 2.5)), 2),
                              round(float(np.percentile(poc_s, 97.5)), 2)]
        out["poc_boot_mean_db"] = round(float(np.mean(poc_s)), 3)
    return out


def point_poc(arms):
    psrs_g, flags_g = arms["genie"]
    psrs_m, flags_m = arms["cv2x_mask"]
    asr_g, _ = curve_from_flags(flags_g, np.arange(flags_g.shape[1]))
    asr_m, _ = curve_from_flags(flags_m, np.arange(flags_m.shape[1]))
    mg, _ = meap_curve(list(psrs_g), list(asr_g), 20.0)
    mm, _ = meap_curve(list(psrs_m), list(asr_m), 20.0)
    return mg, mm, (None if (mg is None or mm is None) else mm - mg)


def t_ci(vals, alpha=0.05):
    """Student-t CI for the mean of vals."""
    v = np.asarray(vals, dtype=float)
    if v.size < 2:
        return None
    from math import sqrt
    try:
        from scipy import stats
        tq = stats.t.ppf(1 - alpha / 2, df=v.size - 1)
    except Exception:
        tq = 2.776 if v.size == 4 else (4.303 if v.size == 3 else 12.706)
    m = v.mean()
    h = tq * v.std(ddof=1) / sqrt(v.size)
    return [round(m - h, 3), round(m + h, 3)]


def main():
    rng = np.random.default_rng(RNG_SEED)
    report = {
        "experiment": "paired-bootstrap CIs for MEAP/PoC from per-window "
                      "outcomes (35-b-11, council 34-d iii)",
        "method": {
            "bootstrap": f"percentile, B={B}, rng default_rng({RNG_SEED})",
            "pairing": "genie and mask arms attack the SAME windows — the "
                       "same resampled indices are applied to both arms "
                       "(paired difference), replacing the conservative "
                       "worst-case bound propagation",
            "meap": "meap_curve (20% threshold, censoring-aware) per "
                    "resample; reps where a curve fails to cross 20% are "
                    "dropped and counted",
            "seeds": "canonical protocol is MODEL-seed replication "
                     "(retrained checkpoints, attack eval seed 7); the "
                     "3-seed mean resamples each seed independently",
        },
        "canonical": {},
        "adaptive": {},
        "comparison": {},
    }

    # ---- 1. canonical PGD-10, model seeds 7/123/456 ----
    canon = {}
    per_seed_pocs = []
    for seed in (7, 123, 456):
        path = os.path.join(OUT, f"w14b_perwindow_s{seed}.json")
        if not os.path.exists(path):
            continue
        arms, cfg = load_capture(path)
        mg, mm, poc = point_poc(arms)
        boot = bootstrap_poc(arms, rng)
        canon[f"model_seed_{seed}"] = {
            "checkpoint": cfg.get("checkpoint"),
            "meap_genie_db": None if mg is None else round(mg, 3),
            "meap_mask_db": None if mm is None else round(mm, 3),
            "poc_db": None if poc is None else round(poc, 3),
            **boot,
        }
        if poc is not None:
            per_seed_pocs.append(poc)
        print(f"[canon s{seed}] PoC {None if poc is None else round(poc,3)} "
              f"boot CI {boot.get('poc_ci95_db')} "
              f"(dropped {boot['dropped_no_crossing']}/{B})", flush=True)

    # 3-seed mean bootstrap (each seed independently resampled)
    if len(per_seed_pocs) == 3:
        arms_list = [load_capture(os.path.join(OUT, f"w14b_perwindow_s{s}.json"))[0]
                     for s in (7, 123, 456)]
        n = arms_list[0]["genie"][1].shape[1]
        means = []
        for _ in range(B):
            rep_pocs = []
            for arms in arms_list:
                idx = rng.integers(0, n, size=n)
                asr_g, _ = curve_from_flags(arms["genie"][1], idx)
                asr_m, _ = curve_from_flags(arms["cv2x_mask"][1], idx)
                mg, _ = meap_curve(list(arms["genie"][0]), list(asr_g), 20.0)
                mm, _ = meap_curve(list(arms["cv2x_mask"][0]),
                                   list(asr_m), 20.0)
                if mg is not None and mm is not None:
                    rep_pocs.append(mm - mg)
            if len(rep_pocs) == 3:
                means.append(float(np.mean(rep_pocs)))
        canon["three_seed_mean"] = {
            "point_pocs_db": [round(p, 3) for p in per_seed_pocs],
            "point_mean_db": round(float(np.mean(per_seed_pocs)), 3),
            "point_spread_db": round(float(np.ptp(per_seed_pocs)), 3),
            "t_ci95_db": t_ci(per_seed_pocs),
            "boot_mean_ci95_db": [round(float(np.percentile(means, 2.5)), 2),
                                  round(float(np.percentile(means, 97.5)), 2)],
            "n_reps": len(means),
        }
        print(f"[canon 3-seed] mean "
              f"{round(float(np.mean(per_seed_pocs)),3)} dB "
              f"t-CI {t_ci(per_seed_pocs)} "
              f"boot-mean CI "
              f"[{np.percentile(means,2.5):.2f},{np.percentile(means,97.5):.2f}]",
              flush=True)
    report["canonical"] = canon

    # ---- 2. adaptive grids with win_flags (attack-seed replication) ----
    # seed 7: per-window capture files from w14b_adaptive_capture.py
    #         (the canonical adaptive_*_s7_r5.json files stay flag-free and
    #          untouched; the capture re-runs reproduced every stored
    #          cond_asr exactly - continuity_all_match)
    # seeds 11/22: the item-12 replication grids (win_flags stored inline)
    for defense in ("dual", "at", "trades"):
        entries = {}
        arms_by_seed = {}
        cap_path = os.path.join(
            OUT, f"w14b_perwindow_adaptive_{defense}_s7.json")
        if os.path.exists(cap_path):
            cap_doc = json.load(open(cap_path))
            cont = all(c.get("matches_stored_canonical")
                       for v in cap_doc["runs"].values()
                       for c in v.values())
            arms, cfg = load_capture(cap_path)
            if arms is not None:
                mg, mm, poc = point_poc(arms)
                boot = bootstrap_poc(arms, rng)
                entries["attack_seed_7"] = {
                    "source": os.path.basename(cap_path),
                    "continuity_all_match": bool(cont),
                    "meap_genie_db": None if mg is None else round(mg, 3),
                    "meap_mask_db": None if mm is None else round(mm, 3),
                    "poc_db": None if poc is None else round(poc, 3),
                    **boot,
                }
                if poc is not None:
                    arms_by_seed[7] = (arms, poc)
                print(f"[adaptive {defense} s7] PoC "
                      f"{None if poc is None else round(poc,3)} "
                      f"boot CI {boot.get('poc_ci95_db')} "
                      f"continuity {cont}", flush=True)
        for s in (11, 22):
            path = os.path.join(OUT, f"adaptive_{defense}_s{s}_r5.json")
            if not os.path.exists(path):
                continue
            try:
                arms, cfg = load_capture(path)
            except Exception:
                arms = None
            if arms is None:
                continue
            mg, mm, poc = point_poc(arms)
            boot = bootstrap_poc(arms, rng)
            entries[f"attack_seed_{s}"] = {
                "source": os.path.basename(path),
                "meap_genie_db": None if mg is None else round(mg, 3),
                "meap_mask_db": None if mm is None else round(mm, 3),
                "poc_db": None if poc is None else round(poc, 3),
                **boot,
            }
            if poc is not None:
                arms_by_seed[s] = (arms, poc)
            print(f"[adaptive {defense} s{s}] PoC "
                  f"{None if poc is None else round(poc,3)} "
                  f"boot CI {boot.get('poc_ci95_db')}", flush=True)

        # 3-seed-mean bootstrap (each seed independently resampled, paired
        # within seed) + t-interval over the point PoCs
        if len(arms_by_seed) == 3:
            seeds = sorted(arms_by_seed)
            n = arms_by_seed[seeds[0]][0]["genie"][1].shape[1]
            means = []
            for _ in range(B):
                rep_pocs = []
                for s in seeds:
                    arms, _poc = arms_by_seed[s]
                    idx = rng.integers(0, n, size=n)
                    asr_g, _ = curve_from_flags(arms["genie"][1], idx)
                    asr_m, _ = curve_from_flags(arms["cv2x_mask"][1], idx)
                    mg, _ = meap_curve(list(arms["genie"][0]),
                                       list(asr_g), 20.0)
                    mm, _ = meap_curve(list(arms["cv2x_mask"][0]),
                                       list(asr_m), 20.0)
                    if mg is not None and mm is not None:
                        rep_pocs.append(mm - mg)
                if len(rep_pocs) == 3:
                    means.append(float(np.mean(rep_pocs)))
            pt = [arms_by_seed[s][1] for s in seeds]
            entries["three_seed_mean"] = {
                "point_pocs_db": [round(p, 3) for p in pt],
                "point_mean_db": round(float(np.mean(pt)), 3),
                "point_spread_db": round(float(np.ptp(pt)), 3),
                "t_ci95_db": t_ci(pt),
                "boot_mean_ci95_db": [
                    round(float(np.percentile(means, 2.5)), 2),
                    round(float(np.percentile(means, 97.5)), 2)],
                "n_reps": len(means),
            }
            print(f"[adaptive {defense} 3-seed] mean "
                  f"{round(float(np.mean(pt)),3)} dB t-CI {t_ci(pt)} "
                  f"boot-mean CI "
                  f"[{np.percentile(means,2.5):.2f},"
                  f"{np.percentile(means,97.5):.2f}]", flush=True)
        report["adaptive"][defense] = entries

    # ---- 3. comparison vs conservative propagation ----
    try:
        w12 = json.load(open(os.path.join(OUT,
                                          "w12_confidence_intervals.json")))
        p0 = w12["headline_pairs"][0]
        report["comparison"] = {
            "conservative_headline_poc_ci95": p0["poc_ci95"],
            "note": "the w12 conservative propagation (mask_lo - genie_hi, "
                    "mask_hi - genie_lo) worst-cases each arm separately; "
                    "the paired bootstrap above uses the shared-window "
                    "correlation",
        }
    except Exception:
        pass

    json.dump(report, open(os.path.join(OUT, "w14b_bootstrap_cis.json"),
                           "w"), indent=2)
    print("wrote results/w14b_bootstrap_cis.json", flush=True)


if __name__ == "__main__":
    main()
