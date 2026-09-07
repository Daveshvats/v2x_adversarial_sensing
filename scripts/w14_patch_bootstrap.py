"""Patch w14b_bootstrap_cis.py section 2: replace the glob-based adaptive
section with a per-defense loop that (a) loads the seed-7 per-window capture
files (w14b_perwindow_adaptive_{defense}_s7.json, continuity-checked),
(b) loads the s11/s22 replication grids, and (c) adds a 3-seed-mean
bootstrap + t-interval per defense (mirroring the canonical section).
Sections 1 (canonical PGD-10) and 3 (comparison) are preserved byte-exact.
"""
import re

PATH = "scripts/w14b_bootstrap_cis.py"
src = open(PATH).read()

START = "    # ---- 2. adaptive grids with win_flags (attack-seed replication) ----"
END = "    # ---- 3. comparison vs conservative propagation ----"

assert src.count(START) == 1, "start marker"
assert src.count(END) == 1, "end marker"
i, j = src.index(START), src.index(END)

NEW = '''    # ---- 2. adaptive grids with win_flags (attack-seed replication) ----
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

'''

src = src[:i] + NEW + src[j:]

# also update the stale seed_level note (s7 now has capture)
OLD_NOTE = '"note": "t-interval over attack seeds; s7 grid predates the "\n                        "capture flag (point PoC only); s11/s22 bootstrap "\n                        "CIs above",'
NEW_NOTE = '"note": "t-interval over attack seeds; seed 7 point PoC now "\n                        "backed by the per-window capture "\n                        "(w14b_perwindow_adaptive_*_s7.json); see "\n                        "three_seed_mean for the n=3 bootstrap",'
if OLD_NOTE in src:
    src = src.replace(OLD_NOTE, NEW_NOTE)
    print("seed_level note updated")
else:
    print("WARN: seed_level note not found verbatim - left unchanged")

open(PATH, "w").write(src)
print("section 2 replaced")
