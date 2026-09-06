#!/usr/bin/env python3
"""run_real_wifi.py — Wave 6 G1: real OTA WiFi signal in the attack loop.

Questions answered (each against the synthetic-task canonical numbers):
  Q1 domain gap  : does the frozen model (trained on synthetic WiFi) recognize
                   real OTA WiFi bursts as the WiFi class?  (frozen probe)
  Q2 real attack : with the WiFi class REPLACED by real captures in the eval
                   set, what are the genie/compliant MEAP and PoC for the
                   frozen model?
  Q3 adapted     : after a leakage-free fine-tune that restores clean accuracy
                   on real WiFi, do the MEAP/PoC conclusions survive? — this
                   is the headline: "compliance still costs the attacker ~X dB
                   and the undefended model still breaks at deeply negative
                   PSR when the victim signal is REAL."

Protocol notes (honesty):
  * Real WiFi windows: 96 burst-gated, unit-power-normalized, NON-overlapping
    (eval protocol). NO synthetic victim channel is applied to them (the real
    propagation channel is embedded in the capture); receiver noise is added
    at drawn SNR like every other class.
  * PC5 / 11p classes: synthetic generators + TR 37.885-style victim channel
    (the canonical pipeline, unchanged).
  * Attacker link h_a: drawn per sample as in the canonical attack runs; the
    attack chain itself is untouched.
  * Fine-tune: hop-512 overlapping training windows EXCLUDED from any overlap
    with the 96 eval windows (start-index guard); synthetic classes refreshed
    from the same generators; 15 epochs max, Adam 5e-4, seed 42.
  * PSR reference for real windows: unit mean power => window energy 2048
    (constant), vs per-sample channel-dependent for synthetic classes —
    disclosed asymmetry (controlled: real windows all reference the same PSR).
"""
import sys, os, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

import numpy as np
import torch

from waveforms import gen_signal, CLASS_NAMES, NOISE_CLASS, ATTACK_BANDS
import channels as CH
from receiver import FrontEnd, DualStreamModel
from real_wifi import build_real_wifi_pool
from attack_mask import waveform_pgd, cond_asr, meap_curve, price_of_compliance

OUT = os.path.join(ROOT, "results")
DATA_DEFAULT = "/home/z/my-project/download/wave6_data/extracted"
DEFAULT_PSR = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
               -10.0, -5.0, 0.0, 5.0, 10.0]


def build_mixed_eval_set(pool, n_synth_per_class, seed, scenario="urban"):
    """Eval set with the WiFi class REPLACED by real captures.

    PC5/11p: synthetic + victim channel + noise (canonical). Real WiFi: as
    captured (real channel inside) + noise at drawn SNR. Attacker channel
    drawn for all. Returns x_rx, y, h_a, p_clean, real_mask.
    """
    rng = np.random.default_rng(seed)
    X, y, H, P, RM = [], [], [], [], []
    L = 2048
    for cls in (0, 1):                       # synthetic active classes
        for _ in range(n_synth_per_class):
            s = gen_signal(cls, rng)
            h_v = CH.draw_channel(scenario, rng)
            r = np.convolve(s, h_v)[(len(h_v) - 1) // 2:
                                    (len(h_v) - 1) // 2 + L]
            p_clean = np.sum(np.abs(r) ** 2)
            snr = rng.uniform(5, 25)
            ns = np.sqrt(np.mean(np.abs(r) ** 2) / (10 ** (snr / 10.0)) / 2.0)
            r = r + ns * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
            h_a = CH.draw_channel(scenario, rng)
            X.append(r); y.append(cls); H.append(h_a); P.append(p_clean); RM.append(False)
    for w in pool:                           # real WiFi windows
        p_clean = np.sum(np.abs(w) ** 2)     # = 2048 (unit power)
        snr = rng.uniform(5, 25)
        ns = np.sqrt(np.mean(np.abs(w) ** 2) / (10 ** (snr / 10.0)) / 2.0)
        r = w + ns * (rng.standard_normal(L) + 1j * rng.standard_normal(L))
        h_a = CH.draw_channel(scenario, rng)
        X.append(r); y.append(2); H.append(h_a); P.append(p_clean); RM.append(True)
    return (torch.from_numpy(np.array(X, dtype=np.complex64)),
            torch.tensor(y, dtype=torch.long),
            torch.from_numpy(np.array(H, dtype=np.complex64)),
            torch.tensor(np.array(P), dtype=torch.float32),
            torch.tensor(np.array(RM)))


def attack_grid(model, fe, x_rx, h_a, y, p_clean, psr_grid, steps, band,
                targeted=False):
    """One attack setting across the PSR grid. Returns {psr: (cond_asr, n, rob)}."""
    res = {}
    clean_preds = []
    with torch.no_grad():
        for i in range(0, x_rx.size(0), 256):
            clean_preds.append(model.forward_wave(x_rx[i:i + 256], fe).argmax(1))
    clean_preds = torch.cat(clean_preds)
    for psr in psr_grid:
        p_budget = p_clean * (10 ** (psr / 10.0))
        preds = []
        for i in range(0, x_rx.size(0), 300):
            out = waveform_pgd(model, fe, x_rx[i:i + 300], h_a[i:i + 300],
                               y[i:i + 300], p_budget[i:i + 300], steps=steps,
                               band=band, targeted=targeted,
                               target_class=NOISE_CLASS)
            preds.append(out["preds"])
        adv = torch.cat(preds)
        c, n = cond_asr(clean_preds, adv, y, targeted=targeted,
                        target=NOISE_CLASS)
        rob = (adv == y).float().mean().item()
        res[psr] = (float(c), int(n), float(rob))
    return res, clean_preds


def fine_tune_on_real(pool_train, seed=42, epochs=15, lr=5e-4,
                      n_synth_per_class=600):
    """Leakage-free fine-tune of the frozen dual model on the mixed task
    where the WiFi class = real training windows (hop-512 augmented)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in (0, 1, 3):                    # PC5, 11p, Noise synthetic
        for _ in range(n_synth_per_class):
            s = gen_signal(cls, rng)
            scen = CH.SCENARIO_NAMES[(cls * 200 + _) % 3]
            h_v = CH.draw_channel(scen, rng)
            r = np.convolve(s, h_v)[(len(h_v) - 1) // 2:
                                    (len(h_v) - 1) // 2 + 2048]
            p = np.mean(np.abs(r) ** 2)
            snr = rng.uniform(5, 25)
            ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
            r = r + ns * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
            X.append(r); y.append(cls)
    for w in pool_train:                     # real WiFi
        p = np.mean(np.abs(w) ** 2)
        snr = rng.uniform(5, 25)
        ns = np.sqrt(p / (10 ** (snr / 10.0)) / 2.0)
        r = w + ns * (rng.standard_normal(2048) + 1j * rng.standard_normal(2048))
        X.append(r); y.append(2)
    X = torch.from_numpy(np.array(X, dtype=np.complex64))
    y = torch.from_numpy(np.array(y)).long()
    perm = torch.randperm(X.size(0))
    X, y = X[perm], y[perm]
    n_val = int(0.2 * X.size(0))
    Xv, yv, Xt, yt = X[:n_val], y[:n_val], X[n_val:], y[n_val:]

    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel()
    model.load_state_dict(ckpt["model"])
    fe = FrontEnd()
    fe.set_stats(ckpt["mag_mean"], ckpt["mag_std"])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(Xt.size(0))
        for i in range(0, Xt.size(0), 64):
            idx = perm[i:i + 64]
            logits = model.forward_wave(Xt[idx], fe)
            loss = torch.nn.functional.cross_entropy(logits, yt[idx],
                                                     label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = [model.forward_wave(Xv[i:i + 256], fe).argmax(1)
                     for i in range(0, Xv.size(0), 256)]
            acc = (torch.cat(preds) == yv).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone()
                                         for k, v in model.state_dict().items()}
        print(f"    ft epoch {ep+1:2d}: val acc {acc*100:5.2f}% "
              f"best {best_acc*100:5.2f}%", flush=True)
    model.load_state_dict(best_state)
    return model, fe, best_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--n-synth", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--psr", nargs="+", type=float, default=DEFAULT_PSR)
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--stage", default="all",
                    choices=["all", "frozen", "finetune"],
                    help="chunked execution: 'frozen' runs frozen grids only; "
                         "'finetune' loads the frozen JSON and runs ft+grids")
    args = ap.parse_args()
    t0 = time.time()

    # --- eval pool (non-overlapping, the 96 gated bursts) -------------------
    pool, metas, prov, win_refs = build_real_wifi_pool(args.data, verbose=True)
    print(f"real WiFi eval pool: {pool.shape[0]} windows", flush=True)

    # --- location split (leakage-free by construction) ----------------------
    # train: rabot + reep captures; eval: uz captures. Disjoint capture
    # locations -> zero shared samples AND zero shared propagation
    # environments. Disclosed in the JSON config.
    eval_idx = [i for i, (fn, s) in enumerate(win_refs) if "_uz_" in fn]
    pool = pool[eval_idx]
    prov["eval_files"] = sorted(set(win_refs[i][0] for i in eval_idx))
    pool_tr, _, _, refs_tr = build_real_wifi_pool(args.data, hop=512)
    keep_tr = np.array(["_rabot_" in fn or "_reep_" in fn for (fn, s) in refs_tr])
    pool_tr = pool_tr[keep_tr]
    prov["train_files"] = sorted(set(fn for fn, k in zip(
        [r[0] for r in refs_tr], keep_tr) if k))
    print(f"eval pool (uz only): {pool.shape[0]} windows; "
          f"train pool (rabot/reep, hop-512): {pool_tr.shape[0]} windows",
          flush=True)
    # D3 gate: no file appears in both pools
    assert not (set(prov["eval_files"]) & set(prov["train_files"])), "LEAKAGE"

    x_rx, y, h_a, p_clean, real_mask = build_mixed_eval_set(
        pool, args.n_synth, args.seed)
    print(f"mixed eval set: {x_rx.size(0)} windows "
          f"({int(real_mask.sum())} real WiFi, "
          f"{int((~real_mask).sum())} synthetic active)", flush=True)

    band = ATTACK_BANDS["cv2x_attacker"]
    results = {"experiment": "Wave-6 G1: real OTA WiFi in the attack loop",
               "provenance": prov,
               "config": {"psr_db": args.psr, "steps": args.steps,
                          "seed": args.seed, "n_synth_per_class": args.n_synth,
                          "n_real_wifi": int(pool.shape[0]),
                          "real_wifi_protocol": "no synthetic victim channel "
                          "(real channel in capture); noise at drawn SNR; "
                          "PSR ref = window energy (unit power => 2048)",
                          "location_split": "eval=uz; train=rabot+reep (disjoint capture locations)"}}

    # --- frozen model --------------------------------------------------------
    if args.stage == "finetune":
        prev = json.load(open(os.path.join(OUT, "real_wifi_attack.json")))
        results = prev
    ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                      map_location="cpu", weights_only=False)
    model = DualStreamModel(); model.load_state_dict(ckpt["model"]); model.eval()
    fe = FrontEnd(); fe.set_stats(ckpt["mag_mean"], ckpt["mag_std"])

    with torch.no_grad():
        preds = [model.forward_wave(x_rx[i:i + 256], fe).argmax(1)
                 for i in range(0, x_rx.size(0), 256)]
    preds = torch.cat(preds)
    per_class = {CLASS_NAMES[c]: round(100 * (preds[y == c] == c).float().mean().item(), 2)
                 for c in range(4) if (y == c).any()}
    real_wifi_acc = float((preds[real_mask] == 2).float().mean())
    print(f"frozen clean acc: {per_class} | real WiFi {real_wifi_acc*100:.1f}%",
          flush=True)
    if "frozen" not in results or not isinstance(results.get("frozen"), dict):
        results["frozen"] = {}
    results["frozen"].setdefault("clean_acc_per_class", per_class)
    results["frozen"].setdefault("real_wifi_acc", round(real_wifi_acc, 4))

    for setting, sband in ([] if args.stage == "finetune" else
                            [("genie", None), ("cv2x_mask", band)]):
        res, _ = attack_grid(model, fe, x_rx, h_a, y, p_clean, args.psr,
                             args.steps, sband)
        g = [res[p][0] * 100 for p in args.psr]
        mg, cg = meap_curve(args.psr, g, 20.0)
        print(f"  frozen {setting}: MEAP {mg} ({cg})", flush=True)
        results["frozen"][setting] = {
            "curve": {f"psr={p:+.0f}dB": {"cond_asr": round(res[p][0] * 100, 2),
                                          "n_eligible": res[p][1],
                                          "robust_acc": round(res[p][2] * 100, 2)}
                      for p in args.psr},
            "meap_db": mg, "censor": cg}

    # --- real-WiFi-only subset (the honest per-class view) ------------------
    rm = real_mask.bool()
    for setting, sband in ([] if args.stage == "finetune" else
                            [("genie", None), ("cv2x_mask", band)]):
        res, _ = attack_grid(model, fe, x_rx[rm], h_a[rm], y[rm],
                             p_clean[rm], args.psr, args.steps, sband)
        g = [res[p][0] * 100 for p in args.psr]
        mg, cg = meap_curve(args.psr, g, 20.0)
        print(f"  frozen {setting} [real-WiFi-only]: MEAP {mg} ({cg})", flush=True)
        results["frozen"][setting + "_real_wifi_only"] = {
            "curve": {f"psr={p:+.0f}dB": round(res[p][0] * 100, 2) for p in args.psr},
            "meap_db": mg, "censor": cg}

    # --- fine-tuned model -----------------------------------------------------
    if not args.skip_finetune and args.stage in ("all", "finetune"):
        ft_model, ft_fe, ft_acc = fine_tune_on_real(pool_tr)
        torch.save({"model": ft_model.state_dict(),
                    "mag_mean": ft_fe.mag_mean, "mag_std": ft_fe.mag_std,
                    "config": {"base": "checkpoint_dual.pt",
                               "train_windows_real": int(pool_tr.shape[0]),
                               "epochs_max": 15, "seed": 42}},
                   os.path.join(OUT, "checkpoint_dual_realft.pt"))
        with torch.no_grad():
            preds = [ft_model.forward_wave(x_rx[i:i + 256], ft_fe).argmax(1)
                     for i in range(0, x_rx.size(0), 256)]
        preds = torch.cat(preds)
        per_class = {CLASS_NAMES[c]: round(100 * (preds[y == c] == c).float().mean().item(), 2)
                     for c in range(4) if (y == c).any()}
        real_wifi_acc = float((preds[real_mask] == 2).float().mean())
        print(f"fine-tuned clean acc: {per_class} | real WiFi "
              f"{real_wifi_acc*100:.1f}%", flush=True)
        results["finetuned"] = {"val_acc": round(ft_acc, 4),
                                "clean_acc_per_class": per_class,
                                "real_wifi_acc": round(real_wifi_acc, 4)}
        for setting, sband in [("genie", None), ("cv2x_mask", band)]:
            res, _ = attack_grid(ft_model, ft_fe, x_rx, h_a, y, p_clean,
                                 args.psr, args.steps, sband)
            g = [res[p][0] * 100 for p in args.psr]
            mg, cg = meap_curve(args.psr, g, 20.0)
            print(f"  finetuned {setting}: MEAP {mg} ({cg})", flush=True)
            results["finetuned"][setting] = {
                "curve": {f"psr={p:+.0f}dB": {"cond_asr": round(res[p][0] * 100, 2),
                                              "n_eligible": res[p][1],
                                              "robust_acc": round(res[p][2] * 100, 2)}
                          for p in args.psr},
                "meap_db": mg, "censor": cg}
        # PoC for the fine-tuned model
        try:
            gg = [results["finetuned"]["genie"]["curve"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
            mm = [results["finetuned"]["cv2x_mask"]["curve"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
            poc, cn = price_of_compliance(args.psr, gg, args.psr, mm, 20.0)
            results["finetuned"]["price_of_compliance_db"] = poc
            print(f"  finetuned PoC: {poc} dB {cn}", flush=True)
        except Exception as e:
            print("  PoC skip:", e)

    # PoC frozen (full set)
    try:
        gg = [results["frozen"]["genie"]["curve"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
        mm = [results["frozen"]["cv2x_mask"]["curve"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
        poc, cn = price_of_compliance(args.psr, gg, args.psr, mm, 20.0)
        results["frozen"]["price_of_compliance_db"] = poc
        print(f"  frozen PoC: {poc} dB {cn}", flush=True)
    except Exception as e:
        print("  PoC skip:", e)

    results["elapsed_s"] = round(time.time() - t0, 1)
    fname = os.path.join(OUT, "real_wifi_attack.json")
    # merge-on-write: never drop previously computed sections (the frozen
    # grids must survive a later --stage finetune invocation; W9-A/B guard)
    merged = {}
    if os.path.exists(fname):
        try:
            merged = json.load(open(fname))
        except Exception:
            merged = {}
    for k, v in results.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict) and k != "config":
            merged[k].update(v)
        else:
            merged[k] = v
    with open(fname, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {fname} (merged)", flush=True)


if __name__ == "__main__":
    main()
