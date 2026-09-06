#!/usr/bin/env python3
"""run_victim2.py — train the second victim (ResNet, G2) on the SAME task,
SAME data recipe, SAME seed as the canonical dual model (run_train.py seed 42).

Saves results/checkpoint_resnet.pt + results/victim2_train_report.json.
Then (via --with-attacks) runs the canonical white-box attack grid on it and
the TRANSFER grids both directions (surrogate-model threat, M7 datapoint):

  A2A  white-box : delta crafted on ResNet, evaluated on ResNet
  D2R  transfer  : delta crafted on dual-CNN (canonical victim), applied to
                   ResNet  (surrogate = dual)
  R2D  transfer  : delta crafted on ResNet, applied to dual-CNN
                   (surrogate = resnet)

Transfer = the partial-knowledge attacker: knows the task and has a surrogate
model, but NOT the deployed victim's weights. Attacker channel h_a still
perfect (worst-case on the channel axis — disclosed).
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
from victim_resnet import ResNetVictim
from run_train import build_dataset
from receiver import tf_cutmix, smoothed_ce


def train_model_resumable(model, frontend, tr_w, tr_y, val_w, val_y, epochs,
                          resume_path=None, batch_size=64, lr=5e-4,
                          cutmix_prob=0.3, gauss_prob=0.3, gauss_std=0.02,
                          start_epoch=0, best_acc=0.0, best_state=None):
    """Same recipe as run_train.train_model + per-epoch resume checkpoint
    (sandbox-safe: survives process kills; identical hyperparameters)."""
    import pickle
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    B = tr_w.size(0)
    patience, no_improve = 15, 0
    if best_state is not None:
        model.load_state_dict(best_state)
    for ep in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(B)
        for i in range(0, B, batch_size):
            idx = perm[i:i + batch_size]
            wb, yb = tr_w[idx], tr_y[idx]
            if np.random.rand() < gauss_prob:
                wb = wb + gauss_std * torch.complex(
                    torch.randn_like(wb.real), torch.randn_like(wb.imag))
            mag, ifr = frontend(wb)
            if np.random.rand() < cutmix_prob:
                mag, ifr, y_a, y_b, lam = tf_cutmix(mag, ifr, yb)
            else:
                y_a, y_b, lam = yb, yb, 1.0
            logits = model(mag, ifr)
            loss = smoothed_ce(logits, y_a, y_b, lam)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, val_w.size(0), 256):
                preds.append(model.forward_wave(val_w[i:i + 256], frontend).argmax(1))
            acc = (torch.cat(preds) == val_y).float().mean().item()
        if acc > best_acc + 1e-4:
            best_acc, no_improve = acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
        print(f"    epoch {ep+1:3d}: val acc {acc*100:5.2f}%  best "
              f"{best_acc*100:5.2f}%", flush=True)
        if resume_path:
            with open(resume_path, "wb") as f:
                pickle.dump({"epoch": ep + 1, "best_acc": best_acc,
                             "best_state": best_state}, f, protocol=4)
        if no_improve >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc
from run_attack import build_eval_set
from attack_mask import waveform_pgd, cond_asr, meap_curve, price_of_compliance

OUT = os.path.join(ROOT, "results")
DEFAULT_PSR = [-45.0, -40.0, -35.0, -30.0, -25.0, -20.0, -15.0,
               -10.0, -5.0, 0.0, 5.0, 10.0]


def load_victim(name, ckpt_name):
    if name == "dual":
        ckpt = torch.load(os.path.join(OUT, "checkpoint_dual.pt"),
                          map_location="cpu", weights_only=False)
        m = DualStreamModel()
    elif name == "resnet":
        ckpt = torch.load(os.path.join(OUT, "checkpoint_resnet.pt"),
                          map_location="cpu", weights_only=False)
        m = ResNetVictim()
    else:
        raise ValueError(name)
    m.load_state_dict(ckpt["model"]); m.eval()
    fe = FrontEnd(); fe.set_stats(ckpt["mag_mean"], ckpt["mag_std"])
    return m, fe


def craft_and_eval(surrogate, victim, fe_s, fe_v, x_rx, h_a, y, p_budget,
                   steps, band):
    """Craft delta on surrogate (white-box), evaluate on victim."""
    out = waveform_pgd(surrogate, fe_s, x_rx, h_a, y, p_budget,
                       steps=steps, band=band, return_delta=True)
    delta = out["delta"]
    with torch.no_grad():
        from attack_mask import cconv
        r_adv = x_rx + cconv(delta, h_a)
        preds = []
        for i in range(0, r_adv.size(0), 256):
            preds.append(victim.forward_wave(r_adv[i:i + 256], fe_v).argmax(1))
    return torch.cat(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--n-per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--psr", nargs="+", type=float, default=DEFAULT_PSR)
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--attacks-only", action="store_true")
    ap.add_argument("--max-seconds", type=float, default=1e9,
                    help="soft time budget; exits gracefully and saves state")
    ap.add_argument("--grids", default="all",
                    choices=["all", "wb", "transfers"],
                    help="wb = white-box resnet + canonical dual; "
                         "transfers = D2R + R2D")
    args = ap.parse_args()
    t0 = time.time()

    if not args.attacks_only:
        np.random.seed(args.seed); torch.manual_seed(args.seed)
        print("[1] building dataset (same as run_train seed 42)...", flush=True)
        waves, labels, scen = build_dataset(args.seed, args.n_per_class)
        n_val = int(0.2 * waves.size(0))
        val_w, val_y = waves[:n_val], labels[:n_val]
        tr_w, tr_y = waves[n_val:], labels[n_val:]
        frontend = FrontEnd()
        with torch.no_grad():
            mags = []
            for i in range(0, tr_w.size(0), 512):
                mag, _ = frontend(tr_w[i:i + 512])
                mags.append(mag)
            allmag = torch.cat(mags)
            frontend.set_stats(allmag.mean(), allmag.std())
        print("[2] training ResNet victim (resumable)...", flush=True)
        model = ResNetVictim()
        resume_path = os.path.join(OUT, "victim2_resume.pkl")
        start_ep, best_acc, best_state = 0, 0.0, None
        if os.path.exists(resume_path):
            import pickle
            with open(resume_path, "rb") as f:
                st = pickle.load(f)
            start_ep, best_acc = st["epoch"], st["best_acc"]
            if start_ep < args.epochs:
                best_state = st["best_state"]
                model.load_state_dict(best_state)
                print(f"    resuming from epoch {start_ep} "
                      f"(best {best_acc*100:.2f}%)", flush=True)
            else:
                start_ep = args.epochs   # done already
        acc = best_acc
        if start_ep < args.epochs:
            acc = train_model_resumable(model, frontend, tr_w, tr_y, val_w,
                                        val_y, args.epochs,
                                        resume_path=resume_path,
                                        start_epoch=start_ep,
                                        best_acc=best_acc,
                                        best_state=best_state)
        print(f"    best val acc: {acc*100:.2f}%", flush=True)
        torch.save({"model": model.state_dict(),
                    "mag_mean": frontend.mag_mean,
                    "mag_std": frontend.mag_std,
                    "config": {"arch": "ResNetVictim-2ch-32w",
                               "seed": args.seed, "n_train": tr_w.size(0),
                               "epochs": args.epochs,
                               "recipe": "same as run_train.py v3 recipe"}},
                   os.path.join(OUT, "checkpoint_resnet.pt"))
        report = {"seed": args.seed, "arch": "ResNetVictim (2-ch early fusion)",
                  "params": sum(p.numel() for p in model.parameters()),
                  "val_acc": round(acc, 4),
                  "elapsed_s": round(time.time() - t0, 1)}
        with open(os.path.join(OUT, "victim2_train_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print("DONE train:", report, flush=True)
        if args.train_only:
            return

    # ---- attack phase -------------------------------------------------------
    dual, fe_d = load_victim("dual", "checkpoint_dual.pt")
    resnet, fe_r = load_victim("resnet", "checkpoint_resnet.pt")

    x_rx, y, h_a, p_clean = build_eval_set("urban", args.n_eval, 7)
    print(f"[3] eval set: {x_rx.size(0)} windows", flush=True)

    band = ATTACK_BANDS["cv2x_attacker"]
    results = {"experiment": "Wave-6 G2: second victim + transfer attacks (M7)",
               "config": {"steps": args.steps, "seed": 7,
                          "n_eval_per_class": args.n_eval,
                          "psr_db": args.psr,
                          "surrogate_disclosure": "transfer attacks use a "
                          "perfect attacker channel (worst case); only the "
                          "victim MODEL is unknown to the surrogate"}}
    victim_list = ([] if args.grids == "transfers" else
                   [("dual", dual, fe_d), ("resnet", resnet, fe_r)])
    for vname, victim, fe_v in victim_list:
        with torch.no_grad():
            cp = [victim.forward_wave(x_rx[i:i + 256], fe_v).argmax(1)
                  for i in range(0, x_rx.size(0), 256)]
            cp = torch.cat(cp)
        clean_acc = float((cp == y).float().mean())
        print(f"  {vname} clean acc: {clean_acc*100:.2f}%", flush=True)

    STATE = os.path.join(OUT, "victim2_grid_state.json")
    state = {}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:
            state = {}

    t_limit = getattr(args, "max_seconds", 1e9)

    def save_state():
        with open(STATE, "w") as f:
            json.dump(state, f)

    def batch_size_for(name):
        return 64 if name == "resnet" else 300

    def grid(surrogate_name, surrogate, fe_s, victim_name, victim, fe_v):
        key = f"{surrogate_name}2{victim_name}"
        B = batch_size_for(surrogate_name)
        with torch.no_grad():
            cpc = [victim.forward_wave(x_rx[i:i + 256], fe_v).argmax(1)
                   for i in range(0, x_rx.size(0), 256)]
        cpc = torch.cat(cpc)
        if key not in state:
            state[key] = {"clean_acc_active": round(float((cpc == y).float().mean()), 4)}
        for setting, sband in [("genie", None), ("cv2x_mask", band)]:
            if setting not in state[key]:
                state[key][setting] = {}
            sk = state[key][setting]
            for psr in args.psr:
                pk = f"psr={psr:+.0f}dB"
                if pk in sk:
                    continue
                if time.time() - t0 > t_limit:
                    save_state()
                    print("  [time budget reached - state saved, re-run to continue]", flush=True)
                    return state[key]
                p_budget = p_clean * (10 ** (psr / 10.0))
                if surrogate_name == victim_name:
                    preds = []
                    for i in range(0, x_rx.size(0), B):
                        out = waveform_pgd(victim, fe_v, x_rx[i:i + B],
                                           h_a[i:i + B], y[i:i + B],
                                           p_budget[i:i + B], steps=args.steps,
                                           band=sband)
                        preds.append(out["preds"])
                    adv = torch.cat(preds)
                else:
                    preds = []
                    for i in range(0, x_rx.size(0), B):
                        preds.append(craft_and_eval(
                            surrogate, victim, fe_s, fe_v, x_rx[i:i + B],
                            h_a[i:i + B], y[i:i + B], p_budget[i:i + B],
                            args.steps, sband))
                    adv = torch.cat(preds)
                c, n = cond_asr(cpc, adv, y)
                rob = float((adv == y).float().mean())
                sk[pk] = {"cond_asr": round(100 * c, 2), "n_eligible": n,
                          "robust_acc": round(100 * rob, 2)}
                print(f"    {key}/{setting} PSR {psr:+5.0f}: "
                      f"cond-ASR {100*c:5.1f}%", flush=True)
                save_state()
        # summaries once both settings complete
        if all(f"psr={p:+.0f}dB" in state[key]["genie"] for p in args.psr) and \
           all(f"psr={p:+.0f}dB" in state[key]["cv2x_mask"] for p in args.psr):
            for setting in ("genie", "cv2x_mask"):
                g = [state[key][setting][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
                mg, cg = meap_curve(args.psr, g, 20.0)
                state[key][setting]["meap_db"] = mg
                state[key][setting]["censor"] = cg
            gg = [state[key]["genie"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
            mm = [state[key]["cv2x_mask"][f"psr={p:+.0f}dB"]["cond_asr"] for p in args.psr]
            poc, cn = price_of_compliance(args.psr, gg, args.psr, mm, 20.0)
            state[key]["price_of_compliance_db"] = poc
            print(f"    {key}: MEAP genie {state[key]['genie']['meap_db']} "
                  f"mask {state[key]['cv2x_mask']['meap_db']} PoC {poc}",
                  flush=True)
            save_state()
        return state[key]

    if args.grids in ("all", "wb"):
        grid("resnet", resnet, fe_r, "resnet", resnet, fe_r)   # white-box A2A
        grid("dual", dual, fe_d, "dual", dual, fe_d)           # canonical ref
    if args.grids in ("all", "transfers"):
        grid("dual", dual, fe_d, "resnet", resnet, fe_r)       # D2R transfer
        grid("resnet", resnet, fe_r, "dual", dual, fe_d)       # R2D transfer
    results.update(state)
    results["config"]["state_file"] = "victim2_grid_state.json"

    results["elapsed_s"] = round(time.time() - t0, 1)
    fname = os.path.join(OUT, "victim2_transfer.json")
    with open(fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {fname}", flush=True)


if __name__ == "__main__":
    main()
