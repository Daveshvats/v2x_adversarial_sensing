"""
run_experiments.py
Main pipeline: generate → train → attack → defend → plot → save.
"""

import os, sys, json, time, random
import numpy as np
import torch

# Reproducibility: set all random seeds
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
# 7 SNR levels from -10 dB (very noisy) to 20 dB (clean signal)
SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]


def main():
    t0 = time.time()
    print("=" * 70)
    print("  V2X ADVERSARIAL ROBUSTNESS EXPERIMENT — FULL PIPELINE")
    print("=" * 70)
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    metrics = {"snr_levels": SNR_LEVELS}

    # ── STEP 1: Generate signals ──
    print("\n" + "#" * 70)
    print("# STEP 1: Generate Synthetic V2X RF Signals")
    print("#" * 70)
    from generate_signals import generate_dataset
    generate_dataset()

    # ── STEP 2: Train model ──
    print("\n" + "#" * 70)
    print("# STEP 2: Train Spectrum Sensing CNN")
    print("#" * 70)
    from model import V2XSpectrumCNN, load_data, evaluate_by_snr, DEVICE, train_model
    # Larger batch for training (128) for gradient stability;
    # larger batch for eval (256) for throughput
    train_loader = load_data("train", batch_size=128)
    val_loader = load_data("val", batch_size=256)
    test_loader = load_data("test", batch_size=256)

    model = V2XSpectrumCNN(num_classes=5).to(DEVICE)
    model = train_model(model, train_loader, val_loader, num_epochs=15,
                        checkpoint_name="best_model.pt")

    # Evaluate clean accuracy stratified by SNR
    clean_accs, clean_preds, clean_labels, clean_snrs = evaluate_by_snr(model, test_loader, len(SNR_LEVELS))
    metrics["clean_acc"] = [float(a) for a in clean_accs]
    metrics["clean_preds"] = clean_preds.numpy().tolist()
    metrics["clean_labels"] = clean_labels.numpy().tolist()
    print(f"\n  Clean acc per SNR: {[f'{a:.3f}' for a in clean_accs]}")
    print(f"  Overall clean acc: {np.mean(clean_accs):.4f}")

    # ── STEP 3: Adversarial attacks ──
    print("\n" + "#" * 70)
    print("# STEP 3: Adversarial Attacks")
    print("#" * 70)
    from adversarial_attacks import run_fgsm, run_pgd, run_cw, gen_samples, fgsm_attack

    # FGSM sweep: eps from 0.005 (subtle) to 0.1 (very strong)
    fgsm_res = run_fgsm(model, test_loader, SNR_LEVELS, [0.005, 0.01, 0.03, 0.05, 0.1])
    metrics["fgsm_results"] = fgsm_res

    # FGSM predictions for confusion matrix (need gradients for attack generation)
    model.eval()
    f_preds, f_labels = [], []
    for X, y, snr in test_loader:
        # Generate FGSM examples at the primary epsilon (0.03)
        Xa = fgsm_attack(model, X, y, eps=0.03)
        with torch.no_grad():
            p = model(Xa).argmax(1).cpu()
        f_preds.append(p); f_labels.append(y)
    metrics["fgsm_preds"] = torch.cat(f_preds).numpy().tolist()

    # PGD sweep: eps from 0.01 to 0.05, 10 steps each
    pgd_res = run_pgd(model, test_loader, SNR_LEVELS, [0.01, 0.03, 0.05], [10])
    metrics["pgd_results"] = pgd_res

    # C&W (Auto-PGD L2) sweep: L2 epsilon values 1.0 and 2.0
    cw_res = run_cw(model, test_loader, SNR_LEVELS, [1.0, 2.0])
    metrics["cw_results"] = cw_res

    print("\n  Generating sample spectrograms...")
    cs, fs, ps = gen_samples(model, test_loader, n=5)
    metrics["clean_samples"] = cs.numpy().tolist()[:5]
    metrics["fgsm_samples"] = fs.numpy().tolist()[:5]
    metrics["pgd_samples"] = ps.numpy().tolist()[:5]

    # ── STEP 4: Defenses ──
    print("\n" + "#" * 70)
    print("# STEP 4: Defenses")
    print("#" * 70)
    from defenses import run_adv_train_sweep, run_denoiser_exp

    # Adversarial training sweep: test two mix ratios (0.1 and 0.5)
    # mix_ratio=0.1: mostly clean data with 10% adversarial (mild robustness)
    # mix_ratio=0.5: 50/50 clean/adversarial (strong robustness)
    at_res = run_adv_train_sweep(train_loader, val_loader, test_loader,
                                  SNR_LEVELS, mix_ratios=[0.1, 0.5],
                                  fgsm_eps=0.03, num_classes=5, num_epochs=5)
    metrics["adv_train_results"] = at_res

    # Denoiser defense: train a denoising autoencoder and evaluate
    dn_res = run_denoiser_exp(model, train_loader, val_loader, test_loader,
                               SNR_LEVELS, fgsm_eps=0.03)
    metrics["denoise_results"] = dn_res

    # ── STEP 5: Generate figures ──
    print("\n" + "#" * 70)
    print("# STEP 5: Generate Publication Figures")
    print("#" * 70)
    from plot_results import generate_all_figures
    generate_all_figures(metrics)

    # ── STEP 6: Save metrics ──
    print("\n" + "#" * 70)
    print("# STEP 6: Save Results")
    print("#" * 70)
    # Exclude large array data (samples, per-sample predictions) from JSON
    # to keep the metrics file manageable in size
    jm = {k: v for k, v in metrics.items()
          if not k.endswith("_samples") and not k.endswith("_preds") and not k.endswith("_labels")}
    p = os.path.join(RESULTS_DIR, "metrics.json")
    with open(p, "w") as f:
        # Custom JSON encoder to handle numpy types
        json.dump(jm, f, indent=2, default=lambda o: int(o) if isinstance(o, np.integer) else
                  float(o) if isinstance(o, np.floating) else o.tolist() if isinstance(o, np.ndarray) else o)
    print(f"  Saved: {p}")

    # ── SUMMARY ──
    el = time.time() - t0
    print("\n" + "=" * 70)
    print("  EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"  Time: {el:.1f}s ({el/60:.1f} min)")
    print(f"\n  KEY RESULTS:")
    print(f"    Clean accuracy (avg):    {np.mean(metrics['clean_acc']):.4f}")
    print(f"    Clean accuracy (20 dB):  {metrics['clean_acc'][-1]:.4f}")
    print(f"    FGSM ε=0.03 avg:         {np.mean(fgsm_res['per_snr']['0.03']):.4f}")
    print(f"    FGSM ε=0.1 avg:          {np.mean(fgsm_res['per_snr']['0.1']):.4f}")
    best_pgd = pgd_res["configurations"][-1]
    print(f"    PGD {best_pgd} avg:      {np.mean(pgd_res['per_snr'][best_pgd]):.4f}")
    print(f"    C&W L2 eps=2.0 avg:      {np.mean(cw_res['per_snr']['2.0']):.4f}")
    print(f"    AdvTrain r=0.5 robust:   {np.mean(at_res['fgsm_robust_acc']['0.5']):.4f}")
    print(f"    Denoiser robust:         {dn_res['avg_fgsm_robust']:.4f}")
    print(f"\n  OUTPUT:")
    print(f"    Figures: {os.path.join(BASE_DIR, '..', 'figures')}/")
    print(f"    Results: {p}")
    print(f"    Model:   {os.path.join(BASE_DIR, '..', 'checkpoints')}/")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        sys.exit(1)
