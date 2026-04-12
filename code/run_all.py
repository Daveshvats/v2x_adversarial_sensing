#!/usr/bin/env python3
"""
Main runner: executes the complete adversarial robustness analysis pipeline.

Pipeline:
1. Generate signals (or load if already exists)
2. Train model (or load if already exists)
3. Run all adversarial attacks
4. Run all defenses
5. Generate all figures
6. Save all numerical results to results/summary.json

Designed for reproducibility: each step can be skipped if its output
already exists (use --force to override).
"""

import os
import sys
import json
import time
import numpy as np
from datetime import datetime

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(CODE_DIR), "results")
FIGURES_DIR = os.path.join(os.path.dirname(CODE_DIR), "figures")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def step1_generate_signals(force=False):
    """Step 1: Generate synthetic V2X RF signals.
    
    Skip generation if the output file already exists (unless --force).
    This avoids costly re-generation when re-running later pipeline steps.
    """
    npz_path = os.path.join(RESULTS_DIR, "v2x_signals.npz")
    if not force and os.path.exists(npz_path):
        print(f"[Step 1] Dataset already exists: {npz_path}")
        print(f"         Use --force to regenerate.")
        data = np.load(npz_path)
        return (
            data["X_train"], data["y_train"], data["snr_train"],
            data["X_val"], data["y_val"], data["snr_val"],
            data["X_test"], data["y_test"], data["snr_test"],
        )

    from generate_signals import generate_dataset
    return generate_dataset()


def step2_train_model(X_train, y_train, X_val, y_val, X_test, y_test, force=False):
    """Step 2: Train the spectrum sensing CNN.
    
    Skip training if a checkpoint already exists (unless --force).
    Returns (model, results, preds, labels) — None values when skipped.
    """
    model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    if not force and os.path.exists(model_path):
        print(f"\n[Step 2] Model already exists: {model_path}")
        print(f"         Use --force to retrain.")
        return None, None, None, None

    from model import train_model
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))
    model, results, preds, labels = train_model(
        X_train, y_train, X_val, y_val, X_test, y_test,
        input_shape=input_shape, num_classes=num_classes, epochs=5,
    )
    return model, results, preds, labels


def step3_run_attacks(X_test, y_test, snr_test, snr_levels, device, force=False):
    """Step 3: Run all adversarial attacks (FGSM, PGD, C&W).
    
    Loads the trained model checkpoint and evaluates attack effectiveness
    stratified by SNR level.
    """
    results_path = os.path.join(RESULTS_DIR, "attack_results.json")
    if not force and os.path.exists(results_path):
        print(f"\n[Step 3] Attack results already exist: {results_path}")
        with open(results_path) as f:
            return json.load(f)

    from adversarial_attacks import run_all_attacks
    from model import SpectrumCNN

    model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = SpectrumCNN(ckpt["input_shape"][0], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return run_all_attacks(model, X_test, y_test, snr_test, snr_levels, device)


def step4_run_defenses(X_train, y_train, X_val, y_val, X_test, y_test,
                       snr_test, snr_levels, input_shape, num_classes, device, force=False):
    """Step 4: Run all defenses (adversarial training, denoising).
    
    Evaluates defense effectiveness under FGSM attack, stratified by SNR.
    """
    results_path = os.path.join(RESULTS_DIR, "defense_results.json")
    if not force and os.path.exists(results_path):
        print(f"\n[Step 4] Defense results already exist: {results_path}")
        with open(results_path) as f:
            defense_results = json.load(f)
        return defense_results, None, None

    from defenses import run_defenses
    from model import SpectrumCNN

    model_path = os.path.join(RESULTS_DIR, "best_model.pt")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = SpectrumCNN(ckpt["input_shape"][0], ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return run_defenses(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        snr_test, snr_levels, input_shape, num_classes, device,
    )


def step5_generate_figures(all_results, X_test=None, y_test=None):
    """Step 5: Generate all publication-quality figures."""
    from plot_results import generate_all_figures
    return generate_all_figures(all_results, X_test, y_test)


def step6_save_summary(all_results):
    """Step 6: Save consolidated summary to results/summary.json.
    
    Aggregates key metrics from all pipeline steps into a single JSON file
    with a timestamp for provenance tracking.
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "dataset": {},
        "model": {},
        "attacks": {},
        "defenses": {},
    }

    # Dataset info
    meta_path = os.path.join(RESULTS_DIR, "dataset_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            summary["dataset"] = json.load(f)

    # Model results
    summary["model"]["clean_accuracy"] = all_results["attack"]["clean_accuracy"]
    summary["model"]["clean_snr_accuracy"] = all_results["attack"]["clean_snr_accuracy"]

    # Attack summaries: extract accuracy, ASR, and mean L2 perturbation
    for attack_type in ["fgsm", "pgd", "cw"]:
        if attack_type in all_results["attack"]:
            summary["attacks"][attack_type] = {}
            for key, data in all_results["attack"][attack_type].items():
                summary["attacks"][attack_type][key] = {
                    "accuracy": data["accuracy"],
                    "asr": data.get("asr", "N/A"),
                    "mean_l2": data.get("mean_l2", "N/A"),
                }

    # Defense summaries: extract clean accuracy and F1 score
    for def_type in ["adversarial_training", "input_denoising"]:
        if def_type in all_results["defense"]:
            d = all_results["defense"][def_type]
            summary["defenses"][def_type] = {
                "clean_accuracy": d.get("test_accuracy", d.get("clean_accuracy", "N/A")),
                "f1_macro": d.get("f1_macro", "N/A"),
            }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n[Step 6] Summary saved to: {summary_path}")
    return summary


def main():
    """Main pipeline execution.
    
    The --force flag regenerates all outputs from scratch;
    otherwise, existing outputs are reused for efficiency.
    """
    force = "--force" in sys.argv

    print("=" * 70)
    print("  V2X Adversarial Robustness Analysis - Complete Pipeline")
    print("=" * 70)
    t_start = time.time()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: Generate signals
    X_train, y_train, snr_train, X_val, y_val, snr_val, X_test, y_test, snr_test = \
        step1_generate_signals(force=force)

    snr_levels = np.load(os.path.join(RESULTS_DIR, "v2x_signals.npz"))["snr_levels"].tolist()
    input_shape = X_train.shape[1:]
    num_classes = len(np.unique(y_train))

    # Step 2: Train model
    model, train_results, train_preds, train_labels = step2_train_model(
        X_train, y_train, X_val, y_val, X_test, y_test, force=force
    )

    # Step 3: Run attacks
    attack_results = step3_run_attacks(X_test, y_test, snr_test, snr_levels, device, force=force)

    # Step 4: Run defenses
    defense_results, adv_model, denoiser = step4_run_defenses(
        X_train, y_train, X_val, y_val, X_test, y_test,
        snr_test, snr_levels, input_shape, num_classes, device, force=force
    )

    # Step 5: Generate figures
    all_results = {"attack": attack_results, "defense": defense_results}
    step5_generate_figures(all_results, X_test, y_test)

    # Step 6: Save summary
    summary = step6_save_summary(all_results)

    t_total = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  PIPELINE COMPLETE in {t_total:.1f}s ({t_total/60:.1f} min)")
    print("=" * 70)
    print(f"\nKey Results Summary:")
    print(f"  Clean Accuracy:       {summary['model']['clean_accuracy']:.4f}")

    # Report the worst-performing epsilon for each attack type
    # Best and worst FGSM
    fgsm_accs = [(k, v["accuracy"]) for k, v in summary["attacks"].get("fgsm", {}).items()]
    if fgsm_accs:
        best_fgsm = min(fgsm_accs, key=lambda x: x[1])
        print(f"  FGSM worst (ε={best_fgsm[0]}):  {best_fgsm[1]:.4f}")

    # Best and worst PGD
    pgd_accs = [(k, v["accuracy"]) for k, v in summary["attacks"].get("pgd", {}).items()]
    if pgd_accs:
        worst_pgd = min(pgd_accs, key=lambda x: x[1])
        print(f"  PGD worst ({worst_pgd[0]}):  {worst_pgd[1]:.4f}")

    # C&W
    cw_accs = [(k, v["accuracy"]) for k, v in summary["attacks"].get("cw", {}).items()]
    if cw_accs:
        worst_cw = min(cw_accs, key=lambda x: x[1])
        print(f"  C&W worst ({worst_cw[0]}):    {worst_cw[1]:.4f}")

    # Defenses
    for dname, dinfo in summary["defenses"].items():
        print(f"  {dname:25s}: {dinfo['clean_accuracy']}")

    print(f"\nFiles generated:")
    print(f"  Results:  {RESULTS_DIR}/")
    print(f"  Figures:  {FIGURES_DIR}/")

    # List generated files with sizes
    for d in [RESULTS_DIR, FIGURES_DIR]:
        print(f"\n  {d}:")
        for f in sorted(os.listdir(d)):
            fpath = os.path.join(d, f)
            size = os.path.getsize(fpath) / (1024 * 1024)
            print(f"    {f:40s} ({size:.2f} MB)")

    return summary


if __name__ == "__main__":
    main()
