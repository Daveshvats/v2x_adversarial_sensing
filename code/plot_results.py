"""
plot_results.py
Publication-quality figures (300 DPI).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Publication style: white grid background with colorblind-friendly palette
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("colorblind")

# Font sizes for publication figures
FT, FL, FK, FLG = 14, 12, 11, 10
DPI = 300  # Publication-quality resolution
MODULATIONS = ["BPSK", "QPSK", "16-QAM", "64-QAM", "Noise"]
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    """Save figure at 300 DPI with tight bounding box and white background."""
    fig.savefig(os.path.join(FIG_DIR, name), dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


def fig1_clean(snr, acc):
    """Figure 1: Clean classification accuracy vs. SNR.
    
    This is the baseline performance figure showing how classification
    accuracy improves with signal quality. The annotated values help
    readers quickly assess performance at each SNR level.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(snr, [a*100 for a in acc], "o-", lw=2, ms=8, color="#2196F3", label="Clean")
    for x, y in zip(snr, [a*100 for a in acc]):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("SNR (dB)", fontsize=FL); ax.set_ylabel("Accuracy (%)", fontsize=FL)
    ax.set_title("Clean Classification Accuracy vs. SNR", fontsize=FT, fontweight="bold")
    ax.set_xticks(snr); ax.set_ylim(10, 105); ax.legend(fontsize=FLG)
    fig.tight_layout(); _save(fig, "fig1_clean_accuracy_vs_snr.png")


def fig2_fgsm(snr, clean, res):
    """Figure 2: Accuracy degradation under FGSM attack at various epsilon values.
    
    Shows how increasing perturbation magnitude (epsilon) degrades accuracy.
    The clean accuracy line provides reference for comparison.
    Different marker styles distinguish epsilon values in print (accessibility).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # Different marker styles for each epsilon (accessible in grayscale)
    mk = ["o", "s", "^", "D", "v"]
    ax.plot(snr, [a*100 for a in clean], "k-", lw=2.5, label="Clean", zorder=10)
    for i, e in enumerate(res["eps_list"]):
        a = res["per_snr"][str(e)]
        ax.plot(snr, [x*100 for x in a], f"{mk[i]}--", lw=1.8, ms=6, label=f"FGSM ε={e}")
    ax.set_xlabel("SNR (dB)", fontsize=FL); ax.set_ylabel("Accuracy (%)", fontsize=FL)
    ax.set_title("Accuracy Under FGSM Attack", fontsize=FT, fontweight="bold")
    ax.set_xticks(snr); ax.set_ylim(0, 105); ax.legend(fontsize=FLG, ncol=2)
    fig.tight_layout(); _save(fig, "fig2_fgsm_attack.png")


def fig3_pgd(snr, clean, res):
    """Figure 3: Accuracy degradation under PGD attack (multi-step).
    
    PGD configurations are parameterized by (epsilon, steps). Each configuration
    produces a separate line showing the SNR-dependent accuracy curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cm = plt.cm.tab10  # Use tab10 colormap for distinct colors
    ax.plot(snr, [a*100 for a in clean], "k-", lw=2.5, label="Clean", zorder=10)
    for i, cfg in enumerate(res["configurations"]):
        a = res["per_snr"][cfg]
        ax.plot(snr, [x*100 for x in a], "o-", lw=1.8, ms=5, color=cm(i/max(len(res["configurations"]),1)), label=f"PGD {cfg}")
    ax.set_xlabel("SNR (dB)", fontsize=FL); ax.set_ylabel("Accuracy (%)", fontsize=FL)
    ax.set_title("Accuracy Under PGD Attack", fontsize=FT, fontweight="bold")
    ax.set_xticks(snr); ax.set_ylim(0, 105); ax.legend(fontsize=FLG, ncol=2)
    fig.tight_layout(); _save(fig, "fig3_pgd_attack.png")


def fig4_confusion(clean_l, clean_p, atk_l, atk_p, def_l=None, def_p=None):
    """Figure 4: Confusion matrices — Clean vs. Under Attack vs. Defended.
    
    Each confusion matrix shows the per-class classification performance.
    Values are normalized to percentages (row-wise) for easy comparison
    across classes of different sizes.
    
    The 1e-9 term prevents division by zero for empty rows.
    """
    n = 3 if def_l is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5))
    titles = ["Clean", "Under Attack (FGSM)", "Defended"]
    pairs = [(clean_l, clean_p), (atk_l, atk_p)]
    if def_l is not None:
        pairs.append((def_l, def_p))
    for ax, (l, p), t in zip(axes, pairs, titles):
        # Compute confusion matrix with integer labels [0,4]
        cm_ = confusion_matrix(l, p, labels=list(range(5)))
        # Row-normalize to percentages for class-conditional accuracy
        cm_pct = cm_.astype(float) / (cm_.sum(axis=1, keepdims=True) + 1e-9) * 100
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                    xticklabels=MODULATIONS, yticklabels=MODULATIONS, vmin=0, vmax=100)
        ax.set_title(t, fontsize=FT, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=FL); ax.set_ylabel("True", fontsize=FL)
        ax.tick_params(labelsize=FK)
    fig.suptitle("Confusion Matrices", fontsize=FT+2, fontweight="bold", y=1.02)
    fig.tight_layout(); _save(fig, "fig4_confusion_matrices.png")


def fig5_defense(snr, clean, at_res, dn_res=None):
    """Figure 5: Defense effectiveness comparison under FGSM attack.
    
    Compares clean accuracy, adversarial training at various mix ratios,
    and the denoiser defense. All evaluated under FGSM ε=0.03 attack.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cm = plt.cm.Set2
    ax.plot(snr, [a*100 for a in clean], "k-", lw=2.5, label="Clean (no defence)", zorder=10)
    for i, r in enumerate(at_res["mix_ratios"]):
        a = at_res["fgsm_robust_acc"][str(r)]
        ax.plot(snr, [x*100 for x in a], "s-", lw=1.8, ms=6,
                color=cm(i/max(len(at_res["mix_ratios"]),1)), label=f"AdvTrain r={r}")
    if dn_res:
        ax.plot(snr, [x*100 for x in dn_res["fgsm_robust_acc"]], "D-", lw=1.8, ms=6,
                color="#00BCD4", label="Denoiser")
    ax.set_xlabel("SNR (dB)", fontsize=FL); ax.set_ylabel("Accuracy (%)", fontsize=FL)
    ax.set_title("Defence Effectiveness Under FGSM", fontsize=FT, fontweight="bold")
    ax.set_xticks(snr); ax.set_ylim(0, 105); ax.legend(fontsize=FLG, ncol=2)
    fig.tight_layout(); _save(fig, "fig5_adversarial_training.png")


def fig6_spectrograms(clean, fgsm, pgd, labels):
    """Figure 6: Side-by-side sample spectrograms (clean vs. FGSM vs. PGD).
    
    Visual comparison of 5 randomly selected spectrograms showing:
      - Row 1: Clean spectrograms (true labels as titles)
      - Row 2: FGSM-perturbed spectrograms (ε=0.03)
      - Row 3: PGD-perturbed spectrograms (ε=0.05)
    
    The vmin/vmax range of [-3, 3] corresponds to the z-score normalized
    spectrogram values after clipping.
    """
    n = min(5, clean.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(3*n, 9))
    titles = ["Clean", "FGSM (ε=0.03)", "PGD (ε=0.05)"]
    sets = [clean, fgsm, pgd]
    for row, (S, t) in enumerate(zip(sets, titles)):
        for col in range(n):
            ax = axes[row, col]
            # Display as grayscale image; squeeze channel dim
            ax.imshow(S[col, 0].numpy(), cmap="gray", aspect="auto", vmin=-3, vmax=3)
            ax.set_xticks([]); ax.set_yticks([])  # Remove axis ticks for clean look
            if row == 0:
                # Show true modulation type as column title
                li = labels[col].item() if hasattr(labels[col], 'item') else int(labels[col])
                ax.set_title(MODULATIONS[li], fontsize=FL, fontweight="bold")
            if col == 0:
                ax.set_ylabel(t, fontsize=FL, fontweight="bold", rotation=90, labelpad=15)
    fig.suptitle("Sample Spectrograms: Clean vs. Adversarial", fontsize=FT+2, fontweight="bold", y=1.01)
    fig.tight_layout(); _save(fig, "fig6_sample_spectrograms.png")


def generate_all_figures(m):
    """Generate all publication figures from the experiment metrics dict."""
    print("\n" + "=" * 50)
    print("GENERATING PUBLICATION FIGURES")
    print("=" * 50)
    snr = m["snr_levels"]
    fig1_clean(snr, m["clean_acc"])
    fig2_fgsm(snr, m["clean_acc"], m["fgsm_results"])
    fig3_pgd(snr, m["clean_acc"], m["pgd_results"])

    # Confusion matrices: compare clean vs. FGSM-attacked
    cl = m["clean_labels"]; cp = m["clean_preds"]
    al = m["clean_labels"]; ap = m["fgsm_preds"]
    fig4_confusion(cl, cp, al, ap)

    fig5_defense(snr, m["clean_acc"], m["adv_train_results"], m.get("denoise_results"))

    if "clean_samples" in m:
        cs = torch.tensor(m["clean_samples"])
        fs = torch.tensor(m["fgsm_samples"])
        ps = torch.tensor(m["pgd_samples"])
        fig6_spectrograms(cs, fs, ps, cl[:5])

    print(f"\nAll figures saved to {FIG_DIR}/")
