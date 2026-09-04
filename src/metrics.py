"""metrics.py — corrected adversarial evaluation metrics.

Raw ASR (as used in the ICE2CT-2026 paper) counts *all* misclassifications after an
attack, including samples the model already got wrong on clean input. At low epsilon
this is dominated by the base error rate and makes different attacks look identical
(see AUDIT.md §3). The metrics below make both views explicit.

All functions take numpy arrays or torch tensors of predicted class indices and true
labels, and are attack-agnostic.
"""

import numpy as np
import torch


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def clean_accuracy(clean_preds, labels):
    """Fraction of samples the model classifies correctly without attack."""
    clean_preds, labels = _to_numpy(clean_preds), _to_numpy(labels)
    return float((clean_preds == labels).mean())


def robust_accuracy(adv_preds, labels):
    """Fraction of samples still correct *after* the attack (1 - raw ASR)."""
    adv_preds, labels = _to_numpy(adv_preds), _to_numpy(labels)
    return float((adv_preds == labels).mean())


def raw_asr(adv_preds, labels):
    """Paper-style ASR: misclassification rate over ALL samples."""
    return 1.0 - robust_accuracy(adv_preds, labels)


def conditional_asr(clean_preds, adv_preds, labels, target_class=None):
    """Attack success restricted to samples the model got RIGHT on clean input.

    target_class: if set, only count flips *into* this class (targeted attacks);
    otherwise any wrong class counts (untargeted).
    Returns (cond_asr, n_eligible). If no sample is clean-correct, returns (nan, 0).
    """
    clean_preds = _to_numpy(clean_preds)
    adv_preds = _to_numpy(adv_preds)
    labels = _to_numpy(labels)

    eligible = clean_preds == labels
    n_elig = int(eligible.sum())
    if n_elig == 0:
        return float("nan"), 0

    flipped = eligible & (adv_preds != labels)
    if target_class is not None:
        flipped = flipped & (adv_preds == target_class)

    return float(flipped.sum()) / n_elig, n_elig


def evaluate_attack(model_forward, clean_inputs, labels, adv_inputs=None,
                    target_class=None):
    """Convenience wrapper.

    model_forward: callable(inputs) -> predictions (argmax indices)
    clean_inputs / adv_inputs: whatever model_forward consumes (e.g. tuple of streams).
    Returns a dict with clean_acc, robust_acc, raw_asr, cond_asr, n_eligible.
    """
    clean_preds = _to_numpy(model_forward(clean_inputs))
    if adv_inputs is None:
        adv_inputs = clean_inputs
    adv_preds = _to_numpy(model_forward(adv_inputs))
    c_asr, n_elig = conditional_asr(clean_preds, adv_preds, labels,
                                    target_class=target_class)
    return {
        "clean_acc": clean_accuracy(clean_preds, labels),
        "robust_acc": robust_accuracy(adv_preds, labels),
        "raw_asr": raw_asr(adv_preds, labels),
        "cond_asr": c_asr,
        "n_eligible": n_elig,
        "targeted": target_class is not None,
    }


def margin_statistics(logits, labels, adv_logits=None):
    """Logit margin (top1 - top2) and true-class margin (logit_true - max_other).

    Margin histograms before/after attack are the diagnostic that explains WHY attacks
    succeed or fail, and whether 'identical ASR across attacks' is a saturation or a
    base-error artifact (AUDIT.md §3).
    """
    logits = _to_numpy(logits).astype(np.float64)
    labels = _to_numpy(labels)
    srt = np.sort(logits, axis=1)
    top_margin = srt[:, -1] - srt[:, -2]
    idx = np.arange(len(labels))
    true_logit = logits[idx, labels]
    row = logits.copy()
    row[idx, labels] = -np.inf
    true_margin = true_logit - row.max(axis=1)
    out = {
        "top_margin_mean": float(top_margin.mean()),
        "top_margin_std": float(top_margin.std()),
        "true_margin_mean": float(true_margin.mean()),
        "true_margin_std": float(true_margin.std()),
        "true_margin_frac_negative": float((true_margin < 0).mean()),
    }
    if adv_logits is not None:
        adv_logits = _to_numpy(adv_logits).astype(np.float64)
        row = adv_logits.copy()
        row[idx, labels] = -np.inf
        adv_true_margin = adv_logits[idx, labels] - row.max(axis=1)
        out["adv_true_margin_mean"] = float(adv_true_margin.mean())
        out["adv_true_margin_frac_negative"] = float((adv_true_margin < 0).mean())
    return out
