"""Step-6 honest validation helpers — turn saved out-of-fold predictions into defensible numbers.

No new modelling. Steps 4 (baseline) and 5 (attention-MIL) already wrote one out-of-fold P(active)
per slide; this module computes the metrics that say *how good* and *how sure*:

  - classification_metrics : confusion matrix + sensitivity/specificity/PPV/accuracy/kappa at a threshold
  - bootstrap_auroc_ci     : AUROC with a 95% CI by resampling whole PATIENTS (honest about how few we have)
  - call_agreement         : how often two heads make the same active/healed call

Torch-free (numpy + sklearn) -> runs in the light env.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score


def classification_metrics(y, p, threshold: float = 0.5) -> dict:
    """Confusion-matrix metrics at a probability cutoff. Active (=1) is the positive class.

    sensitivity = recall for active (caught / all truly active) — missing active disease is the costly error.
    specificity = recall for healed (correctly-left-alone / all truly healed).
    """
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    def _safe(num, den):
        return float(num) / den if den else float("nan")

    return dict(
        threshold=threshold,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        sensitivity=_safe(tp, tp + fn),
        specificity=_safe(tn, tn + fp),
        ppv=_safe(tp, tp + fp),
        npv=_safe(tn, tn + fn),
        accuracy=_safe(tp + tn, len(y)),
        kappa=float(cohen_kappa_score(y, pred)),
    )


def bootstrap_auroc_ci(y, p, groups, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """AUROC point estimate + (1-alpha) CI, resampling whole PATIENTS (groups) with replacement.

    Why resample patients, not slides: a patient's biopsies are not independent, so resampling slides
    would pretend we have more evidence than we do and give a too-narrow interval. Draws that end up
    single-class (no AUROC defined) are skipped. Returns (point, lo, hi, all_bootstrap_values).
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)

    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    point = float(roc_auc_score(y, p))

    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yy = y[idx]
        if np.unique(yy).size < 2:        # both classes needed to define AUROC
            continue
        boots.append(roc_auc_score(yy, p[idx]))
    boots = np.asarray(boots)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi), boots


def call_agreement(p_a, p_b, threshold: float = 0.5) -> float:
    """Fraction of items where two heads make the same call at the threshold."""
    a = (np.asarray(p_a) >= threshold).astype(int)
    b = (np.asarray(p_b) >= threshold).astype(int)
    return float((a == b).mean())
