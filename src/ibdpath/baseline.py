"""Step-4 baseline: mean-pool each slide's tile embeddings -> logistic regression on
active/inactive, evaluated leave-patients-out. The "does the signal even exist?" check.

Torch-free (sklearn only) -> runs in the light env on the cached embeddings.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import embed
from .labels import load_slide_labels, patient_id


def make_classifier():
    """Standardize features then logistic regression, class-weighted for the 54/86 imbalance."""
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=5000, class_weight="balanced"))


def build_slide_dataset(encoder: str, labels=None):
    """Per-slide mean-pooled embedding + label + patient group.

    Returns X (n_slides, dim), y (n_slides,), groups (patient id per slide), slide_ids.
    Slides without a cached embedding are skipped (so it works on a partial cache too).
    """
    if labels is None:
        labels = load_slide_labels()
    X, y, groups, ids = [], [], [], []
    for sid, target in zip(labels["slide_id"], labels["slide_target"]):
        if not embed.slide_embedding_path(encoder, sid).exists():
            continue
        X.append(embed.load_slide_embedding(encoder, sid).mean(axis=0))   # mean-pool the bag
        y.append(int(target))
        groups.append(patient_id(sid))
        ids.append(sid)
    return np.asarray(X), np.asarray(y), np.asarray(groups), ids


def leave_patients_out_auroc(X, y, groups, n_splits: int = 5):
    """Out-of-fold P(active) via GroupKFold grouped by patient -> (auroc, oof_proba).

    Grouping by patient (not slide) keeps a patient's biopsies on one side of the split, so the
    AUROC is honest (no leakage). Out-of-fold = every slide scored by a model that never saw it.
    """
    gkf = GroupKFold(n_splits=n_splits)
    oof = cross_val_predict(make_classifier(), X, y, groups=groups, cv=gkf,
                            method="predict_proba")[:, 1]
    return roc_auc_score(y, oof), oof
