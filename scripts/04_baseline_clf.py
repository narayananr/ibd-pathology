#!/usr/bin/env python3
"""
Step 4 — baseline signal check (mean-pool + logistic regression).
=================================================================
Mean-pool each slide's tile embeddings into one vector, fit logistic regression on
active/inactive, and report a held-out **AUROC** using leave-patients-out CV. This is the
cheap "do the embeddings even separate the two classes?" gate — if AUROC ≳ 0.70 the signal
is there and we proceed to attention-MIL.

Light env (sklearn only), from the repo root:
    .venv/bin/python scripts/04_baseline_clf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import paths   # noqa: E402
from ibdpath.baseline import build_slide_dataset, leave_patients_out_auroc, make_classifier  # noqa: E402

ENCODER = "hoptimus0"


def main() -> None:
    X, y, groups, ids = build_slide_dataset(ENCODER)
    if len(y) == 0:
        raise SystemExit("no cached embeddings — run scripts/03_embed_tiles.py first")
    print(f"slides={len(y)} | dim={X.shape[1]} | active={int(y.sum())} "
          f"inactive={int((y == 0).sum())} | patients={len(set(groups))}")
    if len(y) < 140:
        print(f"⚠️  only {len(y)}/140 slides embedded so far — numbers are preliminary")

    auroc, oof = leave_patients_out_auroc(X, y, groups, n_splits=5)
    pred = (oof >= 0.5).astype(int)
    print(f"\nLeave-patients-out AUROC (mean-pool + logreg): {auroc:.3f}")
    print(f"Cohen's kappa @0.5: {cohen_kappa_score(y, pred):.3f}")
    print(classification_report(y, pred, target_names=["inactive", "active"], digits=3, zero_division=0))

    # per-fold AUROC, to show the estimate is stable (not one lucky split)
    fold = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = make_classifier().fit(X[tr], y[tr])
        try:
            fold.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        except ValueError:
            pass
    print("per-fold AUROC:", [round(a, 3) for a in fold])

    paths.ensure_artifacts()
    out_path = paths.ARTIFACTS_DIR / "baseline_oof_predictions.csv"
    pd.DataFrame({"slide_id": ids, "patient_id": groups, "y": y, "p_active": oof}).to_csv(out_path, index=False)
    print(f"\n✓ wrote {out_path.relative_to(REPO_ROOT)}")
    print("verdict:", "signal present ✓ (AUROC ≥ 0.70) → proceed to MIL"
          if auroc >= 0.70 else "WEAK (AUROC < 0.70) → investigate before proceeding")


if __name__ == "__main__":
    main()
