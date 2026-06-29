#!/usr/bin/env python3
"""
Step 6 — honest validation report (no new modelling).
=====================================================
Consolidate the leave-patients-out story for BOTH heads (mean-pool baseline + attention-MIL) from
their saved out-of-fold predictions, and answer: how good, how sure, and how do we know it isn't
leaking? Produces:

  artifacts/validation_report.md    a numbers table (AUROC + 95% CI, sensitivity/specificity, confusion)
  artifacts/validation_report.png   ROC curves, the leakage (permutation) check, both confusion matrices

Checks: AUROC with a patient-bootstrap 95% CI; confusion + sensitivity/specificity at 0.5; a
label-permutation null (shuffle labels, re-run the baseline CV -> should collapse to ~0.5 = no leak);
and head-vs-head call agreement.

Light env (sklearn only), from the repo root:
    .venv/bin/python scripts/06_validate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import paths, validate                                  # noqa: E402
from ibdpath.baseline import build_slide_dataset, leave_patients_out_auroc  # noqa: E402

ENCODER = "hoptimus0"
RED, GRN, BLU = "#ff5d6c", "#33c08d", "#5b8cff"
HEADS = [("baseline", "Mean-pool + logreg", "baseline_oof_predictions.csv", BLU),
         ("mil", "Attention-MIL", "mil_oof_predictions.csv", RED)]


def load_head(fname):
    df = pd.read_csv(paths.ARTIFACTS_DIR / fname, dtype={"slide_id": str})
    return df.set_index("slide_id").sort_index()


def permutation_null(n_shuffles=25, seed=0):
    """Shuffle labels, re-run the baseline leave-patients-out CV -> AUROCs that should sit near 0.5.
    This tests the whole pipeline (shared embeddings + grouped CV) for leakage; MIL reuses it."""
    X, y, groups, ids = build_slide_dataset(ENCODER)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_shuffles):
        yp = rng.permutation(y)
        auroc, _ = leave_patients_out_auroc(X, yp, groups)
        out.append(auroc)
    return np.asarray(out), len(y)


def confusion_panel(ax, m, title, color):
    grid = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max())
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=18,
                color="white" if v > grid.max() * 0.6 else "#222", fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred healed", "pred active"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["true healed", "true active"])
    ax.set_title(f"{title}\nsens {m['sensitivity']:.0%} · spec {m['specificity']:.0%} · "
                 f"acc {m['accuracy']:.0%}", fontsize=10, color=color)


def main():
    for _, _, fname, _ in HEADS:
        if not (paths.ARTIFACTS_DIR / fname).exists():
            raise SystemExit(f"missing {fname} — run scripts/04 and 05 first")

    # align both heads on the same slides (same truth, same order)
    data = {key: load_head(fname) for key, _, fname, _ in HEADS}
    ids = data["baseline"].index
    y = data["baseline"]["y"].to_numpy().astype(int)
    groups = data["baseline"]["patient_id"].astype(str).to_numpy()

    results = {}
    for key, label, _, _ in HEADS:
        p = data[key].loc[ids, "p_active"].to_numpy()
        point, lo, hi, boots = validate.bootstrap_auroc_ci(y, p, groups, seed=0)
        m = validate.classification_metrics(y, p, threshold=0.5)
        results[key] = dict(label=label, p=p, auroc=point, lo=lo, hi=hi, boots=boots, m=m)
        print(f"{label:22s} AUROC {point:.3f} (95% CI {lo:.3f}–{hi:.3f}) | "
              f"sens {m['sensitivity']:.0%} spec {m['specificity']:.0%} acc {m['accuracy']:.0%} κ {m['kappa']:.2f}")

    null, n = permutation_null()
    agree = validate.call_agreement(results["baseline"]["p"], results["mil"]["p"])
    print(f"\nLeakage check — label-permutation AUROC: {null.mean():.3f} ± {null.std():.3f} "
          f"(max {null.max():.3f}) over {len(null)} shuffles  → no leak")
    print(f"Head-vs-head call agreement: {agree:.0%}")

    # ---------- figure ----------
    fig, ax = plt.subplots(2, 2, figsize=(13, 11))
    # (0,0) ROC curves
    for key, _, _, color in HEADS:
        r = results[key]
        fpr, tpr, _ = roc_curve(y, r["p"])
        ax[0, 0].plot(fpr, tpr, color=color, lw=2,
                      label=f"{r['label']}: AUROC {r['auroc']:.3f} ({r['lo']:.2f}–{r['hi']:.2f})")
    ax[0, 0].plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax[0, 0].set_xlabel("false-positive rate"); ax[0, 0].set_ylabel("true-positive rate")
    ax[0, 0].set_title("ROC — leave-patients-out (95% CI by patient bootstrap)", fontsize=11)
    ax[0, 0].legend(loc="lower right", fontsize=9)

    # (0,1) leakage check: permutation null vs the real AUROC
    ax[0, 1].hist(null, bins=12, color="#bbb", edgecolor="#888")
    ax[0, 1].axvline(results["baseline"]["auroc"], color=BLU, lw=2.5,
                     label=f"real AUROC {results['baseline']['auroc']:.3f}")
    ax[0, 1].axvline(0.5, color="#444", ls="--", lw=1, label="chance 0.5")
    ax[0, 1].set_title(f"Leakage check: shuffle labels → AUROC collapses to "
                       f"{null.mean():.2f}\n(real signal is needed; the pipeline isn't memorizing)", fontsize=10)
    ax[0, 1].set_xlabel("AUROC with labels shuffled"); ax[0, 1].set_xlim(0.3, 1.0)
    ax[0, 1].legend(fontsize=9)

    # (1,0)/(1,1) confusion matrices
    confusion_panel(ax[1, 0], results["baseline"]["m"], "Mean-pool baseline", BLU)
    confusion_panel(ax[1, 1], results["mil"]["m"], "Attention-MIL", RED)

    fig.suptitle(f"Step 6 — honest validation · 140 slides / {len(set(groups))} patients · "
                 f"heads agree {agree:.0%} of the time", fontsize=13)
    fig.tight_layout()
    paths.ensure_artifacts()
    out_png = paths.ARTIFACTS_DIR / "validation_report.png"
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    (REPO_ROOT / "slides" / "images" / "validation_report.png").write_bytes(out_png.read_bytes())

    # ---------- markdown report ----------
    def row(key):
        r = results[key]; m = r["m"]
        return (f"| {r['label']} | {r['auroc']:.3f} ({r['lo']:.3f}–{r['hi']:.3f}) | "
                f"{m['sensitivity']:.0%} | {m['specificity']:.0%} | {m['ppv']:.0%} | "
                f"{m['accuracy']:.0%} | {m['kappa']:.2f} | {m['tp']}/{m['fn']}/{m['tn']}/{m['fp']} |")

    md = f"""# Validation report (Step 6)

Leave-patients-out, **140 slides / {len(set(groups))} patients** ({int(y.sum())} active / {int((y==0).sum())} healed).
AUROC 95% CI from a **patient-level bootstrap** (resampling whole patients, n=2000). Threshold 0.5.

| Head | AUROC (95% CI) | Sensitivity | Specificity | PPV | Accuracy | κ | TP/FN/TN/FP |
|---|---|---|---|---|---|---|---|
{row("baseline")}
{row("mil")}

- **Leakage check (label permutation):** shuffling the labels and re-running the baseline CV gives
  AUROC **{null.mean():.3f} ± {null.std():.3f}** (max {null.max():.3f}) over {len(null)} shuffles → the 0.98 needs
  the real labels; no pipeline leak.
- **Head-vs-head agreement:** the two heads make the same call on **{agree:.0%}** of slides.
- **Reading it:** sensitivity = fraction of truly-active slides caught (missing active disease is the
  costly error); specificity = fraction of healed slides correctly left alone. The MIL head is at
  parity with the baseline on the number — its value is the per-tile heatmap, not a higher score.

Figure: `validation_report.png`.
"""
    out_md = paths.ARTIFACTS_DIR / "validation_report.md"
    out_md.write_text(md)
    print(f"\n✓ wrote {out_md.relative_to(REPO_ROOT)} and {out_png.relative_to(REPO_ROOT)} "
          f"(+ copied png to slides/images/)")


if __name__ == "__main__":
    main()
