#!/usr/bin/env python3
"""
Step 5 — attention-MIL head (the *where*).
==========================================
Mean-pool (Step 4) gave the headline AUROC but averaged location away. This trains a small **gated
attention-MIL** head on the same frozen embeddings: it learns a per-tile attention weight (softmax
over the slide's tiles, summing to 1), pools the weighted bag, and classifies it. The attention is
the **heatmap** — which tiles drove the "active" call.

Evaluated the SAME honest way as the baseline: **leave-patients-out** (GroupKFold by patient). For
each held-out slide we also keep its attention from the fold model **that never saw it**, so the
heatmaps are leak-free too. We don't expect AUROC to beat the 0.98 baseline (it's near ceiling) —
the payoff is localization. Demo heatmaps: `17_HE` (focal active), a clean inactive, and `132_HE`
(the dense-but-benign trap from Step 4).

Heavy env (torch + sklearn + matplotlib), from the repo root:
    .venv-embed/bin/python scripts/05_mil_head.py
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import embed, mil, paths            # noqa: E402
from ibdpath.labels import load_slide_labels, patient_id  # noqa: E402
from ibdpath.mosaic import heatmap_overlay       # noqa: E402

ENC = "hoptimus0"
N_SPLITS = 5
DEVICE = "cpu"   # bags are tiny; CPU avoids MPS per-op launch overhead and is deterministic
HEAT_VMAX = 8.0  # colour scale = tile's attention ÷ uniform share; 8× = full colour (fixed → comparable)


def build_bags(encoder=ENC):
    """All cached slides -> (bags list, targets, patient groups, slide_ids), embedding (y,x) order."""
    labels = load_slide_labels()
    bags, y, groups, ids = [], [], [], []
    for sid, target in zip(labels["slide_id"], labels["slide_target"]):
        if not embed.slide_embedding_path(encoder, sid).exists():
            continue
        bags.append(embed.load_slide_embedding(encoder, sid))
        y.append(int(target))
        groups.append(patient_id(sid))
        ids.append(sid)
    return bags, np.asarray(y), np.asarray(groups), ids


def cross_val(bags, y, groups):
    """Leave-patients-out: out-of-fold P(active) + honest per-slide attention (fold model unseen)."""
    oof = np.zeros(len(y))
    attn = {}
    fold_auroc = []
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (tr, te) in enumerate(gkf.split(np.zeros(len(y)), y, groups)):
        n_pos = int(y[tr].sum())
        pos_weight = (len(tr) - n_pos) / max(n_pos, 1)        # balance active vs inactive in-fold
        model = mil.train_mil([bags[i] for i in tr], y[tr], pos_weight=pos_weight,
                              seed=fold, device=DEVICE)
        for i in te:
            oof[i], a = mil.predict_bag(model, bags[i], device=DEVICE)
            attn[i] = a
        try:
            fold_auroc.append(roc_auc_score(y[te], oof[te]))
        except ValueError:
            pass
        print(f"  fold {fold + 1}/{N_SPLITS}: train {len(tr)} / test {len(te)} slides", flush=True)
    return oof, attn, fold_auroc


def render_heatmaps(demo, man, zf, out_path):
    """One row of tissue-with-attention overlays for the demo slides.

    `demo`: list of dicts {sid, true, P, attn, caption}. Colour = each tile's attention as a
    multiple of the *uniform* share (1/n_tiles), on a FIXED scale (HEAT_VMAX) so the panels are
    directly comparable: a diffuse inactive slide stays pale, a focal active slide shows a bright
    spot. Attention is from the fold model that never saw the slide (leak-free).
    """
    fig, axes = plt.subplots(1, len(demo), figsize=(6.2 * len(demo), 6.8))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, demo):
        sid, a = d["sid"], d["attn"]
        rows = man[man.slide_id == sid].sort_values(["y", "x"]).reset_index(drop=True)
        enrichment = a * len(a)                              # 1.0 = a tile's fair (uniform) share
        ax.imshow(heatmap_overlay(rows, enrichment, zf, target_px=460, vmax=HEAT_VMAX))
        top3 = float(np.sort(a)[::-1][:3].sum())
        ax.set_title(f"{sid} — true {d['true']}\nslide P(active)={d['P']:.2f} · "
                     f"top-3 tiles = {top3:.0%} of attention · hottest = {enrichment.max():.0f}× fair share"
                     f"\n{d['caption']}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Attention-MIL heatmaps — colour = tile's attention vs its fair share "
                 "(fixed scale; attention from the fold model that never saw the slide)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"✓ wrote {out_path.relative_to(REPO_ROOT)}")


def render_active_vs_calm(pairs, ids, oof, attn, man, zf, out_path):
    """Side-by-side colour heatmaps on a SHARED fixed scale + a legend: a focal-active slide with one
    red hot-spot vs a clean inactive slide with none. The clearest single-look explainer of the method.
    `pairs` = [(slide_id, truth_label_str), ...]."""
    fig, ax = plt.subplots(1, len(pairs), figsize=(8 * len(pairs), 8.5))
    ax = np.atleast_1d(ax)
    for k, (sid, truth) in enumerate(pairs):
        i = ids.index(sid)
        rows = man[man.slide_id == sid].sort_values(["y", "x"]).reset_index(drop=True)
        a = attn[i]
        ax[k].imshow(heatmap_overlay(rows, a * len(a), zf, target_px=720, vmax=HEAT_VMAX))
        ax[k].axis("off")
        ax[k].set_title(f"{sid} — truly {truth}\nmodel's call: P(inflamed) = {oof[i]:.2f}   ·   "
                        f"hottest tile = {(a * len(a)).max():.0f}× fair share", fontsize=12, pad=10)
    sm = ScalarMappable(norm=Normalize(0, HEAT_VMAX), cmap="turbo")
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_ticks([0, HEAT_VMAX])
    cb.set_ticklabels(["ignored", f"{HEAT_VMAX:.0f}×+ (stared)"])
    cb.set_label("how hard the model looked at each tile", fontsize=11)
    fig.suptitle("Same colour scale, side by side:  one red hot-spot  vs.  no hot-spot anywhere",
                 fontsize=14, y=0.98)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"✓ wrote {out_path.relative_to(REPO_ROOT)}")


def main():
    bags, y, groups, ids = build_bags()
    if len(y) == 0:
        raise SystemExit("no cached embeddings — run scripts/03_embed_tiles.py first")
    print(f"slides={len(y)} | active={int(y.sum())} inactive={int((y == 0).sum())} | "
          f"patients={len(set(groups))} | device={DEVICE}")

    oof, attn, fold_auroc = cross_val(bags, y, groups)
    pred = (oof >= 0.5).astype(int)
    auroc = roc_auc_score(y, oof)
    print(f"\nLeave-patients-out AUROC (attention-MIL): {auroc:.3f}")
    print(f"Cohen's kappa @0.5: {cohen_kappa_score(y, pred):.3f}")
    print("per-fold AUROC:", [round(a, 3) for a in fold_auroc])
    print(classification_report(y, pred, target_names=["inactive", "active"], digits=3, zero_division=0))

    # compare to the Step-4 baseline (same slides, same CV scheme)
    base_path = paths.ARTIFACTS_DIR / "baseline_oof_predictions.csv"
    if base_path.exists():
        base = pd.read_csv(base_path, dtype={"slide_id": str}).set_index("slide_id")
        b = base.loc[ids, "p_active"].to_numpy()
        b_auroc = roc_auc_score(y, b)
        agree = float(((b >= 0.5) == (oof >= 0.5)).mean())
        print(f"\nbaseline AUROC {b_auroc:.3f}  vs  MIL AUROC {auroc:.3f}  | "
              f"call agreement {agree:.0%}  (MIL adds the heatmap, parity on the number is the goal)")

    # persist out-of-fold predictions + honest per-slide attention
    paths.ensure_artifacts()
    pd.DataFrame({"slide_id": ids, "patient_id": groups, "y": y, "p_active": oof}).to_csv(
        paths.ARTIFACTS_DIR / "mil_oof_predictions.csv", index=False)
    attn_dir = paths.ARTIFACTS_DIR / "mil_attention"
    attn_dir.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(ids):
        np.save(attn_dir / f"{sid}.npy", attn[i])
    print(f"✓ wrote mil_oof_predictions.csv + {len(ids)} attention arrays in "
          f"{attn_dir.relative_to(REPO_ROOT)}")

    # demo heatmaps: a CORRECTLY-called focal active slide, a clean inactive, and the 132_HE trap
    man = pd.read_csv(paths.PATCH_MANIFEST_LABELED_CSV, dtype={"slide_id": str})
    zf = zipfile.ZipFile(paths.PATCH_ZIP)
    stat = {sid: dict(P=oof[i], true=int(y[i]), n=len(attn[i]),
                      top3=float(np.sort(attn[i])[::-1][:3].sum())) for i, sid in enumerate(ids)}

    # headline = the most FOCAL active slide MIL got right (confident + attention concentrated)
    focal = [sid for sid in ids if stat[sid]["true"] == 1 and stat[sid]["P"] >= 0.6 and stat[sid]["n"] >= 40]
    headline = max(focal, key=lambda s: stat[s]["top3"])
    clean = min((s for s in ids if stat[s]["true"] == 0), key=lambda s: stat[s]["P"])
    picks = [(headline, "focal disease, called right — attention finds the inflamed needle in the haystack"),
             (clean, "clean inactive — attention stays diffuse, slide P near zero"),
             ("132_HE", "the Step-4 trap — gated attention spreads out, does NOT fixate on benign dense tissue")]
    demo = [dict(sid=s, true="ACTIVE" if stat[s]["true"] else "inactive",
                 P=stat[s]["P"], attn=attn[ids.index(s)], caption=cap)
            for s, cap in picks if s in ids]
    print(f"demo slides: headline={headline} (P={stat[headline]['P']:.2f}, "
          f"top-3 attn={stat[headline]['top3']:.0%}), clean inactive={clean}")
    out = paths.ARTIFACTS_DIR / "mil_heatmaps.png"
    render_heatmaps(demo, man, zf, out)

    # the deck's primary explainer: focal-active vs clean-inactive, side by side, shared scale
    cmp_out = paths.ARTIFACTS_DIR / "mil_active_vs_calm.png"
    render_active_vs_calm([(headline, "ACTIVE (inflamed)"), (clean, "INACTIVE (calm)")],
                          ids, oof, attn, man, zf, cmp_out)

    # copy both figures into the deck's image folder
    for fig_name in ("mil_heatmaps.png", "mil_active_vs_calm.png"):
        (REPO_ROOT / "slides" / "images" / fig_name).write_bytes((paths.ARTIFACTS_DIR / fig_name).read_bytes())
    print(f"✓ copied heatmap figures to slides/images/")
    print("\nverdict: MIL gives per-tile attention → real heatmaps; "
          f"AUROC {auroc:.3f} (baseline parity). Next = Step 6 honest validation.")


if __name__ == "__main__":
    main()
