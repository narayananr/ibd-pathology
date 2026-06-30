#!/usr/bin/env python3
"""
EXPERIMENTAL (deferred scope) — epithelium segmentation as a light head on frozen H-optimus.
============================================================================================
Gate-1 feasibility test for the segmentation plan: can we segment epithelium from the existing
512-px patches by putting a *light* decoder on the frozen encoder's **patch tokens**?

Approach (smallest useful slice, ~a dozen slides):
  tile -> frozen H-optimus `forward_features` -> 256 patch tokens (16x16 x 1536)
       -> a per-patch **logistic probe** (sklearn) -> P(epithelium) per patch
       -> upsample 16x16 -> 512 -> threshold -> compare to the ground-truth mask (Dice).
Trained/evaluated with a **patient split** (held-out patients). If even this coarse linear probe
gets a decent Dice, the signal is there and a conv decoder is the refinement.

Heavy env:  HF_HUB_OFFLINE=1 .venv-embed/bin/python scripts/exp_epithelium_seg.py
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import matplotlib
import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torchvision import transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import epithelium as ep, paths   # noqa: E402

HF_MODEL = "hf-hub:bioptimus/H-optimus-0"
NORM_MEAN = (0.707223, 0.578729, 0.703617)
NORM_STD = (0.211883, 0.230117, 0.177517)
GRID = 16            # 16x16 patch grid (224/14)
N_PER_CLASS = 6      # slides per class (distinct patients)


def select_slides(man):
    s = man.drop_duplicates("slide_id")[["slide_id", "slide_target", "patient_id"]]
    act = s[s.slide_target == 1].drop_duplicates("patient_id").head(N_PER_CLASS)
    ina = s[s.slide_target == 0].drop_duplicates("patient_id").head(N_PER_CLASS)
    chosen = pd.concat([act, ina])
    # patient split: last 2 of each class held out for test
    test_ids = set(act.slide_id.tail(2)) | set(ina.slide_id.tail(2))
    chosen["split"] = ["test" if sid in test_ids else "train" for sid in chosen.slide_id]
    return chosen


def patch_tokens(model, tf, zf, image_path, device):
    x = tf(Image.open(io.BytesIO(zf.read(image_path))).convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        feats = model.forward_features(x)                 # (1, 5+256, 1536)
    pt = feats[:, model.num_prefix_tokens:, :]            # (1, 256, 1536)
    return pt.squeeze(0).float().cpu().numpy()            # (256, 1536)


def patch_targets(zf, mask_path):
    """Per-patch epithelium fraction (16x16) from the full 512 mask, + the full mask for Dice."""
    full = ep.load_mask(zf, mask_path)                                  # (512,512) {0,1}
    frac = np.array(Image.fromarray(full * 255).resize((GRID, GRID), Image.BILINEAR)) / 255.0
    return frac.reshape(-1), full                                       # (256,), (512,512)


def dice(pred_bool, gt_bool):
    inter = np.logical_and(pred_bool, gt_bool).sum()
    s = pred_bool.sum() + gt_bool.sum()
    return 1.0 if s == 0 else float(2 * inter / s)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    man = pd.read_csv(paths.PATCH_MANIFEST_LABELED_CSV, dtype={"slide_id": str})
    chosen = select_slides(man)
    print(f"device={device} | slides: {list(chosen.slide_id)}")
    print(f"test (held-out patients): {list(chosen[chosen.split=='test'].slide_id)}")

    model = timm.create_model(HF_MODEL, pretrained=True, init_values=1e-5,
                              dynamic_img_size=False).to(device).eval()
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                             transforms.Normalize(NORM_MEAN, NORM_STD)])
    zf = zipfile.ZipFile(paths.PATCH_ZIP)

    Xtr, ytr = [], []
    test_tiles = []   # (X_patch(256,1536), full_mask, slide_id, image_path)
    for k, (_, srow) in enumerate(chosen.iterrows(), 1):
        rows = man[man.slide_id == srow.slide_id]
        for _, r in rows.iterrows():
            pt = patch_tokens(model, tf, zf, r.image_path, device)
            frac, full = patch_targets(zf, r.mask_path)
            if srow.split == "train":
                Xtr.append(pt); ytr.append((frac >= 0.5).astype(int))
            else:
                test_tiles.append((pt, full, srow.slide_id, r.image_path))
        print(f"  [{k}/{len(chosen)}] {srow.slide_id} ({srow.split}) done", flush=True)

    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    print(f"\ntrain patches: {Xtr.shape} | epithelium {ytr.mean():.0%}")
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced").fit(Xtr, ytr)

    # ---- evaluate on held-out patients ----
    recs, patch_y, patch_p = [], [], []
    for pt, full, sid, ipath in test_tiles:
        prob = clf.predict_proba(pt)[:, 1].reshape(GRID, GRID)
        up = np.array(Image.fromarray((prob * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)) / 255.0
        gtf = float((full > 0).mean())
        recs.append(dict(sid=sid, ipath=ipath, up=up, full=full, gtf=gtf, dice=dice(up >= 0.5, full > 0)))
        frac = np.array(Image.fromarray(full * 255).resize((GRID, GRID), Image.BILINEAR)) / 255.0
        patch_y.append((frac.reshape(-1) >= 0.5).astype(int)); patch_p.append(prob.reshape(-1))
    patch_y = np.concatenate(patch_y); patch_p = np.concatenate(patch_p)
    auroc = roc_auc_score(patch_y, patch_p)
    df = pd.DataFrame([{k: r[k] for k in ("sid", "gtf", "dice")} for r in recs])
    has_ep = df[df.gtf > 0.05]
    print(f"\nHELD-OUT ({len(df)} tiles; {len(has_ep)} contain epithelium, {(df.gtf < 0.01).sum()} empty):")
    print(f"  patch-level AUROC {auroc:.3f}   (signal quality — NOT confounded by empty tiles)")
    print(f"  mean Dice on epithelium tiles (GT>5%): {has_ep.dice.mean():.3f}")
    print(f"  mean Dice all tiles (empty score 1.0): {df.dice.mean():.3f}   ← inflated, don't headline")

    # ---- render representative tiles: most epithelium, spread across the test slides ----
    show, seen = [], {}
    for i in sorted(range(len(recs)), key=lambda k: recs[k]["gtf"], reverse=True):
        s = recs[i]["sid"]
        if seen.get(s, 0) < 2:
            show.append(recs[i]); seen[s] = seen.get(s, 0) + 1
        if len(show) >= 4:
            break
    fig, ax = plt.subplots(3, len(show), figsize=(4 * len(show), 11))
    for j, r in enumerate(show):
        tile = ep.load_tile(zf, r["ipath"])
        ax[0, j].imshow(tile); ax[0, j].set_title(f"{r['sid']}\nH&E", fontsize=10); ax[0, j].axis("off")
        ax[1, j].imshow(ep.overlay(tile, (r["full"] > 0).astype(np.uint8))); ax[1, j].axis("off")
        ax[1, j].set_title("ground truth", fontsize=10)
        ax[2, j].imshow(ep.overlay(tile, (r["up"] >= 0.5).astype(np.uint8), color=(91, 140, 255))); ax[2, j].axis("off")
        ax[2, j].set_title(f"predicted · Dice {r['dice']:.2f}", fontsize=10)
    fig.suptitle(f"Epithelium segmentation — linear probe on frozen H-optimus patch tokens "
                 f"(held-out patients · patch AUROC {auroc:.3f} · Dice {has_ep.dice.mean():.2f} on epithelium tiles)",
                 fontsize=12)
    fig.tight_layout()
    out = paths.ARTIFACTS_DIR / "epithelium_seg_demo.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"✓ wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
