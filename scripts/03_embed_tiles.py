#!/usr/bin/env python3
"""
Step 3 — embed every tile with the frozen H-optimus-0 foundation model.
=======================================================================
Read each tile from the patch zip, run it through the frozen encoder on the Mac GPU (MPS), and
cache one ~1,536-number embedding per tile. Output, per slide and keyed by encoder:

    artifacts/embeddings/<encoder>/<slide_id>.npy   # (n_tiles, 1536) float32, rows = manifest order

"Frozen" = forward passes only, no training. Compute once, reuse forever. **Idempotent**: any slide
already cached is skipped, so a re-run resumes where it left off.

Must run with the heavy env (torch/timm). From the repo root:
    HF_HUB_OFFLINE=1 .venv-embed/bin/python scripts/03_embed_tiles.py [--limit N] [--batch 32]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # unsupported ops fall back to CPU

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import embed, paths   # noqa: E402

ENCODER = "hoptimus0"
HF_MODEL = "hf-hub:bioptimus/H-optimus-0"
NORM_MEAN = (0.707223, 0.578729, 0.703617)   # H-optimus-0's documented normalization
NORM_STD = (0.211883, 0.230117, 0.177517)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])


def load_encoder(device):
    model = timm.create_model(HF_MODEL, pretrained=True, init_values=1e-5, dynamic_img_size=False)
    return model.to(device).eval()


def embed_slide(model, tf, zf, image_paths, device, batch_size):
    """Embed one slide's tiles -> (n_tiles, 1536) float32, L2-normalized, in the given order."""
    out = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        batch = torch.stack([tf(Image.open(io.BytesIO(zf.read(p))).convert("RGB"))
                             for p in chunk]).to(device)
        with torch.inference_mode():
            feats = model(batch)
        out.append(feats.float().cpu().numpy())
    return embed.l2_normalize(np.concatenate(out)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N slides (testing)")
    ap.add_argument("--batch", type=int, default=32, help="tiles per forward pass")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    df = pd.read_csv(paths.PATCH_MANIFEST_LABELED_CSV, dtype={"slide_id": str})
    slides = list(df.groupby("slide_id"))
    if args.limit:
        slides = slides[:args.limit]
    out_dir = embed.embeddings_dir(ENCODER)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} | encoder={ENCODER} | {len(slides)} slides | "
          f"out={out_dir.relative_to(REPO_ROOT)}", flush=True)

    print("loading frozen encoder (from cache)...", flush=True)
    t0 = time.time()
    model = load_encoder(device)
    tf = build_transform()
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    zf = zipfile.ZipFile(paths.PATCH_ZIP)
    done_tiles = done_slides = skipped = 0
    t_start = time.time()
    for k, (sid, g) in enumerate(slides, 1):
        if embed.slide_embedding_path(ENCODER, sid).exists():
            skipped += 1
            continue
        g = g.sort_values(["y", "x"])                      # stable tile order within a slide
        feats = embed_slide(model, tf, zf, list(g["image_path"]), device, args.batch)
        embed.save_slide_embedding(ENCODER, sid, feats)
        done_tiles += len(feats)
        done_slides += 1
        if done_slides % 10 == 0 or k == len(slides):
            rate = done_tiles / max(time.time() - t_start, 1e-6)
            print(f"  [{k}/{len(slides)}] {sid}: {feats.shape} | {done_tiles} tiles done "
                  f"| {rate:.0f} tiles/s", flush=True)

    meta = {
        "encoder": ENCODER, "hf_model": HF_MODEL, "embed_dim": embed.EMBED_DIM,
        "input_size": 224, "source_stored_px": 512, "l2_normalized": True,
        "norm_mean": list(NORM_MEAN), "norm_std": list(NORM_STD),
        "n_slides_cached": len(list(out_dir.glob("*.npy"))),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✓ embedded {done_slides} slides ({done_tiles} tiles), skipped {skipped} cached.")
    print(f"  cache now holds {meta['n_slides_cached']} slide files in {out_dir.relative_to(REPO_ROOT)}")
    print(f"  wall time this run: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
