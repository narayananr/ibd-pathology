#!/usr/bin/env python3
"""Build a TWO-LEVEL visual review of all 140 slides:

  artifacts/review/index.html            gallery: each slide's reconstructed thumbnail + TRUE
                                         label + the model's out-of-fold CALL (disagreements flagged).
  artifacts/review/thumbs/<slide>.png    reconstructed slide images (architecture level).
  artifacts/review/details/<slide>.png   the slide's individual TILES sorted by per-tile P(active)
                                         — click a card to hunt for neutrophils at the cell level.
  artifacts/review/errors_grid.png       the misclassified slides in one image.

Why two levels: a coarse thumbnail shows architecture, not cells, but "active" = neutrophils in the
epithelium (a microscopic feature). So the thumbnail sanity-checks the gross picture; the per-tile
detail is where a call can actually be adjudicated.

Light env, from the repo root:  .venv/bin/python scripts/make_review_gallery.py
"""
from __future__ import annotations

import html
import io
import sys
import zipfile
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from ibdpath import embed, paths               # noqa: E402
from ibdpath.baseline import build_slide_dataset  # noqa: E402
from ibdpath.mosaic import slide_thumbnail      # noqa: E402

ENC = "hoptimus0"
RED, GRN = "#ff5d6c", "#33c08d"


def tile_montage(rows, sc, zf, tile_px=210, cols=6, pad=4):
    """The slide's tiles sorted by P(active) high->low, each labeled + colour-bordered (red=hot)."""
    order = np.argsort(sc)[::-1]
    n = len(order)
    rowsn = (n + cols - 1) // cols
    cw, ch = cols * (tile_px + pad) + pad, rowsn * (tile_px + pad) + pad
    canvas = Image.new("RGB", (cw, ch), (15, 17, 23))
    draw = ImageDraw.Draw(canvas)
    for k, idx in enumerate(order):
        im = Image.open(io.BytesIO(zf.read(rows.image_path.iloc[idx]))).convert("RGB").resize((tile_px, tile_px))
        cx, cy = pad + (k % cols) * (tile_px + pad), pad + (k // cols) * (tile_px + pad)
        canvas.paste(im, (cx, cy))
        p = float(sc[idx])
        col = (255, 93, 108) if p >= 0.5 else (51, 192, 141)
        draw.rectangle([cx, cy, cx + tile_px - 1, cy + tile_px - 1], outline=col, width=3)
        draw.rectangle([cx, cy, cx + 46, cy + 15], fill=(0, 0, 0))
        draw.text((cx + 3, cy + 3), f"{p:.2f}", fill=col)
    return canvas


def main():
    review = paths.ARTIFACTS_DIR / "review"
    thumbs, details = review / "thumbs", review / "details"
    thumbs.mkdir(parents=True, exist_ok=True)
    details.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(paths.PATCH_MANIFEST_LABELED_CSV, dtype={"slide_id": str})
    oof = pd.read_csv(paths.ARTIFACTS_DIR / "baseline_oof_predictions.csv").set_index("slide_id")
    zf = zipfile.ZipFile(paths.PATCH_ZIP)

    # per-tile localizer = linear "active direction" (no scaler); illustrative, proper one = MIL
    X, y, groups, ids = build_slide_dataset(ENC)
    loc = LogisticRegression(max_iter=5000, class_weight="balanced").fit(X, y)

    order = oof.sort_values("p_active", ascending=False).index.tolist()
    recs = []
    for i, sid in enumerate(order, 1):
        rows = man[man.slide_id == sid].sort_values(["y", "x"]).reset_index(drop=True)  # embedding order
        sc = loc.predict_proba(embed.load_slide_embedding(ENC, sid))[:, 1]
        if not (thumbs / f"{sid}.png").exists():
            slide_thumbnail(rows, zf, target_px=300).save(thumbs / f"{sid}.png")
        if not (details / f"{sid}.png").exists():
            tile_montage(rows, sc, zf).save(details / f"{sid}.png")
        p = float(oof.loc[sid, "p_active"])
        true = "active" if oof.loc[sid, "y"] == 1 else "inactive"
        recs.append(dict(sid=sid, true=true, call="active" if p >= 0.5 else "inactive",
                         p=p, wrong=(("active" if p >= 0.5 else "inactive") != true),
                         hot=float(sc.max()), ntiles=len(sc)))
        if i % 20 == 0:
            print(f"  {i}/{len(order)} slides", flush=True)
    df = pd.DataFrame(recs)
    errors = df[df.wrong].reset_index(drop=True)

    # ---- errors grid ----
    n, cols = len(errors), 3
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(cols * 3.2, ((n + cols - 1) // cols) * 3.5))
    for ax, (_, r) in zip(axes.ravel(), errors.iterrows()):
        ax.imshow(plt.imread(thumbs / f"{r.sid}.png"))
        ax.set_title(f"{r.sid}\nTRUE={r.true} → CALL={r.call}\nP={r.p:.2f}  (hottest tile {r.hot:.2f})", fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"The {n} misclassified slides", fontsize=13)
    fig.tight_layout()
    fig.savefig(review / "errors_grid.png", dpi=120, bbox_inches="tight")

    # ---- HTML ----
    def chip(label):
        c = RED if label == "active" else GRN
        return f'<span class="chip" style="background:{c}22;color:{c};border:1px solid {c}">{label}</span>'

    def card(r):
        cls = "card wrong" if r.wrong else "card"
        mark = "✗ MISCALL" if r.wrong else "✓"
        return (f'<a class="{cls}" href="details/{html.escape(r.sid)}.png" target="_blank" '
                f'title="open the {r.ntiles} tiles of {r.sid}, sorted by P(active)">'
                f'<img src="thumbs/{html.escape(r.sid)}.png" loading="lazy">'
                f'<div class="meta"><b>{html.escape(r.sid)}</b> {mark}<br>'
                f'true {chip(r.true)} · call {chip(r.call)}<br>'
                f'<span class="p">slide P={r.p:.2f} · 🔬 hottest tile {r.hot:.2f}</span></div></a>')

    err_cards = "\n".join(card(r) for _, r in errors.iterrows())
    all_cards = "\n".join(card(r) for _, r in df.iterrows())
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Slide review — true label vs model call</title><style>
body{{background:#0f1117;color:#e8eaf0;font-family:-apple-system,Arial,sans-serif;margin:0;padding:24px}}
h1{{font-size:22px;margin:0 0 .3em}} h2{{margin:30px 0 10px}}
.note{{color:#9aa3b2;font-size:14px;line-height:1.6;max-width:1000px}}
.note b{{color:#e8eaf0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px}}
.card{{background:#171a23;border:1px solid #2a2f3d;border-radius:10px;padding:8px;text-decoration:none;color:inherit;display:block}}
.card:hover{{border-color:#5b8cff}}
.card.wrong{{border:2px solid #f5c451;box-shadow:0 0 0 1px #f5c451 inset}}
.card img{{width:100%;border-radius:6px;background:#000;display:block}}
.meta{{font-size:13px;margin-top:6px;line-height:1.55}}
.chip{{padding:.05em .55em;border-radius:999px;font-size:11px;font-weight:700}}
.p{{color:#9aa3b2}}
</style></head><body>
<h1>Slide review — true label vs the model's out-of-fold call</h1>
<p class="note"><b>Two levels.</b> The thumbnail shows <b>architecture</b>, but "active" is defined by
<b>neutrophils in the epithelium</b> (cryptitis, crypt abscess) — a <i>microscopic</i> feature you can't
see in a coarse mosaic. So the thumbnail only sanity-checks the gross picture (focal patterns, tiny
biopsies); <b>to actually judge a call, click a card</b> to open that slide's individual tiles, sorted
by per-tile P(active) (red border = hot) — then hunt for neutrophils in the hottest tiles.
Sorted by slide P(active), high → low. Yellow border = model disagreed with the label.
Baseline AUROC 0.984; {len(errors)}/140 disagreements.</p>
<h2>⚠️ The {len(errors)} disagreements — check these first</h2>
<div class="grid">{err_cards}</div>
<h2>All 140 slides (P(active) high → low)</h2>
<div class="grid">{all_cards}</div>
</body></html>"""
    (review / "index.html").write_text(doc)
    print(f"\n✓ wrote {(review / 'index.html').relative_to(REPO_ROOT)}  "
          f"({len(df)} slides, {len(errors)} disagreements; per-slide tile views in details/)")


if __name__ == "__main__":
    main()
