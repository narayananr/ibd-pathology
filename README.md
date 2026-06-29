# IBD H&E — Inflamed vs Healed

> A computational-pathology prototype that turns an H&E whole-slide image of a colon biopsy into a
> per-region **inflamed-vs-healed** heatmap, with honest, leakage-aware validation — built entirely on a
> **frozen** pathology foundation model (no encoder training) on **public** data.

> ⚠️ **Research & learning prototype. Not a medical device, not validated for clinical use.**

![Attention heatmap: an active slide lights up one inflamed focus; a healed slide stays cool](slides/images/mil_active_vs_calm.png)

*Left: an **active** slide — the model spotlights the one inflamed region. Right: a **healed** slide — nothing
lights up. The model was trained on slide-level labels only; nobody marked individual tiles.*

---

## What it does

Inflammatory bowel disease (Crohn's, ulcerative colitis) biopsies are graded by pathologists as **active**
(neutrophils attacking the gut lining) vs **healed** — a manual, subjective call that guides treatment. This
project predicts that call per slide **and** produces a per-tile heatmap of *where* the inflammation is, so a
human can check the evidence.

## Results (honest)

Evaluated **leave-patients-out** (a patient's slides never straddle the train/test split) on 140 H&E slides:

| Head | AUROC | What it adds |
|---|---|---|
| Mean-pool + logistic regression (baseline) | **0.984** | the slide-level active/healed call |
| Gated attention-MIL (Step 5) | **0.976** | a per-tile **heatmap** — the *where* |

- **No leakage:** shuffling the labels collapses AUROC to **0.52**.
- **Honest limitation:** the MIL head does **not** beat the baseline on the number, and **focal disease** (a
  tiny inflamed spot in otherwise-normal tissue) is **missed by both heads**. The heatmap, not a higher score,
  is the deliverable. See the deck's "disease is a spectrum" slide.

## How it works

```
whole-slide image → tiles → [ frozen H-optimus-0, 1.1B-param ViT ] → 1,536-d embedding per tile (cached)
                                                                   → light head → stitched heatmap
```

One **frozen** pathology foundation model encodes tiles into embeddings; every model on top is a *light head*
trained on the cached embeddings. The heavy lifting is pretrained — so the trainable parts are tiny and run on
a laptop GPU.

## Repository layout

```
src/ibdpath/      reusable module: paths, manifest parsing, labels, embedding cache, baseline, MIL head, mosaics
scripts/          numbered pipeline (01 build manifest → 05 attention-MIL) + figure/gallery builders
tests/            unit/integration tests (38) — run per-env (see below)
metadata/         slide_labels.csv — curated slide→active/inactive labels (derived; committed)
slides/           index.html (full build log) · overview.html (5-slide summary) · NOTES.md · images/
artifacts/        generated tables, embeddings, figures, galleries  (git-ignored)
data/             you download the dataset here                       (git-ignored)
REFERENCES.md     citations & attribution    ·    PROGRESS.md  resume/status    ·    APPROACH.md  plain-language walkthrough
```

## Setup

Two virtual environments (the embedding/MIL step needs PyTorch; everything else stays light):

```bash
# light env — analysis, baseline, plotting  (Python 3.11+)
python3 -m venv .venv
.venv/bin/pip install numpy pandas pillow tifffile scikit-learn matplotlib h5py

# heavy env — embedding + attention-MIL  (Python 3.12; pulls in torch/timm)
python3.12 -m venv .venv-embed
.venv-embed/bin/pip install torch timm scikit-learn matplotlib pandas pillow
```

## Get the data (not hosted here — linked)

The dataset is **not** redistributed in this repo; download it from the source:

- **IBDColEpi** (CC0 public domain) — DataverseNO `doi:10.18710/TLA01U`, or the
  [HuggingFace mirror](https://huggingface.co/datasets/andreped/IBDColEpi). This project uses the **pre-tiled
  H&E patch set**; place it at `data/ibdcolepi/patch-dataset-HE.zip`.
- **H-optimus-0** weights are **gated** on HuggingFace — request access, then authenticate in the heavy env:
  ```bash
  .venv-embed/bin/hf auth login        # needs read access to bioptimus/H-optimus-0
  ```

## Run the pipeline

```bash
.venv/bin/python        scripts/01_build_manifest.py     # patch zip → artifacts/patch_manifest.csv
.venv/bin/python        scripts/02_attach_labels.py      # + active/inactive + patient_id
.venv-embed/bin/python  scripts/03_embed_tiles.py        # frozen H-optimus-0 → cached embeddings
.venv/bin/python        scripts/04_baseline_clf.py       # mean-pool + logreg → AUROC 0.984
.venv-embed/bin/python  scripts/05_mil_head.py           # attention-MIL → AUROC 0.976 + heatmaps
# figures & review gallery (optional):
.venv/bin/python        scripts/make_review_gallery.py
.venv/bin/python        scripts/make_disease_spectrum.py
.venv/bin/python        scripts/make_overview_figs.py
```

Tests:

```bash
.venv/bin/python -m unittest tests.test_manifest tests.test_labels tests.test_embed tests.test_baseline
.venv-embed/bin/python -m unittest tests.test_mil
```

## Slides

- `slides/index.html` — the full build log (every step, with evidence).
- `slides/overview.html` — a standalone **5-slide** overview for someone seeing the work for the first time.

Open either in a browser; navigate with arrow keys / space.

## Limitations

- Slide-level labels only; **focal** disease (a small inflamed focus) is the hard, unsolved case here.
- Slide-level "active" = intraepithelial neutrophils; cell-level neutrophil confirmation is **deferred** (a
  future head on the same embeddings, e.g. CellViT/HoVer-Net).
- Single public cohort (NTNU / St. Olavs). No external/multi-center validation. Not a clinical tool.

## References & citation

Full citations (dataset, model, method, libraries, tools) are in **[`REFERENCES.md`](REFERENCES.md)**. If you
use this, please cite the dataset:

> Pettersen, H. S., et al. (2021). *Code-Free Development and Deployment of Deep Segmentation Models for Digital
> Pathology.* Frontiers in Medicine, 8:816281. Dataset: `doi:10.18710/TLA01U` (DataverseNO, CC0).

## License

- **Code** in this repository: **MIT** — see [`LICENSE`](LICENSE).
- **IBDColEpi dataset**: CC0 (public domain); not redistributed here — linked above. Result figures contain
  small tile excerpts under that license, with citation.
- **Teaching H&E images** in the deck (cryptitis / crypt-abscess): Wikimedia Commons, CC-BY-SA, by *Nephron*.

## Disclaimer

This is an independent research/learning project. It is **not** affiliated with the dataset authors or
Bioptimus, has **not** been clinically validated, and must **not** be used for diagnosis or treatment.
