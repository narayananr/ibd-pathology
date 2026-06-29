# Roadmap

## Status — v1 complete (Steps 1–6)

Frozen **H-optimus-0** embeddings → light heads → per-tile inflamed/healed heatmap + an honest validation
report.

- **Baseline** (mean-pool + logistic regression): leave-patients-out **AUROC 0.984**.
- **Attention-MIL**: **0.976** + a per-tile attention **heatmap** (the *where*).
- **Validation**: AUROC + 95% CI (patient bootstrap), sensitivity/specificity, a label-permutation leakage
  check (collapses to ~0.46), head-vs-head agreement (96%).
- **Open problem**: *focal* disease — a small, few-tile inflamed focus — is still **missed by both heads**.

## Deferred upgrades

Each is a **new head on the SAME frozen embeddings / cached tiles** — the foundation model never changes.
Per the project guardrails these cross into **deferred scope**: confirm before building.

### 1. Cell-level quantification — cell type & inflammation status per tile ⟵ strongest next step

Turns the current black-box per-tile score into **interpretable, countable, clinically-aligned** features
(what the Geboes / Nancy / Robarts indices actually grade).

**Pipeline, per tile:**

1. **Detect & classify every nucleus** — a pretrained cell model (**HoVer-Net** or **CellViT/CellViT++**,
   trained on PanNuke / CoNIC / MoNuSAC), run frozen → every nucleus with location, size, and **type**
   (neutrophil, lymphocyte, plasma cell, eosinophil, epithelial, connective).
2. **Per-tile cell counts** — neutrophil / lymphocyte / plasma-cell densities, neutrophil-to-lymphocyte
   ratio, epithelial-cell count.
3. **Compartment context (where the cells are)** — segment epithelium vs lamina propria vs crypt lumen vs
   surface, then intersect cell locations with compartments: neutrophils **in epithelium** → *cryptitis*;
   **filling a crypt lumen** → *crypt abscess*; denuded surface → *erosion / ulcer*. IBDColEpi ships
   pixel-level **epithelium masks**, which directly enable "intraepithelial neutrophil" detection (the formal
   definition of active disease).
4. **Per-tile status** — a structured readout, e.g. `{cryptitis: yes, crypt_abscesses: 2, surface_erosion:
   no, lamina_propria_neutrophils: high}`, or a graded category (quiescent → mild → moderate → severe).
5. **Roll up to indices** — aggregate the per-tile structure → slide-level **Geboes / Nancy / Robarts** scores.

**How it fits:** a second branch alongside v1 — keep the attention heatmap to *find* the active region, then
run cell detection *there* to *quantify* it.

**Hard parts (honest):** neutrophils are the tricky cell type (small; confused with eosinophils / apoptotic
debris); generic cell models may transfer imperfectly to colon H&E; compartment segmentation is its own
model; trusting the numbers needs some cell/index ground truth.

**First slice idea:** run CellViT on the `137_HE` active focus, count intraepithelial neutrophils, and
overlay the detections on the tile.

➡️ **Detailed plans:** [`docs/segmentation-and-cell-typing-plan.md`](docs/segmentation-and-cell-typing-plan.md)
(master — compartment **segmentation** + **cell typing** + the join) and
[`docs/cell-quantification-plan.md`](docs/cell-quantification-plan.md) (cell-typing deep-dive: the two
feasibility gates — resolution & neutrophil specificity).

### 2. Stain normalization

Standardize H&E colour across scanners/labs before embedding → robustness to a new site/cohort.

### 3. Crypt / gland architecture

Segment crypts/glands → architectural-distortion features (branching, crypt dropout) = the *chronic* axis of
IBD, complementing the *active* (neutrophil) axis.

### 4. Geboes / Nancy / Robarts scoring heads

The formal histologic-activity indices, built on the cell-level and architecture features above — the
long-term target (a histologic-remission call).
