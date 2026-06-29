# BUILD_PLAN.md — IBD Inflamed/Healed v1

Work top to bottom; check items off as you go. **Stop and confirm before starting anything
listed under "DEFERRED" in CLAUDE.md.** Keep each script small and runnable on its own.

## Phase 0 — Environment & data
- [ ] Set up Python env; install `requirements.txt`; confirm the OpenSlide **system** lib is present
- [ ] Install TRIDENT (`mahmoodlab/trident`); confirm `run_batch_of_slides.py --help` works
- [ ] Download IBDColEpi HE WSIs (HuggingFace `andreped/IBDColEpi` or DataverseNO `10.18710/TLA01U`)
- [ ] Write `data/download_ibdcolepi.py` that fetches HE slides + active/inactive labels into a
      manifest CSV: `slide_id, path, label`
- [ ] Sanity: open 2–3 slides, confirm `.ndpi` reads, eyeball one active and one inactive slide

## Phase 1 — Embeddings
- [ ] Pick the foundation model (default `hoptimus1`/H0-mini if avoiding HF gating; else `uni_v2`)
- [ ] Run TRIDENT seg → coords → feat at 20x / 256px on the HE slides
- [ ] Verify feature `.h5` files exist; load one and check `features`/`coords` shapes
- [ ] Organize/cache embeddings keyed by `(slide, encoder, mag, patch_size)`

## Phase 2 — Signal check (does the FM separate active/inactive at all?)
- [ ] Write `baseline_slide_clf.py`: mean-pool each slide's tile embeddings, fit logistic
      regression on active/inactive with **GroupKFold by slide**
- [ ] Report held-out AUROC. If ≳0.70, embeddings carry the signal → proceed.
- [ ] If weak: try a different encoder; check segmentation didn't drop real tissue

## Phase 3 — Region map (the actual deliverable)
- [ ] Option A (recommended, matches the labels): implement an **attention-MIL** head (ABMIL/CLAM)
      on the embeddings, train on slide labels, export per-tile attention → heatmap.
      Compare behaviour against `owkin/imilia`.
- [ ] Option B: annotate ~5 slides in QuPath (inflamed/healed polygons), export GeoJSON to
      `./annotations`, run `ibd_inflamed_probe.py` for the linear-probe heatmap.
- [ ] Produce inflamed/healed overlays for a held-out set; eyeball vs known-active slides

## Phase 4 — Honest validation
- [ ] Leave-slides-out (or patient-out) CV; report AUROC + agreement (kappa) on active/inactive
- [ ] Calibrate the inflamed threshold; log where the map fails (focal disease, artifacts)
- [ ] Short results note: encoder, metrics, example overlays, known failure modes

## DEFERRED — only after v1 works, and only after confirming scope (see CLAUDE.md)
- [ ] Compartment segmentation using IBDColEpi epithelium masks
- [ ] Cell seg/typing (CellViT++/HoVer-Net; fine-tune on Lizard/CoNIC)
- [ ] Geboes/Nancy/Robarts heads on cell + compartment features, with consensus validation
