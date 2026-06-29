# References & attribution

This project is a learning/portfolio prototype built on **public data** and **open tools**. Nothing here
is a validated clinical product. Below is what we used and how, with citations. The dataset is **CC0**
(public-domain) so attribution is a courtesy, not a legal requirement — but we cite it gladly.

---

## Dataset — IBDColEpi

140 H&E + 111 CD3 whole-slide images of colonic mucosa (active / inactive IBD + controls) from NTNU /
St. Olavs Hospital, Trondheim, with pixel-level epithelium annotations. **This project uses only the H&E
subset and the slide-level active/inactive labels.** Released under **CC0 1.0** (public domain).

- **Dataset record (cite this):** Pettersen, H. S., Belevich, I., Røyset, E. S., Smistad, E., Jokitalo, E.,
  Reinertsen, I., Bakke, I., & Pedersen, A. (2021). *140 HE and 111 CD3-stained colon biopsies of active and
  inactive inflammatory bowel disease with epithelium annotated: the IBDColEpi dataset.* DataverseNO, V1.
  https://doi.org/10.18710/TLA01U  (CC0 1.0)
- **Companion paper:** Pettersen, H. S., et al. (2021). *Code-Free Development and Deployment of Deep
  Segmentation Models for Digital Pathology.* **Frontiers in Medicine**, 8:816281.
  https://doi.org/10.3389/fmed.2021.816281
- **Mirrors:** HuggingFace [`andreped/IBDColEpi`](https://huggingface.co/datasets/andreped/IBDColEpi) ·
  Kaggle [`henrikpe/251-he-cd3-wsis-annotated-epithelium-ibdcolepi`](https://www.kaggle.com/datasets/henrikpe/251-he-cd3-wsis-annotated-epithelium-ibdcolepi)

> Note: the epithelium annotation masks are **not** used here (compartment segmentation is deferred scope).
> Slide images shown in this repo's figures are excerpts of this CC0 dataset.

---

## Foundation model — H-optimus-0

A 1.1-billion-parameter ViT pathology foundation model, pretrained self-supervised on H&E. Used **frozen**
(inference only) to embed tiles into 1,536-dim vectors. License **Apache-2.0** (weights gated on HuggingFace).

- Bioptimus (2024). *H-optimus-0.* Model card: https://huggingface.co/bioptimus/H-optimus-0
- Loaded via `timm` as `hf-hub:bioptimus/H-optimus-0`.

Alternatives noted for the same slot (not used in v1): `bioptimus/H0-mini`, `MahmoodLab/UNI2`, `paige-ai/Virchow2`.

---

## Method — attention-based Multiple-Instance Learning

The Step-5 head is a gated attention-MIL (ABMIL).

- Ilse, M., Tomczak, J. M., & Welling, M. (2018). *Attention-based Deep Multiple Instance Learning.*
  **ICML 2018**, PMLR 80:2127–2136. https://proceedings.mlr.press/v80/ilse18a.html
- Related weakly-supervised WSI framework (referenced, not used): Lu, M. Y., et al. (2021). *Data-efficient
  and weakly supervised computational pathology on whole-slide images (CLAM).* **Nature Biomedical
  Engineering**, 5:555–570. https://doi.org/10.1038/s41551-020-00682-w

---

## Reference implementation (sanity-check, not a dependency)

- **Owkin IMILIA** — attention-MIL inflammation prediction on this *exact* IBDColEpi data; used to sanity-check
  our numbers. https://github.com/owkin/imilia

---

## Software libraries (used in the pipeline)

- **PyTorch** — Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning
  Library.* NeurIPS 2019. https://pytorch.org
- **timm (PyTorch Image Models)** — Wightman, R. (2019). https://github.com/huggingface/pytorch-image-models ·
  doi:10.5281/zenodo.4414861
- **scikit-learn** — Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12:2825–2830.
- **NumPy** — Harris, C. R., et al. (2020). *Array programming with NumPy.* Nature 585:357–362.
- **pandas** — McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* SciPy 2010.
- **Matplotlib** — Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.* CiSE 9(3):90–95.
- **Pillow (PIL fork)** — Clark, A., et al. https://python-pillow.org

## Tools referenced / recommended (not directly used in v1)

We used the dataset's **pre-tiled** H&E patch set, so the following WSI tools are recommended in the project
docs but not invoked by the built code:

- **TRIDENT** (Mahmood Lab) — WSI segment → tile → embed. https://github.com/mahmoodlab/trident
- **OpenSlide** — Goode, A., et al. (2013). *OpenSlide: A vendor-neutral software foundation for digital
  pathology.* J Pathol Inform 4:27. https://openslide.org
- **QuPath** — Bankhead, P., et al. (2017). *QuPath: Open source software for digital pathology image
  analysis.* Sci Rep 7:16878. https://qupath.github.io

---

## Teaching figures (not from IBDColEpi)

A few illustrative H&E examples in the build-log deck (cryptitis / crypt-abscess) are from **Wikimedia
Commons** (CC-BY-SA; images by *Nephron*) and are used purely for teaching. All *result* figures use real
IBDColEpi tiles.
