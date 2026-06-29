"""ibdpath — small reusable helpers for the IBD inflamed/healed pipeline.

Kept deliberately tiny. Today it holds:
  - paths.py    : one place for every data/output path
  - manifest.py : parse the patch zip's filenames into a tidy table

Heavier pieces (tile IO, embeddings, heads) arrive in later steps, each as its own module.
"""
