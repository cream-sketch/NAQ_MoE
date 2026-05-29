# Results

This directory stores lightweight summaries that are safe to keep in Git.

Tracked examples:

- `realtext_512/EXPERIMENT_RESULTS.md`
- `summaries/*.json`

Not tracked:

- activation tensors (`*.pt`),
- alignment arrays and dispersion arrays (`*.npy`),
- reconstructed weights,
- Hugging Face model checkpoints,
- offload/cache folders.

Those artifacts are generated under `outputs*/` by the run scripts.

