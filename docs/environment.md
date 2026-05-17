# Environment

Use a Python 3.10+ environment with a GPU-compatible PyTorch installation for model training and inference.

## Install

```bash
pip install -e .
```

For development utilities:

```bash
pip install -e ".[dev]"
```

For future audio experiments:

```bash
pip install -e ".[audio]"
```

Keep local environment names, interpreter paths, CUDA paths, dataset roots, and checkpoint roots out of committed files. Put machine-specific settings in local shell profiles, untracked `.env` files, or user-level IDE settings.
