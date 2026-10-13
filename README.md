# Increasing Width Allows Greedy Layer-wise Training to Rival End-to-End Backpropagation in Self-Supervised Learning

Code for **Increasing Width Allows Greedy Layer-Wise Training to Rival End-to-End Backpropagation in Self-Supervised Learning**, by Syon Mansur and Joel Zylberberg.

https://arxiv.org/abs/2610.00753

Here, we investigate how increasing network width affects greedy layer-wise self-supervised learning in relatively shallow networks. We show that sufficiently wide networks trained one layer at a time can rival end-to-end backpropagation, using Barlow Twins and SimCLR. The code includes training, kNN evaluation, and representational geometry analysis.

Feel free to reach out to syonmansur@g.ucla.edu with any questions! Full instructions to run all paper experiments are included below:

## Running / Reproduction Instructions:

Use Python 3.10+ with a compatible PyTorch/torchvision installation.

```bash
python -m pip install -r requirements.txt
python train.py
```

**Edit `OBJECTIVE` and `SETTINGS` at the top of `train.py` before running.** Set `OBJECTIVE = "barlow_twins"` (default) or `OBJECTIVE = "simclr"`. Both use the same training loop; `simclr.py` is an imported module.

In `SETTINGS`, choose the training method (`"e2e"` or `"greedy"`), dataset (`"cifar10"` or `"cifar100"`), depth, width, seed, and output directory. Objective-specific hyperparameters are in `BARLOW_DEFAULTS` in `train.py` and `DEFAULTS` in `simclr.py`; both contain the selected paper values. Entries in `SETTINGS` override defaults. `epochs` is the total budget: 1,000 epochs means 250 per greedy Conv4 layer or 125 per Conv8 layer.

Data automatically downloads to `data/`. Each run needs a new output directory and saves `configuration.json`, `history.csv` (loss and test kNN accuracy), and `final_encoder.pt`.

## Geometry

Set `geometry` to `"final"` or `"trajectory"` in `train.py` to save SSF/SNF measurements to `geometry.csv`. For a saved checkpoint, edit the paths at the top of `geometry.py`, then run:

```bash
python geometry.py
```

Geometry uses the paper's covariance equations on L2-normalized GAP features from all test images, with global centering and whitened class contrasts.

## Files

- `train.py`: experiment settings, Barlow Twins definitions, and the single training loop.
- `simclr.py`: SimCLR defaults, projector, loss, and combined-view handling.
- `architecture.py`: convolutional encoder and its initialization.
- `data.py`: CIFAR datasets, augmentations, and data loaders.
- `evaluation.py`: kNN accuracy.
- `seeding.py`: seeds for runs, models, and data-loader workers.
- `geometry.py`: SSF/SNF equations and checkpoint analysis.

The objective selects its projector and loss calculation once, before training. Greedy training freezes earlier weights and BatchNorm statistics and uses a fresh projector at every layer. Barlow Twins processes views separately; SimCLR combines them for BatchNorm. Data loading uses saturation jitter 0.2 for Barlow Twins and 0.4 for SimCLR.

For the paper's Barlow Twins CIFAR-10 runs, use both methods, seeds 0–2, and widths 0.25, 0.5, 1, 2, 4, 8, 16 (plus 32 for Conv4). SimCLR/CIFAR-10 and Barlow Twins/CIFAR-100 use Conv4, all eight widths, and seed 1. Temporal geometry uses separate Conv4 32× seed-1 runs. The Barlow Twins cosine-schedule check is available through `cosine_decay`. Seeded CUDA runs are not guaranteed bitwise identical across hardware or software versions.
