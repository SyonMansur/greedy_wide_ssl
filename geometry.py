# class geometry of normalized, pooled encoder features from paper sections 3.3 and a.3

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# set these paths for an existing checkpoint, then run python geometry.py
CHECKPOINT = "results/barlow/final_encoder.pt"
DATA_DIR = "data"
OUTPUT = "results/geometry.csv"
BATCH_SIZE = 512
WORKERS = 4


def class_geometry(normalized_features, labels):
    # get ssf and snf from the paper's psi, omega, and phi equations
    # pass in l2-normalized feature rows; we'll center them below
    # the paper uses all 10,000 cifar-10 test images, balanced across ten classes
    # use float64 and divide all moments by the number of images
    X = np.array(normalized_features, dtype=np.float64, copy=True)
    labels = np.asarray(labels)
    if not np.isfinite(X).all() or not np.isfinite(labels).all():
        raise ValueError("Features and class labels must be finite")
    classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    K, N = len(classes), len(X)
    if K < 2 or counts.min() < 2 or np.any(counts != counts[0]):
        raise ValueError("Need >=2 balanced classes and >=2 images per class")
    # subtract a shared offset first to make centering more numerically stable
    X -= X[0].copy()
    X -= X.mean(axis=0)

    # turn the centered class labels into orthonormal contrasts with unit variance
    # cifar-10 gives nine columns, so omega is the identity matrix
    contrasts = np.zeros((K, K - 1))
    for j in range(K - 1):
        scale = np.sqrt((j + 1) * (j + 2))
        contrasts[: j + 1, j] = 1 / scale
        contrasts[j + 1, j] = -(j + 1) / scale
    Z = np.sqrt(K) * contrasts[inverse]
    # omega is the label covariance; phi links the features to those labels
    Omega = Z.T @ Z / N
    Phi = X.T @ Z / N
    phi_gram = Phi.T @ Phi
    solved_gram = np.linalg.solve(Omega, phi_gram)
    signal = np.trace(phi_gram)
    # get the trace of the feature covariance without building the full matrix
    psi_trace = np.einsum("ij,ij->", X, X) / N
    # the ratios aren't meaningful once the class signal is effectively zero
    if psi_trace == 0 or signal <= 64 * np.finfo(float).eps * psi_trace:
        return {"ssf": float("nan"), "snf": float("nan")}

    # subtract the class-predicted part of the features, then project onto phi
    # this gives the paper's aligned-noise term without building the full
    # feature covariance or residual matrix, which get large at high widths
    residual_phi = X @ Phi - Z @ solved_gram
    aligned_noise = np.einsum("ij,ij->", residual_phi, residual_phi) / N
    # plug the signal and noise terms into the two ratios
    # nonzero signal with zero aligned noise gives infinite snf
    ssf = signal**2 / (np.trace(Omega) * np.trace(phi_gram @ solved_gram))
    snf = signal**2 / (np.trace(Omega) * aligned_noise) if aligned_noise > 0 else np.inf
    return {"ssf": float(ssf), "snf": float(snf)}


def measure_geometry(encoder, loader, device, stop_layer=None):
    # collect features from every trained layer using the full evaluation set
    stop_layer = encoder.depth if stop_layer is None else stop_layer
    if not 1 <= stop_layer <= encoder.depth:
        raise ValueError("stop_layer must select an existing encoder layer")
    # reserve space for one feature row per image at each layer
    features = [
        np.empty((len(loader.dataset), c), dtype=np.float32) for c in encoder.channels[:stop_layer]
    ]
    labels = np.empty(len(loader.dataset), dtype=np.int64)
    # save the layer modes and stop batch norm from updating during extraction
    modes = [(module, module.training) for module in encoder.modules()]
    encoder.eval()
    offset = 0
    with torch.inference_mode():
        for images, batch_labels in loader:
            maps = images.to(device, non_blocking=True)
            stop = offset + len(images)
            # reuse the feature maps as we move through the encoder
            for index, layer in enumerate(encoder.layers[:stop_layer]):
                maps = layer(maps)
                # pool after relu, then normalize each image's feature vector
                pooled = F.normalize(maps.mean(dim=(2, 3)), dim=1)
                features[index][offset:stop] = pooled.cpu().numpy()
            labels[offset:stop] = batch_labels.cpu().numpy()
            offset = stop
    # restore the original modes, including the frozen layers in a greedy run
    for module, training in modes:
        module.training = training
    if offset != len(labels):
        raise ValueError("Geometry loader must visit every image exactly once")
    # calculate ssf and snf separately for each layer's collected features
    return [
        {"layer": layer, **class_geometry(values, labels)}
        for layer, values in enumerate(features, 1)
    ]


if __name__ == "__main__":
    from architecture import Encoder
    from data import make_test_loader

    # rebuild the encoder from the saved run settings and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    settings = checkpoint["configuration"]
    encoder = Encoder(settings["depth"], settings["width"])
    encoder.load_state_dict(checkpoint["state_dict"])
    completed_layer = checkpoint.get("completed_layer", encoder.depth)
    del checkpoint
    encoder.to(device)
    # accept the dataset spelling used by both old and new checkpoints
    dataset = settings["dataset"].lower().replace("-", "")
    loader = make_test_loader(dataset, DATA_DIR, BATCH_SIZE, WORKERS)
    rows = measure_geometry(encoder, loader, device, completed_layer)
    # write one row per layer and leave any existing output file alone
    Path(OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    with Path(OUTPUT).open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {OUTPUT}")
