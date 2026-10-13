# choose the objective and settings below, then run python train.py

import csv
import json
from pathlib import Path

import torch
from torch import nn

import simclr
from architecture import make_encoder
from data import make_loaders
from evaluation import knn_accuracy
from geometry import measure_geometry
from seeding import model_seed, seed_everything


OBJECTIVE = "barlow_twins"  # "barlow_twins" or "simclr"

# main settings shared by both objectives
SETTINGS = {
    "method": "greedy",  # "e2e" or "greedy"
    "dataset": "cifar10",  # "cifar10" or "cifar100"
    "depth": 4,  # 4 or 8 convolutional layers
    "width": 1.0,  # multiply every layer's channel count by this
    "seed": 1 if OBJECTIVE == "simclr" else 0,
    "epochs": 1000,  # total budget, split equally across greedy layers
    "batch_size": 512,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "crop_min": 0.2,  # smallest crop as a fraction of the image area
    "eval_every": 20,
    "eval_batch_size": 512,
    "knn_k": 200,  # number of neighbors voting on each test image
    "geometry": "none",  # "none", "final", or "trajectory"
    "cosine_decay": False,
    "minimum_learning_rate": 1e-6,
    "workers": 4,
    "data_dir": "data",
    "output_dir": "results/simclr" if OBJECTIVE == "simclr" else "results/barlow",
}

# barlow twins defaults; the simclr ones are at the top of simclr.py
BARLOW_DEFAULTS = {"projector_dim": 256, "lambd": 0.1632, "loss_epsilon": 1e-5}


class BarlowProjector(nn.Module):
    def __init__(self, input_dim, q):
        super().__init__()
        # map the pooled encoder features to the space used by the loss
        self.net = nn.Sequential(
            nn.Linear(input_dim, q, bias=False),
            nn.BatchNorm1d(q),
            nn.ReLU(inplace=True),
            nn.Linear(q, q, bias=False),
        )

    def forward(self, features):
        return self.net(features)


def barlow_loss(first, second, lambd, epsilon):
    # standardize each feature across the batch, separately for the two views
    first = (first - first.mean(0)) / (first.var(0, unbiased=False) + epsilon).sqrt()
    second = (second - second.mean(0)) / (second.var(0, unbiased=False) + epsilon).sqrt()
    correlation = first.T @ second / len(first)
    # match the same feature across views and discourage overlap between features
    invariance = (correlation.diagonal() - 1).square().sum()
    off_diagonal = ~torch.eye(correlation.shape[0], dtype=torch.bool, device=first.device)
    redundancy = correlation[off_diagonal].square().sum()
    return invariance + lambd * redundancy


def make_barlow_projector(input_dim, layer, settings, device):
    # start the final greedy head and the end-to-end head from the same weights
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed("projector", settings, layer, settings["projector_dim"]))
        projector = BarlowProjector(input_dim, settings["projector_dim"])
    return projector.to(device)


def barlow_batch_loss(encoder, projector, first, second, layer, greedy, settings):
    # pass the views through separately so batch norm sees one view at a time
    first = projector(encoder.features(first, layer, greedy))
    second = projector(encoder.features(second, layer, greedy))
    return barlow_loss(first, second, settings["lambd"], settings["loss_epsilon"])


def setup(settings):
    # check the training mode, batch size, and total epoch budget before starting
    assert settings["method"] in ("e2e", "greedy")
    assert settings["epochs"] > 0 and settings["batch_size"] >= 2
    if settings["method"] == "greedy":
        assert settings["epochs"] % settings["depth"] == 0, (
            "Epochs must divide evenly across layers"
        )
    # use the original runs' seed and gpu precision settings
    seed_everything(settings["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(False)
    # save the settings in a new directory so an earlier run doesn't get overwritten
    output = Path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    (output / "configuration.json").write_text(json.dumps(settings, indent=2) + "\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, output


def write_rows(path, rows):
    # append the new rows and add a header only when the file is first created
    path = Path(path)
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def train(settings=SETTINGS, objective=OBJECTIVE):
    # pick the projector and loss once, then use the same training loop for both
    if objective == "barlow_twins":
        defaults = BARLOW_DEFAULTS
        make_projector = make_barlow_projector
        batch_loss = barlow_batch_loss
    elif objective == "simclr":
        defaults = simclr.DEFAULTS
        make_projector = simclr.make_projector
        batch_loss = simclr.batch_loss
    else:
        raise ValueError("OBJECTIVE must be 'barlow_twins' or 'simclr'")
    # settings entered above take priority over the objective's defaults
    settings = {**defaults, **settings, "objective": objective}
    assert settings["geometry"] in ("none", "final", "trajectory")
    device, output = setup(settings)
    train_loader, memory_loader, test_loader = make_loaders(settings, objective)
    encoder = make_encoder(settings, device)
    # greedy runs visit each layer in turn; end-to-end runs train the whole encoder
    greedy = settings["method"] == "greedy"
    stages = range(1, settings["depth"] + 1) if greedy else [settings["depth"]]
    epochs_per_stage = settings["epochs"] // settings["depth"] if greedy else settings["epochs"]
    global_epoch = 0

    for layer in stages:
        # each stage starts with a fresh projector and optimizer
        projector = make_projector(encoder.channels[layer - 1], layer, settings, device)
        if greedy:
            # freeze the weights and batch norm statistics, then enable the active layer
            encoder.eval().requires_grad_(False)
            encoder.layers[layer - 1].train().requires_grad_(True)
        else:
            encoder.train()
        # update the active encoder weights together with the projector
        parameters = [p for p in encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            parameters + list(projector.parameters()),
            lr=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
        )
        # optionally decay the learning rate over this stage's epoch budget
        scheduler = None
        if settings["cosine_decay"]:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, epochs_per_stage, eta_min=settings["minimum_learning_rate"]
            )

        for epoch in range(1, epochs_per_stage + 1):
            global_epoch += 1
            total_loss = torch.zeros((), device=device)
            examples = 0
            for (first, second), _labels in train_loader:
                # train on the two augmented views; class labels aren't used here
                first, second = first.to(device), second.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = batch_loss(encoder, projector, first, second, layer, greedy, settings)
                loss.backward()
                optimizer.step()
                # track the average loss per image without keeping the gradient graph
                total_loss.add_(loss.detach(), alpha=len(first))
                examples += len(first)

            # evaluate at the chosen interval and at the end of every stage
            accuracy = None
            evaluate = global_epoch % settings["eval_every"] == 0 or epoch == epochs_per_stage
            if evaluate:
                accuracy = knn_accuracy(
                    encoder, memory_loader, test_loader, device, settings["knn_k"], layer
                )
                print(f"Epoch {global_epoch}, layer {layer}: kNN {accuracy:.2f}%", flush=True)
                # measure geometry at each evaluation or just once at the end
                if settings["geometry"] == "trajectory" or (
                    settings["geometry"] == "final" and global_epoch == settings["epochs"]
                ):
                    rows = measure_geometry(encoder, test_loader, device, layer)
                    rows = [
                        dict(global_epoch=global_epoch, active_layer=layer, **row) for row in rows
                    ]
                    write_rows(output / "geometry.csv", rows)
            # save one history row per epoch, leaving accuracy blank between evaluations
            write_rows(
                output / "history.csv",
                [
                    {
                        "global_epoch": global_epoch,
                        "active_layer": layer,
                        "loss": (total_loss / examples).item(),
                        "test_knn_accuracy": accuracy,
                    }
                ],
            )
            if scheduler is not None:
                scheduler.step()
        # the projector is discarded after this stage

    # save the encoder weights and run settings for later evaluation or geometry analysis
    torch.save(
        {
            "configuration": settings,
            "state_dict": {name: value.cpu() for name, value in encoder.state_dict().items()},
        },
        output / "final_encoder.pt",
    )
    return encoder


if __name__ == "__main__":
    train()
