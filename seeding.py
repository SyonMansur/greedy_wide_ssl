# keep track of seeds for the run, model weights, and data loading

import hashlib
import random

import numpy as np
import torch


def seed_everything(seed):
    # start the random number generators from the chosen run seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(*parts):
    # turn the run details into a seed that stays the same across python sessions
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def model_seed(kind, settings, *parts):
    # keep the original cifar-10 seed format; other datasets also include their name
    dataset = [] if settings["dataset"] == "cifar10" else [settings["dataset"]]
    return stable_seed(
        kind, *dataset, settings["seed"], settings["depth"], f"{settings['width']:g}", *parts
    )


def seed_worker(_worker_id):
    # pytorch gives each worker a seed; use it for python and numpy too
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
