# simclr projector and loss, selected by the objective setting in train.py

import torch
from torch import nn
from torch.nn import functional as F

from seeding import model_seed


# settings used for the simclr runs
DEFAULTS = {
    "projector_hidden_dim": 2048,
    "projection_dim": 128,
    "temperature": 0.5,
}


class ScaleOnlyBatchNorm(nn.Module):
    # normalize the output and learn its scale, with no learned offset

    def __init__(self, features):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(features, affine=False)
        self.scale = nn.Parameter(torch.ones(features))

    def forward(self, inputs):
        return self.batch_norm(inputs) * self.scale


class Projector(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # expand the encoder features, then map them to the contrastive embedding
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
            ScaleOnlyBatchNorm(output_dim),
        )
        # use the same small random linear weights as the original simclr runs
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

    def forward(self, features):
        return self.net(features)


def simclr_loss(first, second, temperature):
    # put both views in one batch and use normalized dot products for similarity
    first, second = F.normalize(first, dim=1), F.normalize(second, dim=1)
    views = torch.cat((first, second))
    logits = views @ views.T / temperature
    logits.fill_diagonal_(-torch.finfo(logits.dtype).max)  # don't match an image to itself
    # the matching view is in the other half of the combined batch
    positives = (torch.arange(len(views), device=views.device) + len(first)) % len(views)
    # each view predicts its partner among all the other views
    return F.cross_entropy(logits, positives, reduction="none").mean()


def make_projector(input_dim, layer, settings, device):
    # seed the head from the run settings so paired methods get matching starting weights
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            model_seed(
                "projector",
                settings,
                layer,
                settings["projection_dim"],
                settings["projector_hidden_dim"],
            )
        )
        projector = Projector(
            input_dim, settings["projector_hidden_dim"], settings["projection_dim"]
        )
    return projector.to(device)


def batch_loss(encoder, projector, first, second, layer, greedy, settings):
    # combine the views so batch norm sees both halves of the batch together
    views = torch.cat((first, second))
    projected = projector(encoder.features(views, layer, greedy))
    first_projection, second_projection = projected.chunk(2)
    return simclr_loss(first_projection, second_projection, settings["temperature"])
