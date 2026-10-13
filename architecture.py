# encoder used by both barlow twins and simclr

import torch
from torch import nn

from seeding import model_seed


# base channel counts and downsampling steps for the two network depths
CHANNELS = {4: (32, 64, 128, 256), 8: (32, 32, 64, 64, 128, 128, 256, 256)}
STRIDES = {4: (1, 2, 2, 2), 8: (1, 1, 2, 1, 2, 1, 2, 1)}


class Encoder(nn.Module):
    # stack convolution, batch norm, and relu, then pool the spatial dimensions

    def __init__(self, depth, width):
        super().__init__()
        self.depth = depth
        # widen every layer by the same multiplier
        self.channels = [round(c * width) for c in CHANNELS[depth]]
        self.layers = nn.ModuleList()
        input_channels = 3
        for channels, stride in zip(self.channels, STRIDES[depth]):
            self.layers.append(
                nn.Sequential(
                    nn.Conv2d(input_channels, channels, 3, stride, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                )
            )
            input_channels = channels

    def features(self, images, stop_layer=None, greedy=False):
        # stop at the active layer during greedy training, or use the full network
        stop_layer = self.depth if stop_layer is None else stop_layer
        maps = images
        if greedy:
            # pass through the frozen layers without tracking their gradients
            with torch.no_grad():
                for layer in self.layers[: stop_layer - 1]:
                    maps = layer(maps)
            # track gradients for the layer we're training now
            maps = self.layers[stop_layer - 1](maps)
        else:
            for layer in self.layers[:stop_layer]:
                maps = layer(maps)
        # average each feature map into one value per channel
        return maps.mean(dim=(2, 3))


def make_encoder(settings, device):
    # seed the weights separately and restore the cpu random state afterward
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model_seed("encoder", settings))
        encoder = Encoder(settings["depth"], settings["width"])
    return encoder.to(device)
