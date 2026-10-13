# check the encoder with an unweighted cosine knn classifier

import torch
from torch.nn import functional as F


@torch.inference_mode()
def knn_accuracy(encoder, memory_loader, test_loader, device, k, layer):
    # save each layer's mode so frozen layers stay frozen after evaluation
    modes = [(module, module.training) for module in encoder.modules()]
    encoder.eval()
    # build a bank of normalized training features and their class labels
    memory_features, memory_labels = [], []
    for images, labels in memory_loader:
        features = encoder.features(images.to(device), layer)
        memory_features.append(F.normalize(features, dim=1).cpu())
        memory_labels.append(labels)
    # put the bank on the device with one reference image per column
    memory_features = torch.cat(memory_features).T.contiguous().to(device)
    memory_labels = torch.cat(memory_labels).to(device)
    correct = total = 0
    for images, labels in test_loader:
        # normalized dot products give cosine similarity
        features = F.normalize(encoder.features(images.to(device), layer), dim=1)
        neighbors = (features @ memory_features).topk(k, dim=1).indices
        # each neighbor gets one vote; ties go to the lowest class label
        predictions = torch.mode(memory_labels[neighbors], dim=1).values.cpu()
        correct += (predictions == labels).sum().item()
        total += len(labels)
    # put every layer back in the mode it had before this call
    for module, mode in modes:
        module.training = mode
    return 100.0 * (correct / total)
