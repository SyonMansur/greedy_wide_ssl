# load cifar images and set up the training and evaluation views

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from seeding import seed_worker, stable_seed


# dataset classes and their channel means and standard deviations
CIFAR = {
    "cifar10": (datasets.CIFAR10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": (datasets.CIFAR100, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}


class TwoViews:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image):
        # augment the same image twice with separate random draws
        return self.transform(image), self.transform(image)


def make_loaders(settings, objective):
    dataset_class, mean, std = CIFAR[settings["dataset"]]
    # simclr uses stronger saturation jitter than the barlow twins runs
    saturation = 0.4 if objective == "simclr" else 0.2
    # crop, flip, and change the colors before normalizing each training view
    augmentation = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                32,
                scale=(settings["crop_min"], 1.0),
                ratio=(0.75, 4 / 3),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, saturation, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # keep evaluation images unchanged apart from conversion and normalization
    evaluation = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = dataset_class(
        settings["data_dir"], train=True, download=True, transform=TwoViews(augmentation)
    )
    # the training images also form the labeled reference bank for knn evaluation
    memory = dataset_class(settings["data_dir"], train=True, transform=evaluation)
    test = dataset_class(settings["data_dir"], train=False, transform=evaluation)
    loaders = []
    # keep the loader seed names used in the original experiments
    test_tag = "evaluation-loader" if objective == "simclr" else "test-loader"
    for dataset, tag in ((train, "train-loader"), (memory, "memory-loader"), (test, test_tag)):
        training = tag == "train-loader"
        # shuffle and drop incomplete batches only for training; evaluate every image
        loaders.append(
            DataLoader(
                dataset,
                batch_size=settings["batch_size"] if training else settings["eval_batch_size"],
                shuffle=training,
                drop_last=training,
                num_workers=settings["workers"],
                persistent_workers=training and settings["workers"] > 0,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=torch.Generator().manual_seed(stable_seed(tag, settings["seed"])),
            )
        )
    if len(loaders[0]) == 0:
        raise ValueError("batch_size exceeds the training dataset size")
    return loaders


def make_test_loader(dataset, data_dir, batch_size=512, workers=4):
    # load just the test images when analyzing an existing checkpoint
    dataset_class, mean, std = CIFAR[dataset]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    test = dataset_class(data_dir, train=False, download=True, transform=transform)
    return DataLoader(
        test,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
