from typing import Any, Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST


class FewCIFAR10(CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        if self.train:
            max_cnt = 5
            class_cnt = {k: 0 for k in set(self.targets)}
            few_data, few_targets = [], []
            for img, target in zip(self.data, self.targets):
                if class_cnt[target] < max_cnt:
                    few_data.append(img)
                    few_targets.append(target)
                    class_cnt[target] += 1
            self.data, self.targets = few_data, few_targets


class FewCIFAR100(CIFAR100):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        if self.train:
            max_cnt = 5
            class_cnt = {k: 0 for k in set(self.targets)}
            few_data, few_targets = [], []
            for img, target in zip(self.data, self.targets):
                if class_cnt[target] < max_cnt:
                    few_data.append(img)
                    few_targets.append(target)
                    class_cnt[target] += 1
            self.data, self.targets = few_data, few_targets


def get_datasets(dataset_name, dataset_dir, split=None, download=True):
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trans_emnist = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    trains_cifar10 = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trains_cifar100 = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    trans_cifar_train = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            trains_cifar10,
        ]
    )
    trans_cifar_test = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), trains_cifar10])

    if dataset_name == "mnist":
        trainset = MNIST(dataset_dir, train=True, download=download, transform=trans_mnist)
        testset = MNIST(dataset_dir, train=False, download=download, transform=trans_mnist)
    elif dataset_name == "cifar10":
        trainset = CIFAR10(dataset_dir, train=True, download=download, transform=trans_cifar_train)
        testset = CIFAR10(dataset_dir, train=False, download=download, transform=trans_cifar_test)
    elif dataset_name == "few_cifar10":
        trainset = FewCIFAR10(dataset_dir, train=True, download=download, transform=trans_cifar_train)
        testset = FewCIFAR10(dataset_dir, train=False, download=download, transform=trans_cifar_test)
    elif dataset_name == "cifar100":
        trainset = CIFAR100(dataset_dir, train=True, download=download, transform=trans_cifar_train)
        testset = CIFAR100(dataset_dir, train=False, download=download, transform=trans_cifar_test)
    elif dataset_name == "few_cifar100":
        trainset = FewCIFAR100(dataset_dir, train=True, download=download, transform=trans_cifar_train)
        testset = FewCIFAR100(dataset_dir, train=False, download=download, transform=trans_cifar_test)
    elif dataset_name == "emnist":
        trainset = EMNIST(dataset_dir, train=True, download=download, split=split, transform=trans_emnist)
        testset = EMNIST(
            dataset_dir,
            train=False,
            download=download,
            split=split,
            transform=trans_emnist,
        )
    else:
        raise NotImplementedError("This dataset is not currently supported")
    return trainset, testset


def get_dataloader(trainset, testset, batch_size, num_workers=0, pin_memory=False):
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return trainloader, testloader
