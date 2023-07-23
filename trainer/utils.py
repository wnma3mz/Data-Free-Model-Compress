import torch
import numpy as np


def sample_noises(size):
    """
    Sample noise vectors (z).
    """
    return torch.randn(size)


def sample_labels(num_data, num_classes, dist):
    """
    Sample label vectors (y).
    """
    if dist == "onehot":
        init_labels = np.random.randint(0, num_classes, num_data)
        labels = np.zeros((num_data, num_classes), dtype=int)
        labels[np.arange(num_data), init_labels] = 1
        return torch.tensor(labels, dtype=torch.float32)
    elif dist == "uniform":
        labels = np.random.uniform(size=(num_data, num_classes))
        return torch.tensor(labels, dtype=torch.float32)
    else:
        raise ValueError(dist)
