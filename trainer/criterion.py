# coding: utf-8
import torch.nn as nn
import torch


class KDLoss(nn.Module):
    def __init__(self, temp=1, reduction="batchmean"):
        super().__init__()
        self.kl_div = nn.KLDivLoss(reduction=reduction)
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.temp = temp

    def forward(self, output, target):
        """
        Forward propagation.
        """
        return self.kl_div(self.log_softmax(output / self.temp), self.softmax(target / self.temp)) * (self.temp**2)


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == "l1":
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == "l2":
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == "cosine":
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how="l2")
        return torch.exp(torch.mean(-noise_dist * layer_dist))
