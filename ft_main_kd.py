# coding: utf-8

import random
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet152_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet34,
    resnet152,
    vit_b_16,
)
from Trainer import Trainer

from datasets import get_dataloader, get_datasets
from imprint import imprint, weight_norm
from models import CNN, MLP


def setup_seed(seed):
    """设置随机种子，以便于复现实验"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


class MyTrainer(Trainer):
    def train(self, data_loader, epochs=1):
        self.model.train()
        self.is_train = True
        for _ in range(1, epochs + 1):
            with torch.enable_grad():
                loss, accuracy = self._iteration(data_loader)

            # self.model.fc.weight.data = weight_norm(self.model.fc.weight.data)
            self.history_loss.append(loss)
            self.history_accuracy.append(accuracy)
        return loss, accuracy


def distill_loss(student, teacher, T=1):
    student = F.log_softmax(student / T, dim=-1)
    teacher = (teacher / T).softmax(dim=-1)

    try:
        return -(teacher * student).sum(dim=1).mean()
    except:
        import pdb

        pdb.set_trace()


class DistillTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, display=True, t_model=None):
        super().__init__(model, optimizer, criterion, device, display)
        self.t_model = t_model.to(self.device)

    def batch(self, data, target):
        # data = torch.randn_like(data, device=self.device) if self.is_train and self.t_model else data
        output = self.model(data)
        if self.is_train and self.t_model:
            with torch.no_grad():
                t_output = self.t_model(data)
            # loss = self.criterion(output, target) + distill_loss(output, t_output)
            loss = distill_loss(output, t_output)
        else:
            loss = self.criterion(output, target)

        if self.is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        iter_loss = loss.data.item()
        iter_acc = self.metrics(output, target)
        return iter_loss, iter_acc

    def train(self, data_loader, epochs=1):
        if self.t_model:
            # for name, params in self.t_model.named_parameters():
            #     if "bn" in name:
            #         params.requires_grad = True
            #     else:
            #         params.requires_grad = False
            self.t_model.eval()
            # self.t_model.train()
        return super().train(data_loader, epochs)


if __name__ == "__main__":
    num_class = 10
    # dataset_name, dataset_fpath, batch_size = f"few_cifar{num_class}", "~/.keras/datasets", 16
    dataset_name, dataset_fpath, batch_size = f"cifar{num_class}", "~/.keras/datasets", 16
    trainset, testset = get_datasets(dataset_name, dataset_fpath)
    trainloader, testloader = get_dataloader(
        trainset, testset, num_workers=4, batch_size=batch_size, pin_memory=False
    )
    # assert 1==2

    # model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # feat_num = model.hidden_dim
    # model.heads = nn.Identity()
    # weight = imprint(trainloader, model, num_class, "cuda:0", False, feat_num)
    # model.heads = nn.Linear(feat_num, num_class, bias=False)
    # model.heads.weight.data = weight
    # model.heads = Proj(model.hidden_dim, 10)

    t_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feat_num = t_model.fc.in_features
    t_model.fc = nn.Identity()
    weight = imprint(trainloader, t_model, num_class, "cuda:0", False, feat_num)
    t_model.fc = nn.Linear(feat_num, num_class, bias=False)
    t_model.fc.weight.data = weight

    # t_model = resnet152(weights=ResNet152_Weights.DEFAULT)
    model = resnet18()
    model.fc = nn.Linear(feat_num, num_class, bias=False)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = DistillTrainer(model, optimizer, criterion, device="cuda:0", display=False, t_model=t_model)
    s1 = time.time()
    loss, accuracy = trainer.train(trainloader, epochs=5)
    e1 = time.time()
    print("训练耗时: {}".format(e1 - s1), loss, accuracy)

    feat_num = model.fc.in_features
    model.fc = nn.Identity()
    weight = imprint(trainloader, model, num_class, "cuda:0", False, feat_num)
    model.fc = nn.Linear(feat_num, num_class, bias=False)
    model.fc.weight.data = weight
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cuda:0")

    s2 = time.time()
    loss, accuracy = trainer.test(testloader)
    e2 = time.time()
    print("测试耗时: {}".format(e2 - s2), loss, accuracy)

    assert 1 == 2

    for name, params in model.named_parameters():
        if any(x in name for x in ["fc", "heads"]):
            params.requires_grad = False
        else:
            params.requires_grad = True
        # print(name, params.requires_grad)

    # optimizer = optim.Adam(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, criterion, device="cuda:0")

    s2 = time.time()
    loss, accuracy = trainer.test(testloader)
    e2 = time.time()
    print("测试耗时: {}".format(e2 - s2), loss, accuracy)

    # s1 = time.time()
    # loss, accuracy = trainer.train(trainloader, epochs=10)
    # e1 = time.time()
    # print("训练耗时: {}".format(e1 - s1), loss, accuracy)

    # s2 = time.time()
    # loss, accuracy = trainer.test(testloader)
    # e2 = time.time()
    # print("测试耗时: {}".format(e2 - s2), loss, accuracy)
