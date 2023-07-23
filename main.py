import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ViT_B_16_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet152,
    vit_b_16,
)

from datasets import get_dataloader, get_datasets
from models.generator import Decoder, GeneratorA, ImageGenerator, ImageGenerator2, conv3_gen
from trainer.DFADTrainer import DFADTrainer
from trainer.DFMETrainer import DFMETrainer
from trainer.GaussianTrainer import GaussianTrainer
from trainer.InversionTrainer import InversionTrainer
from trainer.KegTrainer import KegTrainer
from trainer.MAZETrainer import MAZETrainer
from trainer.ZSKDTrainer import ZSKDTrainer


@dataclass
class HyperP:
    batch_size: int = field(default=512)
    epochs: int = field(default=256)
    weight_decay: float = field(default=1e-4)
    epoch_itrs: int = field(default=50)

    momentum: float = field(default=0.9)

    ckpt: str = field(default="ckpt/model.pth")
    log_interval: int = field(default=10)
    step_size: int = field(default=100)
    scheduler: bool = field(default=False)
    verbose: bool = field(default=True)

    num_class: int = 10

    device: str = torch.device("cuda:0")

    def __post_init__(self):
        teacher = resnet18(weights=ResNet18_Weights.DEFAULT).to(self.device)
        feat_num = teacher.fc.in_features
        teacher.fc = nn.Linear(feat_num, self.num_class, bias=False)
        teacher.fc.load_state_dict(torch.load("./resnet18_cifar10_fc.pth"))
        teacher.to(self.device)

        student = resnet18().to(self.device)
        student.fc = nn.Linear(feat_num, self.num_class, bias=False)

        self.teacher, self.student = teacher, student

        dataset_name, dataset_fpath, batch_size = (
            f"few_cifar{self.num_class}",
            "~/.keras/datasets",
            16,
        )
        # dataset_name, dataset_fpath, batch_size = f"cifar{num_class}", "~/.keras/datasets", 16
        trainset, testset = get_datasets(dataset_name, dataset_fpath)
        _, test_loader = get_dataloader(trainset, testset, num_workers=4, batch_size=batch_size, pin_memory=False)
        self.test_loader = test_loader


args = HyperP()


def setup_seed(seed):
    """设置随机种子，以便于复现实验"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # tf.random.set_seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(0)


def get_class_similarity(model):
    t_weights = model.fc.state_dict()["weight"]
    # Compute concentration parameter
    t_weights_norm = F.normalize(t_weights, p=2, dim=1)
    cls_sim = torch.matmul(t_weights_norm, t_weights_norm.T)
    cls_sim_norm = torch.div(
        cls_sim - torch.min(cls_sim, dim=1).values,
        torch.max(cls_sim, dim=1).values - torch.min(cls_sim, dim=1).values,
    )
    return cls_sim_norm


if __name__ == "__main__":
    device = args.device
    os.makedirs("checkpoint/student", exist_ok=True)
    print(args)
    model = args.student
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()

    method = ...
    if method == "DFME":
        G_activation = torch.tanh
        nz = 256
        generator = GeneratorA(nz=nz, nc=3, img_size=32, activation=G_activation)

        optG = optim.Adam(generator.parameters(), lr=1e-2)
        trainer = DFMETrainer(
            model,
            optimizer,
            criterion,
            device,
            False,
            args.teacher,
            generator=generator,
            optimizer_G=optG,
        )
        trainer.zero_train(epoch_itrs=50, g_iter=1, d_iter=5, batch_size=256, nz=nz)
    elif method == "MAZE":
        latent_dim = 100
        generator = conv3_gen(z_dim=latent_dim, start_dim=8, out_channels=3)
        optG = optim.Adam(generator.parameters(), lr=1e-2)
        trainer = MAZETrainer(
            model,
            optimizer,
            criterion,
            device,
            False,
            args.teacher,
            generator=generator,
            optG=optG,
        )
        trainer.zero_train(
            batch_size=128,
            iter_clone=5,
            ndirs=10,
            iter_gen=1,
            budget=5e6,  # "Query Budget for Attack"
            latent_dim=100,
            mu=0.001,
            iter_exp=10,
        )
    elif method == "ZSKD":
        cls_sim_norm = get_class_similarity(args.teacher)
        trainer = ZSKDTrainer(model, optimizer, criterion, device, False, args.teacher)
        fake_traindata_loader = trainer.generate_data(
            cls_sim_norm,
            num_sample=24000,
            batch_size=128,
            lr=1e-2,
            iters=1500,
            beta=[0.1, 1],
            temp=20,
        )
        trainer.zero_train(fake_traindata_loader)
    elif method == "Keg":
        generator = ImageGenerator(num_classes=10, num_channels=3)
        decoder = Decoder(in_features=3072, out_targets=10, n_layers=3)
        optimizer_gc = optim.Adam(list(generator.parameters()) + list(decoder.parameters()), lr=1e-2)
        trainer = KegTrainer(
            model,
            optimizer,
            criterion,
            device,
            False,
            args.teacher,
            generator,
            decoder,
            optimizer_gc,
        )
        trainer.train_generator(100, 128, 1, 1)
        fake_traindata_loader = trainer.generate_data(10000)
        trainer.zero_train(fake_traindata_loader)
    elif method == "Inverion":
        trainer = InversionTrainer(args.student, None, criterion, args.device, teacher=args.teacher)
        import copy

        trainer.zero_train(verify_model=copy.deepcopy(args.teacher))
    elif method == "DFAD":
        nz = 256
        generator = ImageGenerator2(nz=nz, nc=3)
        optimizer_G = optim.Adam(generator.parameters(), lr=1e-3)

        trainer = DFADTrainer(
            model,
            optimizer,
            criterion,
            args.device,
            display=False,
            teacher=args.teacher,
            generator=generator,
            optimizer_G=optimizer_G,
        )
        trainer.zero_train(args.epoch_itrs, args.batch_size, args.nz)
        _, acc = trainer.test(args.test_loader)
        print(acc)
