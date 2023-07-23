# -*- coding: utf-8 -*-

import numpy as np
import torch
from criterion import KDLoss
from tqdm import tqdm
from utils import sample_noises
from trainer.Trainer import Trainer

# https://github.com/Piyush-555/GaussianDistillation


class GaussianTrainer(Trainer):
    # 蒸馏时，教师也更新模型，以确保教师能在新的数据分布上（data-free）拟合数据。仅更新BN层即可，无需更新全部参数
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        display=True,
        teacher=None,
    ):
        super().__init__(model, optimizer, criterion, device, display)
        self.teacher = teacher.to(self.device)
        self.kd_criterion = KDLoss()
        self.batch_img_size = (128, 3, 32, 32)
        self.teacher = teacher

    def zero_train(self, epochs, len_batch):
        teacher, student = self.teacher, self.model
        teacher.train()
        student.train()

        pbar = tqdm(range(epochs), ncols=80, postfix="loss_S: *.****")
        loop_loss_S = []
        for _ in range(epochs):
            for _ in range(len_batch):
                data = sample_noises(self.batch_img_size).to(self.device)
                with torch.no_grad():
                    logits = teacher(data)
                output = student(data)
                loss = self.kd_criterion(output, logits)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loop_loss_S.append(loss.item())
            pbar.postfix = "loss_S: {:.4f}".format(np.mean(loop_loss_S))
