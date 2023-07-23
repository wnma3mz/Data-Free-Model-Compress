# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from criterion import KDLoss
from tqdm import tqdm
from utils import sample_noises

from trainer.Trainer import Trainer


# https://github.com/VainF/Data-Free-Adversarial-Distillation/tree/master
class DFADTrainer(Trainer):
    # 借助Teacher训练GAN，GAN生成数据再分别输入至教师和学生，进行度量。online training
    def __init__(
        self, model, optimizer, criterion, device, display=True, teacher=None, generator=None, optimizer_G=None
    ):
        super().__init__(model, optimizer, criterion, device, display)
        self.generator, self.teacher = generator.to(self.device), teacher.to(self.device)
        self.optimizer_G = optimizer_G
        self.kd_criterion = KDLoss() # F.l1_loss

    def zero_train(self, epochs, g_epochs, d_epochs, batch_size, nz):
        generator, teacher, student = self.generator, self.teacher, self.model
        optimizer_S, optimizer_G = self.optimizer, self.optimizer_G
        teacher.eval(), student.train(), generator.train()

        size = batch_size, nz, 1, 1
        
        pbar = tqdm(range(epochs), ncols=80, postfix="loss_G: *.****; loss_S: *.****")
        loop_loss_S, loop_loss_G = [], []
        for _ in pbar:
            for _ in range(g_epochs):
                z = sample_noises(size).to(self.device) 
                fake = generator(z).detach()
                with torch.no_grad():
                    t_logit = teacher(fake)
                s_logit = student(fake)
                loss_S = self.kd_criterion(s_logit, t_logit.detach())

                optimizer_S.zero_grad()
                loss_S.backward()
                optimizer_S.step()

            for _ in range(d_epochs):
                z = sample_noises(size).to(self.device)
                fake = generator(z)
                with torch.no_grad():
                    t_logit = teacher(fake)
                s_logit = student(fake)

                loss_G = self.kd_criterion(s_logit, t_logit)

                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            loop_loss_G.append(loss_G.item())
            loop_loss_S.append(loss_S.item())
            pbar.postfix = "loss_G: {:.4f}; loss_S: {:.4f}".format(np.mean(loop_loss_G), np.mean(loop_loss_S))
