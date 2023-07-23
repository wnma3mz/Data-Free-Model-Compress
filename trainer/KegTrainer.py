# -*- coding: utf-8 -*-

import copy

import numpy as np
import torch
import torch.nn as nn
from criterion import KDLoss, DiversityLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import sample_labels, sample_noises
from trainer.Trainer import Trainer

# https://github.com/snudatalab/KegNet/


class KegTrainer(Trainer):
    # 借助Teacher训练若干GAN（多一个decoder的损失），GAN生成数据再分别输入至教师和学生，进行度量。online training
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        display=True,
        teacher=None,
        generator=None,
        decoder=None,
        optimizer_gc=None,
    ):
        super().__init__(model, optimizer, criterion, device, display)
        self.generator, self.teacher = generator.to(self.device), teacher.to(self.device)
        self.optimizer_gc = optimizer_gc
        self.decoder = decoder.to(self.device)

        self.div_criterion = DiversityLoss("l1")
        self.dec_criterion = nn.MSELoss()
        self.cls_criterion = nn.KLDivLoss(reduction="batchmean")

        self.criterion = KDLoss()
        self.kd_criterion = KDLoss()

    def zero_train(self, epochs):
        student = self.model
        student.train()

        pbar = tqdm(range(epochs), ncols=80, postfix="loss_S: *.****")
        loop_loss_S = []
        for _ in range(epochs):
            for input_, logits in self.fake_dataloader:
                output = student(input_.to(self.device))
                loss = self.kd_criterion(output, logits)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loop_loss_S.append(loss.item())
            pbar.postfix = "loss_S: {:.4f}".format(np.mean(loop_loss_S))

    @staticmethod
    def metrics(output, target):
        return (output.data.max(1)[1] == target.data.argmax(dim=1)).sum().item() / len(target) * 100

    def train_generator(self, epochs, batch_size, alpha, beta, num_generator=1):
        """
        if dataset == "mnist":
            dec_layers = 1
            lrn_rate = 1e-3
            alpha = 1
            beta = 0
        elif dataset == "fashion":
            dec_layers = 3
            lrn_rate = 1e-2
            alpha = 1
            beta = 10
        elif dataset == "svhn":
            dec_layers = 3
            lrn_rate = 1e-2
            alpha = 1
            beta = 1
        else:
            dec_layers = 2
            lrn_rate = 1e-4
            alpha = 1
            beta = 0
        """
        generator, decoder, teacher = self.generator, self.decoder, self.teacher
        optimizer = self.optimizer_gc

        teacher.eval()

        def _func(generator, decoder):
            generator.train(), decoder.train()

            n_classes = generator.num_classes
            n_noises = generator.num_noises
            pbar = tqdm(range(epochs), ncols=80, postfix="loss1: *.****; loss2: *.****; loss3: *.****")
            loop_loss1, loop_loss2, loop_loss3 = [], [], []
            for _ in pbar:
                noise_size = batch_size, n_noises
                noises = sample_noises(noise_size).to(self.device)
                labels = sample_labels(batch_size, n_classes, dist="onehot").to(self.device)

                images = generator(labels, noises)
                with torch.no_grad():
                    outputs = teacher(images)

                loss1 = self.cls_criterion(nn.LogSoftmax(dim=1)(outputs), labels)
                loss2 = self.dec_criterion(decoder(images), noises) * alpha
                loss3 = self.div_criterion(noises, images) * beta
                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop_loss1.append(loss1.item())
                loop_loss2.append(loss2.item())
                loop_loss3.append(loss3.item())
                pbar.postfix = f"loss1: {np.mean(loop_loss1):.4f}; loss2: {np.mean(loop_loss2):.4f}; loss3: {np.mean(loop_loss3):.4f};"
            return generator.state_dict()

        self.generator_ckpt_lst = [
            _func(copy.deepcopy(generator), copy.deepcopy(decoder)) for _ in range(num_generator)
        ]

    def generate_data(self, num_data):
        self.teacher.eval()
        gen_models = []
        for ckpt in self.generator_ckpt_lst:
            self.generator.load_state_dict(ckpt)
            gen_model = copy.deepcopy(self.generator)
            gen_model.eval()
            gen_models.append(gen_model)

        ny = gen_models[0].num_classes
        nz = gen_models[0].num_noises
        noises = sample_noises(size=(num_data, nz))
        labels_in = sample_labels(num_data, ny, dist="onehot")
        loader = DataLoader(TensorDataset(noises, labels_in), batch_size=256)

        softmax = nn.Softmax(dim=1)

        inputs, scores = [], []
        for generator in gen_models:
            x_lst, y_lst = [], []
            for z, y in loader:
                z, y = z.to(self.device), y.to(self.device)

                x = generator(y, z).detach()
                pred_y = self.teacher(x).detach()
                pred_y = softmax(torch.cat(tuple(pred_y), dim=0))

                x_lst.append(x)
                y_lst.append(y)

            inputs.append(torch.cat(tuple(x_lst), dim=0))
            scores.append(torch.cat(tuple(y_lst), dim=0))

        self.fake_dataloader = DataLoader(
            TensorDataset(torch.cat(tuple(inputs), dim=0), torch.cat(tuple(scores), dim=0)), batch_size=256
        )
        return self.fake_dataloader
