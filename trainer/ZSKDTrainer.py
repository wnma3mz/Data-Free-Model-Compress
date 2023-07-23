# -*- coding: utf-8 -*-
import numpy as np
import torch
from criterion import KDLoss
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils import sample_noises

from trainer.Trainer import Trainer

# https://github.com/da2so/Zero-shot_Knowledge_Distillation_Pytorch
# https://github.com/vcl-iisc/ZSKD
"""
@inproceedings{
    nayak2019zero,
    title={Zero-Shot Knowledge Distillation in Deep Networks},
    author={Nayak, G. K., Mopuri, K. R., Shaj, V., Babu, R. V., and Chakraborty, A.},
    booktitle={International Conference on Machine Learning},
    pages={4743--4751},
    year={2019}
}
"""


class ZSKDTrainer(Trainer):
    # 使用教师模型的最后一层参数生成数据，学生在生成的数据上进行训练
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

        self.cwh, self.num_classes = (3, 32, 32), 10
        self.teacher = teacher

        self.eps = 1e-4

    def zero_train(self, epochs):
        student = self.model
        student.train()

        pbar = tqdm(range(epochs), ncols=80, postfix="loss_S: *.****")
        loop_loss_S = []
        for _ in range(epochs):
            for input_, target in self.fake_dataloader:
                output = student(input_.to(self.device))
                loss = self.criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loop_loss_S.append(loss.item())
            pbar.postfix = "loss_S: {:.4f}".format(np.mean(loop_loss_S))

    def generate_data(self, cls_sim_norm, num_sample, batch_size, lr, iters, beta=None):
        gen_data_lst, gen_label_lst = [], []

        if beta is None:
            beta = [0.1, 1]

        size = batch_size, *self.cwh

        # generate synthesized images.
        for k in range(self.num_classes):
            for b in beta:
                # sampling target label from Dirichlet distribution
                dir_dist = torch.distributions.dirichlet.Dirichlet(b * cls_sim_norm[k] + self.eps)

                len_ = num_sample // len(beta) // batch_size // self.num_classes
                pbar = tqdm(range(len_), ncols=80, postfix=f"class: {k}; generate_S: *.****")
                for _ in pbar:
                    y = Variable(dir_dist.rsample((batch_size,)), requires_grad=False)

                    # optimization for images
                    inputs = sample_noises(size).to(self.device)
                    inputs = Variable(inputs, requires_grad=True)
                    optimizer = torch.optim.Adam([inputs], lr)

                    for _ in range(iters):
                        output = self.teacher(inputs)
                        loss = self.kd_criterion(output, y.detach())

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        pbar.postfix = f"class: {k}; generate_S: {loss.item():.4f}"  # 只看最后优化的结果

                    # save the synthesized images
                    t_cls = torch.argmax(y, dim=1).detach().cpu().numpy()
                    gen_data_lst += inputs.detach().cpu().numpy().tolist()
                    gen_label_lst += t_cls.tolist()

        # Label is one-hot
        self.fake_dataloader = DataLoader(
            TensorDataset(torch.tensor(gen_data_lst), torch.tensor(gen_label_lst)),
            batch_size=256,
        )
        return self.fake_dataloader
