# -*- coding: utf-8 -*-

import itertools

import numpy as np
import torch
import torch.nn as nn
from criterion import KDLoss
from tqdm import tqdm

from trainer.Trainer import Trainer


def zoge_backward(x_pre, x, S, T, ndirs, device, batch_size, mu):
    for p in S.parameters():
        p.requires_grad = False

    grad_est = torch.zeros_like(x_pre)
    d = np.array(x.shape[1:]).prod()

    with torch.no_grad():
        Sout = S(x)
        Tout = T(x)
        lossG = -KDLoss("none")(Tout, Sout)
        for _ in range(ndirs):
            u = torch.randn(x_pre.shape, device=device)
            u_flat = u.view([batch_size, -1])
            u_norm = u / torch.norm(u_flat, dim=1).view([-1, 1, 1, 1])
            x_mod_pre = x_pre + (mu * u_norm)
            x_mod = nn.Tanh()(x_mod_pre)
            Sout = S(x_mod)
            Tout = T(x_mod)
            lossG_mod = -KDLoss("none")(Tout, Sout)
            grad_est += ((d / ndirs) * (lossG_mod - lossG) / mu).view([-1, 1, 1, 1]) * u_norm

    grad_est /= batch_size
    x_det_pre = x_pre.detach()
    x_det_pre.requires_grad = True
    x_det_pre.retain_grad()
    x_det = nn.Tanh()(x_det_pre)
    Sout = S(x_det)
    Tout = T(x_det)
    lossG_det = -KDLoss()(Tout, Sout)
    lossG_det.backward()
    grad_true_flat = x_det_pre.grad.view([batch_size, -1])
    grad_est_flat = grad_est.view([batch_size, -1])
    cos = nn.CosineSimilarity(dim=1)
    cs = cos(grad_true_flat, grad_est_flat)
    mag_ratio = grad_est_flat.norm(2, dim=1) / grad_true_flat.norm(2, dim=1)

    x_pre.backward(grad_est, retain_graph=True)

    for p in S.parameters():
        p.requires_grad = True

    lossG = lossG_det.detach()
    return lossG.mean(), cs.mean(), mag_ratio.mean()


class MAZETrainer(Trainer):
    # 借助Teacher训练GAN（用了zero-grad estimate），GAN生成数据再分别输入至教师和学生，进行度量, 加上了数据回放（防止灾难性遗忘）。online training
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        display=True,
        teacher=None,
        generator=None,
        optG=None,
    ):
        super().__init__(model, optimizer, criterion, device, display)
        self.teacher = teacher.to(self.device)
        self.generator = generator.to(self.device)
        self.optG = optG

        self.kd_criterion = KDLoss()

    def zero_train(
        self,
        batch_size,
        iter_clone,
        ndirs,
        iter_gen,
        epochs,
        latent_dim,
        mu,
        iter_exp,
        white_box=False,
    ):
        teacher, student, generator = self.teacher, self.model, self.generator
        optS, optG = self.optimizer, self.optG
        teacher.eval(), student.train(), generator.train()

        ds = []  # dataset for experience replay
        for p in teacher.parameters():
            p.requires_grad = False

        pbar = tqdm(range(epochs), ncols=80, postfix="loss_G: *.****; loss_S: *.****")
        loop_loss_S, loop_loss_G, loop_loss_S_exp = [], [], []

        for _ in pbar:
            # (1) Update Generator
            for _ in range(iter_gen):
                z = torch.randn((batch_size, latent_dim), device=self.device)
                x, x_pre = generator(z)
                if white_box:
                    Tout = teacher(x)
                    Sout = student(x)
                    (loss_G) = -self.kd_criterion(Tout, Sout)
                    (loss_G).backward(retain_graph=True)  # maybe not need set True
                else:
                    (loss_G), cs, mag_ratio = zoge_backward(
                        x_pre, x, student, teacher, ndirs, self.device, batch_size, mu
                    )
                optG.zero_grad()
                optG.step()

            # (2) Update Clone network
            for c in range(iter_clone):
                with torch.no_grad():
                    if c != 0:  # reuse x from generator update for c == 0
                        z = torch.randn((batch_size, latent_dim), device=self.device)
                        x, _ = generator(z)
                    x = x.detach()
                    Tout = teacher(x)

                Sout = student(x)
                loss_S = self.kd_criterion()(Tout, Sout)

                optS.zero_grad()
                loss_S.backward()
                optS.step()

            # (4) Experience Replay
            # Store the last batch for experience replay
            batch = [(a, b) for a, b in zip(x.cpu().detach().numpy(), Tout.cpu().detach().numpy())]

            ds += batch
            gen_train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            gen_train_loader_iter = itertools.cycle(gen_train_loader)

            for c in range(iter_exp):
                x_prev, T_prev = next(gen_train_loader_iter)
                if x_prev.size(0) < batch_size:
                    break
                x_prev, T_prev = x_prev.to(self.device), T_prev.to(self.device)
                S_prev = student(x_prev)
                loss_S_exp = self.kd_criterion()(T_prev, S_prev)

                optS.zero_grad()
                loss_S_exp.backward()
                optS.step()

            loop_loss_G.append(loss_G.item())
            loop_loss_S.append(loss_S.item())
            loop_loss_S_exp.append(loss_S_exp.item())
            pbar.postfix = f"loss_G: {np.mean(loop_loss_G):.4f}; loss_S: {np.mean(loop_loss_S):.4f}, loss_S_exp: {np.mean(loop_loss_S_exp):.4f}"
