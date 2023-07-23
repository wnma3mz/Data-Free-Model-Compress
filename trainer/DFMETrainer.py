# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from criterion import KDLoss
from tqdm import tqdm
from utils import sample_noises

from trainer.Trainer import Trainer

# https://github.com/cake-lab/datafree-model-extraction


def estimate_gradient_objective(
    teacher,
    student,
    x,
    epsilon=1e-7,
    m=5,
    num_classes=10,
    device="cpu",
    pre_x=False,
):
    # Sampling from unit sphere is the method 3 from this webatch_sizeite:
    #  http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
    # x = torch.Tensor(np.arange(2*1*7*7).reshape(-1, 1, 7, 7))
    G_activation = torch.tanh
    forward_differences = True
    no_logits = 1
    logit_correction = "mean"
    loss = "l1"

    student.eval()
    teacher.eval()
    # Sample unit noise vector
    N = x.size(0)
    C = x.size(1)
    S = x.size(2)
    dim = S**2 * C

    u = np.random.randn(N * m * dim).reshape(-1, m, dim)  # generate random points from normal distribution

    d = np.sqrt(np.sum(u**2, axis=2)).reshape(-1, m, 1)  # map to a uniform distribution on a unit sphere
    u = torch.Tensor(u / d).view(-1, m, C, S, S)
    u = torch.cat((u, torch.zeros(N, 1, C, S, S)), dim=1)  # Shape N, m + 1, S^2

    u = u.view(-1, m + 1, C, S, S)

    evaluation_points = (x.view(-1, 1, C, S, S).cpu() + epsilon * u).view(-1, C, S, S)
    if pre_x:
        evaluation_points = G_activation(evaluation_points)  # Apply G_activation function

    # Compute the approximation sequentially to allow large values of m
    pred_victim = []
    pred_clone = []
    max_number_points = 32 * 156  # Hardcoded value to split the large evaluation_points tensor to fit in GPU

    with torch.no_grad():
        for i in range(N * m // max_number_points + 1):
            pts = evaluation_points[i * max_number_points : (i + 1) * max_number_points]
            pts = pts.to(device)

            pred_victim_pts = teacher(pts).detach()
            pred_clone_pts = student(pts)

            pred_victim.append(pred_victim_pts)
            pred_clone.append(pred_clone_pts)

        pred_victim = torch.cat(pred_victim, dim=0).to(device)
        pred_clone = torch.cat(pred_clone, dim=0).to(device)

        u = u.to(device)

        if loss == "l1":
            loss_fn = F.l1_loss
            if no_logits:
                pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
                if logit_correction == "min":
                    pred_victim = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
                elif logit_correction == "mean":
                    pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
                else:
                    pred_victim = pred_victim_no_logits

        elif loss == "kl":
            loss_fn = KDLoss()

        # Compute loss
        if loss == "kl":
            loss_values = -loss_fn(pred_clone, pred_victim, reduction="none").sum(dim=1).view(-1, m + 1)
        else:
            loss_values = -loss_fn(pred_clone, pred_victim, reduction="none").mean(dim=1).view(-1, m + 1)

        # Compute difference following each direction
        differences = loss_values[:, :-1] - loss_values[:, -1].view(-1, 1)
        differences = differences.view(-1, m, 1, 1, 1)

        # Formula for Forward Finite Differences
        gradient_estimates = 1 / epsilon * differences * u[:, :-1]
        if forward_differences:
            gradient_estimates *= dim

        if loss == "kl":
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S)
        else:
            gradient_estimates = gradient_estimates.mean(dim=1).view(-1, C, S, S) / (num_classes * N)

        student.train()
        loss_G = loss_values[:, -1].mean()
        return gradient_estimates.detach(), loss_G


def compute_gradient(teacher, student, x, pre_x=False, device="cpu"):
    G_activation = torch.tanh
    loss = "l1"
    no_logits = 1
    logit_correction = "mean"

    student.eval()
    N = x.size(0)
    x_copy = x.clone().detach().requires_grad_(True)
    x_ = x_copy.to(device)

    if pre_x:
        x_ = G_activation(x_)

    pred_victim = teacher(x_)
    pred_clone = student(x_)

    if loss == "l1":
        loss_fn = F.l1_loss
        if no_logits:
            pred_victim_no_logits = F.log_softmax(pred_victim, dim=1)
            if logit_correction == "min":
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.min(dim=1).values.view(-1, 1)
            elif logit_correction == "mean":
                pred_victim = pred_victim_no_logits - pred_victim_no_logits.mean(dim=1).view(-1, 1)
            else:
                pred_victim = pred_victim_no_logits
    elif loss == "kl":
        loss_fn = KDLoss()

    loss_values = -loss_fn(pred_clone, pred_victim, reduction="mean")
    # print("True mean loss", loss_values)
    loss_values.backward()
    student.train()
    return x_copy.grad, loss_values


def measure_true_grad_norm(teacher, student, device, x):
    # Compute true gradient of loss wrt x
    true_grad, _ = compute_gradient(teacher, student, x, pre_x=True, device=device)
    true_grad = true_grad.view(-1, 3072)

    # Compute norm of gradients
    norm_grad = true_grad.norm(2, dim=1).mean().cpu()

    return norm_grad


def student_loss(s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    loss = "l1"
    if loss == "l1":
        loss_fn = F.l1_loss
    elif loss == "kl":
        loss_fn = KDLoss()
    else:
        raise SystemError("please input vaild loss function name. l1 or kl")

    loss = loss_fn(s_logit, t_logit.detach())
    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


class DFMETrainer(Trainer):
    # 借助Teacher训练GAN（用了grad estimate），GAN生成数据再分别输入至教师和学生，进行度量。online training
    def __init__(
        self, model, optimizer, criterion, device, display=True, teacher=None, generator=None, optimizer_G=None
    ):
        super().__init__(model, optimizer, criterion, device, display)
        self.teacher, self.generator = teacher.to(self.device), generator.to(self.device)
        self.optimizer_G = optimizer_G

        self.num_classes = 10

        self.approx_grad = 1
        self.grad_epsilon = 1e-3
        self.grad_m = 10

        self.kd_criterion = KDLoss()  # F.l1_loss

    def zero_train(self, epochs, g_epochs, d_epochs, batch_size, nz):
        teacher, student, generator = self.teacher, self.model, self.generator
        optimizer_S, optimizer_G = self.optimizer, self.optimizer_G
        teacher.eval(), student.train(), generator.train()

        pbar = tqdm(range(epochs), ncols=80, postfix="loss_G: *.****; loss_S: *.****")
        loop_loss_S, loop_loss_G = [], []

        size = batch_size, nz

        loss = "l1"  # kl
        no_logits = 1
        logit_correction = "mean"

        for _ in pbar:
            for _ in range(g_epochs):
                # Sample Random Noise
                z = sample_noises(size).to(self.device)
                # Get fake image from generator
                # pre_x returns the output of G before applying the activation
                fake = generator(z, pre_x=self.approx_grad)

                # APPOX GRADIENT
                approx_grad_wrt_x, loss_G = estimate_gradient_objective(
                    teacher,
                    student,
                    fake,
                    epsilon=self.grad_epsilon,
                    m=self.grad_m,
                    num_classes=self.num_classes,
                    device=self.device,
                    pre_x=True,
                )

                optimizer_G.zero_grad()
                fake.backward(approx_grad_wrt_x)
                optimizer_G.step()

            for _ in range(d_epochs):
                z = sample_noises(size).to(self.device)
                fake = generator(z).detach()
                optimizer_S.zero_grad()

                with torch.no_grad():
                    t_logit = teacher(fake)

                if loss == "l1" and no_logits:
                    t_logit = F.log_softmax(t_logit, dim=1).detach()
                    if logit_correction == "min":
                        t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                    elif logit_correction == "mean":
                        t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

                s_logit = student(fake)

                loss_S = self.kd_criterion(s_logit, t_logit)
                loss_S.backward()
                optimizer_S.step()

            loop_loss_G.append(loss_G.item())
            loop_loss_S.append(loss_S.item())
            pbar.postfix = "loss_G: {:.4f}; loss_S: {:.4f}".format(np.mean(loop_loss_G), np.mean(loop_loss_S))
