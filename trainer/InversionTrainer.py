# -*- coding: utf-8 -*-
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from criterion import KDLoss
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
from utils import sample_labels, sample_noises
from torch.autograd import Variable

from trainer.Trainer import Trainer

# https://github.com/NVlabatch_size/DeepInversion


def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)


def clip(image_tensor, use_fp16=False):
    """
    adjust the input based on mean and variance
    """
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


def denormalize(image_tensor, use_fp16=False):
    """
    convert floats back to input
    """
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c] * s + m, 0, 1)

    return image_tensor


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (
        (diff1.abatch_size() / 255.0).mean()
        + (diff2.abatch_size() / 255.0).mean()
        + (diff3.abatch_size() / 255.0).mean()
        + (diff4.abatch_size() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


class DeepInversionFeatureHook:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    """

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()


class InversionTrainer(Trainer):
    # 借助教师模型生成数据，学生在生成的数据上进行训练
    def __init__(self, model, optimizer, criterion, device, display=True, teacher=None):
        super().__init__(model, optimizer, criterion, device, display)
        self.teacher = teacher.to(self.device)
        self.jitter = 30
        self.first_bn_multiplier = 10

        self.var_scale_l2 = 1e-4
        self.var_scale_l1 = 0.0
        self.bn_reg_scale = 1e-2
        self.l2_scale = 1e-5
        self.main_loss_multiplier = 1.0

        self.need_save_imgs = False
        self.loss_r_feature_layers = []

        for module in self.teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        self.num_classes, self.cwh = 10, (3, 32, 32)
        self.kd_criterion = KDLoss()

    def zero_train(self, epochs):
        student = self.model
        student.train()

        pbar = tqdm(range(epochs), ncols=80, postfix="loss_S: *.****")
        loop_loss_S = []
        for _ in range(epochs):
            for input_, target in self.fake_dataloader:
                output = student(input_.to(self.device))
                loss = self.kd_criterion(output, target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            loop_loss_S.append(loss.item())
            pbar.postfix = "loss_S: {:.4f}".format(np.mean(loop_loss_S))

    def generate_data(self, epochs, lr=0.25, batch_size=32):
        criterion = self.criterion  # ce loss
        teacher = self.teacher

        x_lst, y_lst = [], []
        best_inputs, best_cost = None, np.inf
        lower_res, iters = 1, 2000
        lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
        size = batch_size, *self.cwh
        for _ in range(epochs):

            inputs = sample_noises(size)
            inputs = Variable(inputs.to(self.device), requires_grad=True)
            targets = sample_labels(batch_size, self.num_classes, dist="onehot").to(self.device)

            optimizer = optim.Adam([inputs], lr=lr, betas=[0.5, 0.9], eps=1e-8)
            lr_scheduler = lr_cosine_policy(lr, 100, iters)
            pbar = tqdm(range(iters), ncols=80, postfix="loss_feature: *.****; loss: *.****")
            loop_loss_feature, loop_loss = [], []

            for iteration_loc in pbar:
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))
                if random.random() > 0.5:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                outputs = teacher(inputs_jit)
                loss = criterion(outputs, targets)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # R_feature loss
                rescale = [self.first_bn_multiplier] + [1.0 for _ in range(len(self.loss_r_feature_layers) - 1)]
                loss_r_feature = sum(
                    [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)]
                )
                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(batch_size, -1), dim=1).mean()

                # combining losses
                loss_aux = (
                    self.var_scale_l2 * loss_var_l2
                    + self.var_scale_l1 * loss_var_l1
                    + self.bn_reg_scale * loss_r_feature
                    + self.l2_scale * loss_l2
                )

                loss = self.main_loss_multiplier * loss + loss_aux

                optimizer.zero_grad()
                teacher.zero_grad()
                loss.backward()
                optimizer.step()

                # clip color outlayers
                inputs.data = clip(inputs.data)

                if best_cost > loss.item():
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

                loop_loss_feature.append(loss_r_feature.item())
                loop_loss.append(loss.item())
                pbar.postfix = f"loss_feature: {np.mean(loop_loss_feature):.4f}; loss: {np.mean(loop_loss):.4f}"

            x_lst.append(best_inputs.detach())
            y_lst.append(targets.cpu().numpy())

            if self.need_save_imgs:
                best_inputs = denormalize(best_inputs)
                self.save_images(best_inputs, targets)

        self.fake_dataloader = DataLoader(
            TensorDataset(torch.cat(tuple(x_lst), dim=0), torch.cat(tuple(y_lst), dim=0)), batch_size=256
        )
        return self.fake_dataloader

    def save_images(self, images, targets):
        # method to store generated images locally
        self.final_data_path = "./imgs"
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            place_to_store = "{}/img_s{:03d}_id{:03d}.jpg".format(self.final_data_path, class_id, id)

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)
