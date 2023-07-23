# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn.functional as F
from criterion import KDLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from trainer.Trainer import Trainer


def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = x
    # derivatives for each class
    for class_ind in range(nb_classes):
        x_var_exp = x_var.unsqueeze(0)
        score = model(x_var_exp)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()
    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, T, lmbda=0.1, nb_classes=10):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    # For each input in the previous' substitute training iteration
    lst = []
    for x in X_sub_prev:
        grads = jacobian(model, x, nb_classes)
        # Select gradient corresponding to the label predicted by the oracle
        with torch.no_grad():
            score_batch = T(x)
            y_sub = torch.argmax(score_batch, dim=1)

        grad = grads[y_sub]

        # Compute sign matrix
        grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        new_x = x + lmbda * grad_val
        lst += torch.clamp(new_x, -1, 1).cpu().numpy().tolist()
    # Return augmented training data (needs to be labeled afterwards)
    return DataLoader(
        TensorDataset(torch.tensor(lst), torch.tensor(lst)),
        batch_size=256,
    )


class JBDATrainer(Trainer):
    # 使用少量样本的数据，用雅可比的方法增强数据集，增加多样性。在此基础上进行知识蒸馏
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

        self.num_classes = 10
        self.teacher = teacher

        self.kd_criterion = KDLoss()

    def zero_train(self, train_loader, aug_rounds, epochs):
        # train_loader: few data samples
        teacher, student = self.teacher, self.model
        optimizer_S = self.optimizer
        teacher.eval(), student.train()

        # Train the substitute and augment dataset alternatively
        for aug_round in range(aug_rounds):
            pbar = tqdm(range(epochs), ncols=80, postfix="loss_S: *.****")
            loop_loss_S = []
            for _ in pbar:
                for input_, _ in train_loader:
                    with torch.no_grad():
                        y_sub = teacher(input_)
                    Sout = student(input_)
                    loss_S = self.kd_criterion(Sout, y_sub)

                    optimizer_S.zero_grad()
                    loss_S.backward()
                    optimizer_S.step()

                    loop_loss_S.append(loss_S.item())
                    pbar.postfix = "loss_S: {:.4f}".format(np.mean(loop_loss_S))

            # If we are not in the last substitute training iteration, augment dataset
            if aug_round < aug_rounds - 1:
                # Perform the Jacobian augmentation
                train_loader = jacobian_augmentation(student, train_loader, teacher, nb_classes=self.num_classes)
