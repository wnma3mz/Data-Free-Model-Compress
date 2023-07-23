"""
Knowledge Extraction with No Observable Data (NeurIPS 2019)

Authors:
- Jaemin Yoo (jaeminyoo@snu.ac.kr), Seoul National University
- Minyong Cho (chominyong@gmail.com), Seoul National University
- Taebum Kim (k.taebum@snu.ac.kr), Seoul National University
- U Kang (ukang@snu.ac.kr), Seoul National University

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.
"""
import torch
from torch import nn


class DenseGenerator(nn.Module):
    """
    Generator for unstructured datasets.
    """

    def __init__(self, num_classes, num_features, num_noises=10, units=120, n_layers=1):
        """
        Class initializer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_noises = num_noises

        layers = [
            nn.Linear(num_noises + num_classes, units),
            nn.ELU(),
            nn.BatchNorm1d(units),
        ]

        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(units, units), nn.ELU(), nn.BatchNorm1d(units)])

        layers.append(nn.Linear(units, num_features))
        self.layers = nn.Sequential(*layers)
        self.adjust = nn.BatchNorm1d(num_features, affine=False)

    def forward(self, labels, noises, adjust=True):
        """
        Forward propagation.
        """
        out = self.layers(torch.cat((noises, labels), dim=1))
        if adjust:
            out = self.adjust(out)
        return out


class ImageGenerator(nn.Module):
    """
    Generator for image datasets.
    """

    def __init__(self, num_classes, num_channels, num_noises=10):
        """
        Class initializer.
        """
        super(ImageGenerator, self).__init__()

        fc_nodes = [num_noises + num_classes, 256, 128]
        cv_nodes = [fc_nodes[-1], 64, 16, 4, num_channels]

        self.num_classes = num_classes
        self.num_noises = num_noises
        self.fc = nn.Sequential(
            nn.Linear(fc_nodes[0], fc_nodes[1]),
            nn.BatchNorm1d(fc_nodes[1]),
            nn.ReLU(),
            nn.Linear(fc_nodes[1], fc_nodes[2]),
            nn.BatchNorm1d(fc_nodes[2]),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(cv_nodes[0], cv_nodes[1], 4, 2, 0, bias=False),
            nn.BatchNorm2d(cv_nodes[1]),
            nn.ReLU(),
            nn.ConvTranspose2d(cv_nodes[1], cv_nodes[2], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cv_nodes[2]),
            nn.ReLU(),
            nn.ConvTranspose2d(cv_nodes[2], cv_nodes[3], 4, 2, 1, bias=False),
            nn.BatchNorm2d(cv_nodes[3]),
            nn.ReLU(),
            nn.ConvTranspose2d(cv_nodes[3], cv_nodes[4], 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)).std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std

    def forward(self, labels, noises, adjust=True):
        """
        Forward propagation.
        """
        out = self.fc(torch.cat((noises, labels), dim=1))
        out = self.conv(out.view((out.size(0), out.size(1), 1, 1)))
        if adjust:
            out = self.normalize_images(out)
        return out


class ImageGenerator2(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        super(ImageGenerator2, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False),
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class Decoder(nn.Module):
    """
    Decoder for both unstructured and image datasets.
    """

    def __init__(self, in_features, out_targets, n_layers, units=120):
        """
        Class initializer.
        """
        super(Decoder, self).__init__()

        layers = [nn.Linear(in_features, units), nn.ELU(), nn.BatchNorm1d(units)]

        for _ in range(n_layers):
            layers.extend([nn.Linear(units, units), nn.ELU(), nn.BatchNorm1d(units)])

        layers.append(nn.Linear(units, out_targets))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward propagation.
        """
        out = x.view((x.size(0), -1))
        out = self.layers(out)
        return out


class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32, activation=None, final_bn=True):
        super(GeneratorA, self).__init__()

        if activation is None:
            raise ValueError("Provide a valid activation function")
        self.activation = activation

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf * 2 * self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )

        if final_bn:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                nn.BatchNorm2d(nc, affine=False),
            )
        else:
            self.conv_blocks2 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf, 3, stride=1, padding=1),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
                # nn.Tanh(),
                # nn.BatchNorm2d(nc, affine=False)
            )

    def forward(self, z, pre_x=False):
        out = self.l1(z.view(z.shape[0], -1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img, scale_factor=2)
        img = self.conv_blocks2(img)

        if pre_x:
            return img
        else:
            # img = nn.functional.interpolate(img, scale_factor=2)
            return self.activation(img)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class conv3_gen(nn.Module):
    def __init__(self, z_dim, start_dim=8, out_channels=3):
        super(conv3_gen, self).__init__()

        self.linear = nn.Linear(z_dim, 128 * start_dim**2)
        self.flatten = View((-1, 128, start_dim, start_dim))
        self.bn0 = nn.BatchNorm2d(128)

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.up2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(64, out_channels, 3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(out_channels, affine=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.flatten(x)
        x = self.bn0(x)

        x = self.up1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.up2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x_pre = self.bn3(x)
        x = self.tanh(x_pre)
        return x, x_pre
