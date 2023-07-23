import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in=28 * 28 * 1, dim_out=10):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, 128)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.8)
        self.layer_out = nn.Linear(128, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = self.layer_input(x)
        x = self.relu(x)

        x = self.layer_out(x)
        return self.softmax(x)


class CNN(nn.Module):
    def __init__(self, dim_out=10):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, dim_out)

        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.maxpool2d(self.conv1(x)))
        x = self.relu(self.maxpool2d(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
