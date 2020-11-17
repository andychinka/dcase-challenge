import torch.nn as nn


class Baseline(nn.Module):

    def __init__(self, full_connected_in=128, in_channels=1, maxpool=100):
        super(Baseline, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=7,
                padding=3
            ), #(32, 40, 500)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(5), #(32, 8, 100)
            nn.Dropout2d(0.3)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=7,
                padding=3
            ), #(64, 8, 100)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, maxpool)), #(64, 2, 1)
            nn.Dropout2d(0.3)
        )

        self.hidden = nn.Sequential(
            nn.Linear(in_features=full_connected_in, out_features=100),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(in_features=100, out_features=10),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.cnn1(x)
        out = self.cnn2(out)
        out = out.view(x.size(0), -1)
        out = self.hidden(out)
        nn.MSELoss()

        return out

    def get_loss(self, x, targets):
        output = self.forward(x)
        return self.criterion(output, targets)

    def cal_loss(self, output, targets):
        return self.criterion(output, targets)