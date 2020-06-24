import torch.nn as nn
import torch


class AlexNet(nn.Module):

    def __init__(self, in_channel=1, num_classes=1000, dropout=0.5):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 64, kernel_size=5, stride=2, padding=2),
            # nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_loss(self, x, targets):
        output = self.forward(x)
        return self.criterion(output, targets)

    def cal_loss(self, output, targets):
        return self.criterion(output, targets)


class Attention(nn.Module):

    def __init__(self, lambda_weight=0.3, n_channels=256):
        super(Attention, self).__init__()
        self.lambda_weight = torch.tensor(lambda_weight, dtype=torch.float32, requires_grad=False)
        self.linear1 = nn.Linear(in_features=n_channels, out_features=n_channels)
        self.linear2 = nn.Linear(in_features=n_channels, out_features=1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        dims = x.size()
        x = x.reshape(dims[0], -1, dims[-1])
        v = self.tanh(self.linear1(x))
        v = self.linear2(v)
        v = v.squeeze(2)

        v = self.softmax(v * self.lambda_weight)
        v = v.unsqueeze(-1)
        output = (x * v).sum(axis=1)

        return output


class AlexNetWithAtten(nn.Module):
    '''
    Input shape: B x C x T x F
        B: batchsize
        C: input channels
        T: time
        F: frequency
    '''

    def __init__(self, in_channel=1, num_classes=10, dropout=0.5):
        super(AlexNetWithAtten, self).__init__()
        self.features = nn.Sequential(
            # nn.BatchNorm2d(num_features=1),

            nn.Conv2d(in_channel, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Dropout2D(),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.BatchNorm2d(num_features=192),
            # nn.Dropout2D(),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=384),
            # nn.Dropout2D(),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=256),
            # nn.Dropout2D(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.BatchNorm2d(num_features=256)
            nn.Dropout2d(p=dropout)
        )

        self.attention = Attention(n_channels=256)
        self.dropout = nn.Dropout(dropout)
        # self.linearout = nn.Linear(in_features=256, out_features=num_classes)
        self.linearout = nn.Linear(in_features=256, out_features=1024)
        self.linearout2 = nn.Linear(in_features=1024, out_features=num_classes)
        # self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = x.permute([0, 2, 3, 1])
        x = self.attention(x)
        # x = self.dropout(x)
        out = self.linearout(x)
        out = self.dropout(out)
        out = self.linearout2(out)

        return out
