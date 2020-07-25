import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


class Cnn_5layers_AvgPooling(nn.Module):

    def __init__(self, classes_num, activation):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.activation = activation

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.avg_pool2d(x, kernel_size=(1, 1))
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = F.sigmoid(x)

        return output

    def get_loss(self, x, targets):
        output = self.forward(x)
        return self.criterion(output, targets)

    def cal_loss(self, output, targets):
        return self.criterion(output, targets)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn_9layers_AvgPooling(nn.Module):

    def __init__(self, classes_num=10, activation="logsoftmax", permute=True, in_channel=1, skip_output=False):
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.activation = activation
        self.permute = permute
        self.skip_output = skip_output

        self.conv_block1 = ConvBlock(in_channels=in_channel, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        #Original code
        # '''Input: (batch_size, times_steps, freq_bins)'''
        #
        # x = input[:, None, :, :]
        # '''(batch_size, 1, times_steps, freq_bins)'''

        '''Input: (batch_size, 1, freq_bins, times_steps)'''
        x = input.permute(0, 1, 3, 2) if self.permute else input
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)

        if not self.skip_output:
            x = self.fc(x)
        return x

        # if self.activation == 'logsoftmax':
        #     output = F.log_softmax(x, dim=-1)
        #
        # elif self.activation == 'sigmoid':
        #     output = torch.sigmoid(x)

        # return output

    def get_loss(self, x, targets):
        output = self.forward(x)
        return self.criterion(output, targets)

    def cal_loss(self, output, targets):
        return self.criterion(output, targets)

class Cnn_9layers_AvgPooling_SepFreq(nn.Module):

    def __init__(self, classes_num=10, activation="logsoftmax", permute=True, in_channel=1):
        super(Cnn_9layers_AvgPooling_SepFreq, self).__init__()

        self.activation = activation
        self.permute = permute

        self.low_conv = Cnn_9layers_AvgPooling(classes_num, activation, permute, in_channel, skip_output=True)
        self.high_conv = Cnn_9layers_AvgPooling(classes_num, activation, permute, in_channel, skip_output=True)

        self.fc = nn.Linear(1024, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, x):
        # Split the data
        low_x = x[:, :, 0:64, :]
        high_x = x[:, :, 64:128, :]

        low_out = self.low_conv(low_x)
        high_out = self.high_conv(high_x)

        # concate togather
        out = torch.cat((low_out, high_out), 1)
        x = self.fc(out)
        return x



class Cnn_9layers_MaxPooling(nn.Module):
    def __init__(self, classes_num, activation):

        super(Cnn_9layers_MaxPooling, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='max')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = F.sigmoid(x)

        return output

    def get_loss(self, x, targets):
        output = self.forward(x)
        return self.criterion(output, targets)

    def cal_loss(self, output, targets):
        return self.criterion(output, targets)


class Cnn_13layers_AvgPooling(nn.Module):

    def __init__(self, classes_num, activation):
        super(Cnn_13layers_AvgPooling, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

        self.criterion = nn.CrossEntropyLoss()

    def init_weights(self):
        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')

        x = torch.mean(x, dim=3)
        (x, _) = torch.max(x, dim=2)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = F.sigmoid(x)

        return output

    def get_loss(self, x, targets):
        output = self.forward(x)
        return self.criterion(output, targets)

    def cal_loss(self, output, targets):
        return self.criterion(output, targets)