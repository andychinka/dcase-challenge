import torch
import torch.nn as nn
import torch.nn.functional as F

#ref: resnet_layer
class ResBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, learn_bn=True, use_relu=True, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()

        self.use_relu = use_relu

        #Batch Norm
        self.bn = nn.BatchNorm2d(in_channel, affine=learn_bn)

        #Relu
        self.relu = nn.ReLU()

        #Conv2D
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              # groups=groups,
                              bias=False,
                              # dilation=dilation
                              )

    def forward(self, x):
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        x = self.conv(x)
        return x

# ref: one Stack in model_resnet
class BasicBlock(nn.Module):

    def __init__(self, in_channel, out_channel, download_sample, has_pooling:bool):
        super(BasicBlock, self).__init__()

        self.has_pooling = has_pooling
        stride = 1
        if download_sample:
            stride = (1, 2)

        self.conv_path_1_1 = ResBlock(in_channel=in_channel,
                                     out_channel=out_channel,
                                     stride=stride,
                                     learn_bn=False,
                                     use_relu=True)
        self.conv_path_1_2 = ResBlock(in_channel=out_channel,
                                     out_channel=out_channel,
                                     stride=1,
                                     learn_bn=False,
                                     use_relu=True)

        if self.has_pooling:
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=(1,2), padding=1)

        self.conv_path_2_1 = ResBlock(in_channel=out_channel,
                                     out_channel=out_channel,
                                     stride=1,
                                     learn_bn=False,
                                     use_relu=True)
        self.conv_path_2_2 = ResBlock(in_channel=out_channel,
                                     out_channel=out_channel,
                                     stride=1,
                                     learn_bn=False,
                                     use_relu=True)

    def forward(self, x):

        out = self.conv_path_1_1(x)
        out = self.conv_path_1_2(out)

        if self.has_pooling:
            x = self.avg_pool(x)
            pad_channel = out.shape[1] - x.shape[1]
            #fill zero for the channel
            x = F.pad(x.permute(0, 2, 3, 1), (0, pad_channel, 0, 0)).permute(0, 3, 1, 2)

        new_x = out + x

        out = self.conv_path_2_1(new_x)
        out = self.conv_path_2_2(out)

        out += new_x

        return out




#ref: model_resnet before the     ResidualPath = concatenate([ResidualPath1,ResidualPath2],axis=1)
class ResNetModSub(nn.Module):

    def __init__(self, in_channel=3):
        super(ResNetModSub, self).__init__()

        self.res_path1 = ResBlock(in_channel=in_channel,
                          out_channel=24,
                          stride=(1,2),
                          learn_bn=True,
                          use_relu=False
                          )

        self.stack1 = BasicBlock(in_channel=24, out_channel=24, download_sample=False, has_pooling=False)
        self.stack2 = BasicBlock(in_channel=24, out_channel=48, download_sample=True, has_pooling=True)
        self.stack3 = BasicBlock(in_channel=48, out_channel=96, download_sample=True, has_pooling=True)
        self.stack4 = BasicBlock(in_channel=96, out_channel=192, download_sample=True, has_pooling=True)

    def forward(self, x):
        x = self.res_path1(x)

        #stack of block
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        x = self.stack4(x)

        return x

class ResNetMod(nn.Module):

    def __init__(self, out_kernel_size=(132,31), in_channel=3):
        super(ResNetMod, self).__init__()

        self.high_resnet = ResNetModSub(in_channel=in_channel)
        self.low_resnet = ResNetModSub(in_channel=in_channel)

        self.out_path1 = ResBlock(in_channel=192,
                                  out_channel=384,
                                  kernel_size=1,
                                  stride=1,
                                  learn_bn=False,
                                  use_relu=True
                                  )
        self.out_path2 = ResBlock(in_channel=384,
                                  out_channel=10,
                                  kernel_size=1,
                                  stride=1,
                                  learn_bn=False,
                                  use_relu=False
                                  )
        self.out_bn = nn.BatchNorm2d(10, affine=False)

        #ref: global avg pooling https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/3
        self.out_avg_pool = nn.AvgPool2d(kernel_size=out_kernel_size)  #TODO: set the kernel_size

        self.fc = nn.Linear(out_kernel_size[0] * out_kernel_size[1] * 10, 10)

    def forward(self, x):

        # Split the data
        low_x = x[:, :, 0:64, :]
        high_x = x[:, :, 64:128, :]

        low_out = self.low_resnet(low_x)
        high_out = self.high_resnet(high_x)

        # concate togather
        out = torch.cat((low_out, high_out), 2)

        # out layer
        out = self.out_path1(out)
        out = self.out_path2(out)
        out = self.out_bn(out)
        out = self.out_avg_pool(out)
        out = torch.squeeze(out, -1)
        out = torch.squeeze(out, -1)

        # out = torch.flatten(out, 1)
        # out = self.fc(out)
        return out


if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader

    from asc.dataset.task1b_dataset_2019 import Task1bDataSet2019
    from asc import config

    print("-----test----")
    db_path = "/home/hw1-a07/dcase/datasets/TAU-urban-acoustic-scenes-2019-mobile-development"
    feature_folder = "logmel_delta2_128_44k"
    batch_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_set = Task1bDataSet2019(db_path, config.class_map, feature_folder=feature_folder)
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    model = ResNetMod().to(device)

    for batch, (x, targets, cities, devices) in enumerate(dataloader):

        inputs = torch.FloatTensor(x).to(device)
        outputs = model(inputs)
