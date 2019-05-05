import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

def upconv2x2(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

def maxpool2x2():
    return nn.MaxPool2d(kernel_size=2, stride=2)

class DownConv(nn.Module):
    ''' Convolution block: (conv3x3 => BN => ReLU)*2
        if is_pooling = True then maxpool2x2 is applied
    '''
    def __init__(self, in_channels, out_channels, is_pooling):
        super(DownConv, self).__init__()
        self.is_pooling = is_pooling

        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)

        self.conv1_bn = nn.BatchNorm2d(out_channels)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        if self.is_pooling:
            self.pooling = maxpool2x2()

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x_unpooled = x
        if self.is_pooling:
            x = self.pooling(x)

        return x, x_unpooled

class UpConv(nn.Module):
    ''' Convolution block: upconv2x2 => DownConv '''
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()

        self.up_conv = upconv2x2(in_channels, out_channels)
        self.down_conv = DownConv(2*out_channels, out_channels, is_pooling=False)

    def forward(self, x_down, x_up):
        x_up = self.up_conv(x_up)

        # Padding x_up such that it's shape match x_down
        dy = x_down.shape[2] - x_up.shape[2]
        dx = x_down.shape[3] - x_up.shape[3]
        x_up = F.pad(x_up, (dx//2, dx-dx//2, dy//2, dy-dy//2))

        x = torch.cat((x_down, x_up), dim=1)
        x, _ = self.down_conv(x)

        return x