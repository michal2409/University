from .unet_utils import *

class Unet(nn.Module):
    def __init__(self, num_classes, in_channels=3):
        super(Unet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        self.input_channels = [self.in_channels, 64, 128, 256, 512]
        self.output_channels = self.input_channels[1:] + [1024]
        self.is_pooling = [True, True, True, True, False]

        self.down_convs = []
        self.up_convs = []
        
        for i in range(len(self.input_channels)):
            down_conv = DownConv(self.input_channels[i], self.output_channels[i], self.is_pooling[i])
            self.down_convs.append(down_conv)

        for i in range(len(self.input_channels)-1):
            up_conv = UpConv(self.output_channels[-(i+1)], self.input_channels[-(i+1)])
            self.up_convs.append(up_conv)

        self.conv_final = conv1x1(self.input_channels[1], self.num_classes)

        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

    def forward(self, x):
        encoder_outs = []
         
        for module in self.down_convs:
            x, x_unpooled = module(x)
            encoder_outs.append(x_unpooled)

        for i, module in enumerate(self.up_convs):
            x = module(encoder_outs[-(i+2)], x)

        return F.log_softmax(self.conv_final(x), dim=1)
