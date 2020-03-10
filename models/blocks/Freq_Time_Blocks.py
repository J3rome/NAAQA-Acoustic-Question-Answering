from collections import OrderedDict

import torch
from torch import nn
from models.utils import Conv2d_padded


class Freq_Time_Block(nn.Module):
    def __init__(self, input_channels, output_channels, freq_kernel=[3, 1], time_kernel=[1, 3], freq_stride=1,
                 time_stride=1):
        super(Freq_Time_Block, self).__init__()

        # TODO : Add residual connection ?

        self.time_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=input_channels,
                               out_channels=output_channels, kernel_size=time_kernel,
                               stride=time_stride, dilation=1, bias=False, padding='SAME')),        # FIXME : Should we add Bias ?
            ('batchnorm', nn.BatchNorm2d(output_channels, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.freq_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=input_channels,
                               out_channels=output_channels, kernel_size=freq_kernel,
                               stride=freq_stride, dilation=1, bias=False, padding='SAME')),        # FIXME : Should we add Bias ?
            ('batchnorm', nn.BatchNorm2d(output_channels, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.prof_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=2*output_channels,
                               out_channels=output_channels, kernel_size=[1, 1],
                               stride=1, dilation=1, bias=False, padding='SAME')),        # FIXME : Should we add Bias ?
            ('batchnorm', nn.BatchNorm2d(output_channels, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_image):

        time_features = self.time_conv(input_image)
        freq_features = self.freq_conv(input_image)

        concatenated_features = torch.cat([time_features, freq_features], dim=1)

        out = self.prof_conv(concatenated_features)

        return out


class Multi_res_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels_size, stride):
        super(Multi_res_conv, self).__init__()

        # TODO : Make this dynamic according to the number of element in kernels_size
        # TODO : Use grouped convs ?

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=kernels_size[0], stride=stride)),
            ('batchnorm', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=kernels_size[1], stride=stride)),
            ('batchnorm', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=kernels_size[2], stride=stride)),
            ('batchnorm', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.projection_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3*in_channels, out_channels, kernel_size=(1, 1), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_image):
        conv1_out = self.conv1(input_image)
        conv2_out = self.conv2(input_image)
        conv3_out = self.conv3(input_image)

        # Concatenate frequencies features and time features
        concatenated_conv_out = torch.cat([conv1_out, conv2_out, conv3_out], 1)

        out = self.projection_conv(concatenated_conv_out)

        return out