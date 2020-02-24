from collections import OrderedDict

import torch
from torch import nn


class Depthwise_spatially_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, summary_level=0):
        super(Depthwise_spatially_separable_conv, self).__init__()

        assert kernel_size[0] == kernel_size[1], f'Kernel must be square for spatially separable conv. Got {kernel_size}'

        # Used for text summary
        self.summary_level = summary_level

        self.time_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size[1]), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.freq_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size[0], 1), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.projection_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(2*in_channels, out_channels, kernel_size=(1, 1), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_image):
        freq_conv_out = self.freq_conv(input_image)
        time_conv_out = self.time_conv(input_image)

        # Concatenate frequencies features and time features
        concatenated_conv_out = torch.cat([freq_conv_out, time_conv_out], 1)

        out = self.projection_conv(concatenated_conv_out)

        return out


class Spatially_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, freq_first=True):
        super(Spatially_separable_conv, self).__init__()

        self.freq_first = freq_first

        assert kernel_size[0] == kernel_size[1], f'Kernel must be square for spatially separable conv. Got {kernel_size}'

        self.time_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size[1]), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.freq_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size[0], 1), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_image):
        if self.freq_first:
            out = self.freq_conv(input_image)
            out = self.time_conv(out)
        else:
            out = self.time_conv(input_image)
            out = self.freq_conv(out)

        return out


class Depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, summary_level=0):
        super(Depthwise_separable_conv, self).__init__()

        # Used for text summary
        self.summary_level = summary_level

        self.grouped_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels)),
            ('batchnorm', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.depthwise_conv  = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), stride=stride)),
            ('batchnorm', nn.BatchNorm2d(out_channels)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_image):
        out = self.grouped_conv(input_image)
        out = self.depthwise_conv(out)

        return out