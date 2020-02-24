from collections import OrderedDict

import torch
from torch import nn
from models.utils import Conv2d_padded, append_spatial_location
from models.blocks.Freq_Time_Blocks import Freq_Time_Block


class Original_Film_Extractor(nn.Module):
    def __init__(self, config, input_image_channels):
        super(Original_Film_Extractor, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.config = config
        nb_features = config['stem']['conv_out']

        # FIXME : Padding same -- tensorflow
        self.conv1 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=input_image_channels,
                               out_channels=nb_features, kernel_size=[4, 4],
                               stride=2, dilation=1, bias=False, padding='SAME')),
            ('batchnorm', nn.BatchNorm2d(nb_features, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=nb_features,
                               out_channels=nb_features, kernel_size=[4, 4],
                               stride=2, dilation=1, bias=False, padding='SAME')),
            ('batchnorm', nn.BatchNorm2d(nb_features, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=nb_features,
                               out_channels=nb_features, kernel_size=[4, 4],
                               stride=2, dilation=1, bias=False, padding='SAME')),
            ('batchnorm', nn.BatchNorm2d(nb_features, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=nb_features,
                               out_channels=nb_features, kernel_size=[4, 4],
                               stride=2, dilation=1, bias=False, padding='SAME')),
            ('batchnorm', nn.BatchNorm2d(nb_features, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

    def forward(self, input_image):
        out = self.conv1(input_image)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)

        return out

    def get_out_channels(self):
        return self.config['stem']['conv_out']


class Freq_Time_Extractor(nn.Module):
    def __init__(self, input_image_channels, out_channels, nb_blocks=4):
        super(Freq_Time_Extractor, self).__init__()

        self.out_channels = out_channels
        self.blocks = nn.ModuleList()

        in_channels = input_image_channels
        for i in range(nb_blocks):
            self.blocks.append(Freq_Time_Block(in_channels, out_channels,
                                               freq_kernel=[4, 1], freq_stride=2, time_kernel=[1, 4], time_stride=2))

            in_channels = out_channels

    def forward(self, input_image):
        # FIXME : spatial location map ?
        out = input_image
        for block in self.blocks:
            out = block(out)

        return out

    def get_out_channels(self):
        return self.out_channels
    

class Image_pipeline(nn.Module):
    def __init__(self, config, input_image_channels, feature_extraction_config, dropout_drop_prob=0):
        super(Image_pipeline, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.config = config

        spatial_location_extra_channels = 2 if config['stem']['spatial_location'] else 0

        if feature_extraction_config is not None:
            # Instantiating the feature extractor affects the random state (Raw Vs Pre-Extracted Features).
            # We restore it to ensure reproducibility between input type
            random_state = get_random_state()
            self.feature_extractor = Resnet_feature_extractor(resnet_version=feature_extraction_config['version'],
                                                              layer_index=feature_extraction_config['layer_index'])
            input_image_channels = self.feature_extractor.get_out_channels()
            set_random_state(random_state)
        else:
            self.feature_extractor = None

        ## Stem
        self.stem_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=input_image_channels + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv2 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv3 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv4 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.stem_conv5 = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=config['stem']['conv_out'] + spatial_location_extra_channels,
                               out_channels=config['stem']['conv_out'], kernel_size=config['stem']['conv_kernel'],
                               stride=1, padding='SAME', dilation=1)),
            ('batchnorm', nn.BatchNorm2d(config['stem']['conv_out'], eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # Pooling
        # TODO : Combine with avg pooling ?
        self.max_pool_in_freq = nn.MaxPool2d((3, 1))
        self.max_pool_in_time = nn.MaxPool2d((1, 3))
        self.max_pool_square = nn.MaxPool2d((2, 2))
        # self.max_pool_square = nn.MaxPool2d((3, 3))

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_image):
        conv_out = input_image
        if self.feature_extractor:
            # Extract features using pretrained network
            conv_out = self.feature_extractor(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        # FIXME : Move the stem logic inside our own feature extractor
        # FIXME : Probably need to be resnet style, otherwise we'll get vanishing gradients
        # FIXME : We should keep a stem layer after our feature extractor (I guess ?)

        conv_out = self.stem_conv(conv_out)
        conv_out = self.max_pool_in_freq(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.stem_conv2(conv_out)
        conv_out = self.max_pool_in_time(conv_out)

        return conv_out

    def get_out_channels(self):
        return self.config['stem']['conv_out']


class Separable_conv_image_pipeline(nn.Module):
    def __init__(self, config, input_image_channels, dropout_drop_prob=0):
        super(Separable_conv_image_pipeline, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.config = config

        spatial_location_extra_channels = 2 if config['stem']['spatial_location'] else 0

        self.feature_extractor = None

        if True or self.config['depthwise']:
            # TODO : Residual connections ?
            self.conv1 = Depthwise_separable_conv(in_channels=input_image_channels + spatial_location_extra_channels,
                                                  out_channels=32,
                                                  kernel_size=(3, 3),
                                                  stride=2,
                                                  summary_level=2)

            self.conv2 = Depthwise_separable_conv(in_channels=32 + spatial_location_extra_channels,
                                                  out_channels=32,
                                                  kernel_size=(3, 3),
                                                  stride=2,
                                                  summary_level=2)

            self.conv3 = Depthwise_separable_conv(in_channels=32 + spatial_location_extra_channels,
                                                  out_channels=64,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  summary_level=2)

            self.conv4 = Depthwise_separable_conv(in_channels=64 + spatial_location_extra_channels,
                                                  out_channels=64,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  summary_level=2)

            self.conv5 = Depthwise_separable_conv(in_channels=64 + spatial_location_extra_channels,
                                                  out_channels=128,
                                                  kernel_size=(3, 3),
                                                  stride=1,
                                                  summary_level=2)

        # Pooling
        # TODO : Combine with avg pooling ?
        self.max_pool_in_freq = nn.MaxPool2d((3, 1))
        self.max_pool_in_time = nn.MaxPool2d((1, 3))
        self.max_pool_square = nn.MaxPool2d((2, 2))
        # self.max_pool_square = nn.MaxPool2d((3, 3))

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_image):
        conv_out = input_image

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.conv1(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.conv2(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.conv3(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.conv4(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out)

        conv_out = self.conv5(conv_out)

        #conv_out = self.max_pool_in_freq(conv_out)
        #conv_out = self.max_pool_in_time(conv_out)
        #conv_out = self.max_pool_square(conv_out)

        return conv_out
