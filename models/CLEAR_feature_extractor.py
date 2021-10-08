from collections import OrderedDict

import torch
from torch import nn
from models.utils import Conv2d_padded, append_spatial_location, pad2d_and_cat_tensors
from models.blocks.Freq_Time_Blocks import Freq_Time_Depthwise_Block


class Original_Film_Extractor(nn.Module):
    def __init__(self, config, input_image_channels):
        super(Original_Film_Extractor, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.config = config
        self.convs = nn.ModuleList()
        self.out_chan = config['out'][-1]

        in_channels = input_image_channels

        projection_size = None

        if len(config['out']) > len(config['kernels']):
            projection_size = config['out'][-1]
            config['out'] = config['out'][:-1]

        for out_chan, kernel, stride in zip(config['out'], config['kernels'], config['strides']):
            self.convs.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_chan, kernel_size=kernel,
                                       stride=stride, dilation=1, bias=False, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_chan, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
            ])))

            in_channels = out_chan

        if projection_size is not None:
            self.projection = nn.Conv2d(in_channels=in_channels, out_channels=projection_size, kernel_size=[1, 1],
                                         stride=[1, 1], bias=False)
        else:
            self.projection = None

    def forward(self, input_image):
        out = input_image

        for conv in self.convs:
            out = conv(out)

        if self.projection is not None:
            out = self.projection(out)

        return out

    def get_out_channels(self):
        return self.out_chan


class Freq_Time_Separated_Extractor_no_pooling(nn.Module):
    def __init__(self, config, input_channels, with_bias=True):
        super(Freq_Time_Separated_Extractor_no_pooling, self).__init__()

        self.config = config

        self.out_channels = config['out'][-1]

        self.time_blocks = nn.ModuleList()
        self.freq_blocks = nn.ModuleList()
        self.nb_time_blocks = len(config['time_kernels'])
        self.nb_freq_blocks = len(config['freq_kernels'])

        self.do_fusion = len(config['out']) > self.nb_time_blocks

        # TODO : Permit different number of time and freq blocks
        assert self.nb_time_blocks == self.nb_freq_blocks, "Invalid config. Must have same number of time & freq block"

        in_channels = input_channels
        iterator = zip(config['out'][:-1], config['time_kernels'], config['time_strides'],
                       config['freq_kernels'], config['freq_strides'])
        for out_channels, time_kernel, time_stride, freq_kernel, freq_stride in iterator:
            self.time_blocks.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=time_kernel,
                                       stride=time_stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
                #('pooling', nn.MaxPool2d(freq_stride))
            ])))

            self.freq_blocks.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=freq_kernel,
                                       stride=freq_stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
                #('pooling', nn.MaxPool2d(time_stride))
            ])))

            in_channels = out_channels

        if self.do_fusion:
            self.fusion_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=self.out_channels, kernel_size=[1, 1],
                                         stride=[1, 1], bias=False)

    def forward(self, input_image):
        # TODO : Add spatial location maps ?
        time_out = input_image
        freq_out = input_image

        for time_block, freq_block in zip(self.time_blocks, self.freq_blocks):
            time_out = time_block(time_out)
            freq_out = freq_block(freq_out)

        # FIXME : Won't work if time block & freq block don't have the same kernels (inversed).
        # FIXME : We might want to pad and concat ? Or find a better fusing mechanism ?

        out = pad2d_and_cat_tensors([time_out, freq_out], pad_mode='end')

        if self.do_fusion:
            out = self.fusion_conv(out)

        return out

    def get_out_channels(self):
        return self.out_channels


class Freq_Time_Separated_Extractor(nn.Module):
    def __init__(self, config, input_channels, with_bias=True):
        super(Freq_Time_Separated_Extractor, self).__init__()

        self.config = config
        self.spatial_location = config['spatial_location']
        self.nb_spatial_location = len(self.spatial_location)

        self.out_channels = config['out'][-1]

        self.time_blocks = nn.ModuleList()
        self.freq_blocks = nn.ModuleList()
        self.nb_time_blocks = len(config['time_kernels'])
        self.nb_freq_blocks = len(config['freq_kernels'])

        self.do_fusion = len(config['out']) > self.nb_time_blocks

        # TODO : Permit different number of time and freq blocks
        assert self.nb_time_blocks == self.nb_freq_blocks, "Invalid config. Must have same number of time & freq block"

        in_channels = input_channels + self.nb_spatial_location
        iterator = zip(config['out'][:-1], config['time_kernels'], config['time_strides'],
                       config['freq_kernels'], config['freq_strides'])
        for out_channels, time_kernel, time_stride, freq_kernel, freq_stride in iterator:
            self.time_blocks.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=time_kernel,
                                       stride=time_stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True)),
                ('pooling', nn.MaxPool2d(freq_stride))
            ])))

            self.freq_blocks.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=freq_kernel,
                                       stride=freq_stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True)),
                ('pooling', nn.MaxPool2d(time_stride))
            ])))

            in_channels = out_channels + self.nb_spatial_location

        if self.do_fusion:
            in_channels -= self.nb_spatial_location
            self.fusion_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=self.out_channels, kernel_size=[1, 1],
                                         stride=[1, 1], bias=False)

    def forward(self, input_features):
        time_out = input_features
        freq_out = input_features

        for time_block, freq_block in zip(self.time_blocks, self.freq_blocks):
            if self.nb_spatial_location > 0:
                time_out = append_spatial_location(time_out, axis=self.spatial_location)
                freq_out = append_spatial_location(freq_out, axis=self.spatial_location)

            time_out = time_block(time_out)
            freq_out = freq_block(freq_out)

        out = pad2d_and_cat_tensors([time_out, freq_out], pad_mode='end')

        if self.do_fusion:
            out = self.fusion_conv(out)

        return out

    def get_out_channels(self):
        return self.out_channels


class Freq_Time_Separated_No_Pool_Extractor(nn.Module):
    def __init__(self, config, input_channels, with_bias=True):
        super(Freq_Time_Separated_No_Pool_Extractor, self).__init__()

        self.config = config
        self.spatial_location = config['spatial_location']
        self.nb_spatial_location = len(self.spatial_location)

        self.out_channels = config['out'][-1]

        self.time_blocks = nn.ModuleList()
        self.freq_blocks = nn.ModuleList()
        self.nb_time_blocks = len(config['time_kernels'])
        self.nb_freq_blocks = len(config['freq_kernels'])

        self.do_fusion = len(config['out']) > self.nb_time_blocks

        # TODO : Permit different number of time and freq blocks
        assert self.nb_time_blocks == self.nb_freq_blocks, "Invalid config. Must have same number of time & freq block"

        nb_filters = config['out'][:-1] if self.do_fusion else config['out']
        in_channels = input_channels + self.nb_spatial_location
        iterator = zip(nb_filters, config['time_kernels'], config['time_strides'],
                       config['freq_kernels'], config['freq_strides'])
        for out_channels, time_kernel, time_stride, freq_kernel, freq_stride in iterator:

            stride = [freq_stride[0], time_stride[1]]

            self.time_blocks.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=time_kernel,
                                       stride=stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
            ])))

            self.freq_blocks.append(nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=freq_kernel,
                                       stride=stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
            ])))

            in_channels = out_channels + self.nb_spatial_location

        if self.do_fusion:
            self.fusion_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=self.out_channels, kernel_size=[1, 1],
                                         stride=[1, 1], bias=False)

    def forward(self, input_features):
        time_out = input_features
        freq_out = input_features

        for time_block, freq_block in zip(self.time_blocks, self.freq_blocks):
            if self.nb_spatial_location > 0:
                time_out = append_spatial_location(time_out, axis=self.spatial_location)
                freq_out = append_spatial_location(freq_out, axis=self.spatial_location)

            time_out = time_block(time_out)
            freq_out = freq_block(freq_out)

        out = pad2d_and_cat_tensors([time_out, freq_out], pad_mode='end')

        if self.do_fusion:
            out = self.fusion_conv(out)

        return out

    def get_out_channels(self):
        return self.out_channels


class Freq_Time_Interlaced_Extractor(nn.Module):
    def __init__(self, config, input_channels, with_bias=True):
        super(Freq_Time_Interlaced_Extractor, self).__init__()

        self.config = config
        self.spatial_location = config['spatial_location']
        self.nb_spatial_location = len(self.spatial_location)

        self.out_channels = config['out'][-1]

        assert len(config['time_kernels']) == len(config['freq_kernels']) == len(config['time_strides']) == len(config['freq_strides']), \
            "Config must specify same number of time & freq kernels"

        self.blocks = nn.ModuleList()
        self.nb_blocks = len(config['time_kernels'])

        self.need_projection = len(config['out']) != self.nb_blocks

        if config['time_first']:
            iterator = zip(config['out'], config['time_kernels'], config['time_strides'],
                           config['freq_kernels'], config['freq_strides'])
        else:
            iterator = zip(config['out'], config['freq_kernels'], config['freq_strides'],
                           config['time_kernels'], config['time_strides'])

        in_channels = input_channels + self.nb_spatial_location
        for out_channels, first_kernel, first_stride, second_kernel, second_stride in iterator:

            first_conv = nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=first_kernel,
                                       stride=first_stride, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
            ]))

            self.blocks.append(first_conv)

            in_channels = out_channels + self.nb_spatial_location

            second_conv = nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                       out_channels=out_channels, kernel_size=second_kernel,
                                       stride=second_kernel, dilation=1, bias=with_bias, padding='SAME')),
                ('batchnorm', nn.BatchNorm2d(out_channels, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
            ]))

            self.blocks.append(second_conv)

        if self.need_projection:
            self.channel_projection = nn.Conv2d(in_channels - self.nb_spatial_location, self.out_channels, kernel_size=[1, 1], stride=[1, 1])

    def forward(self, input_features):
        out = input_features

        for block in self.blocks:
            if self.nb_spatial_location > 0:
                out = append_spatial_location(out, axis=self.spatial_location)

            out = block(out)

        if self.need_projection:
            out = self.channel_projection(out)

        return out

    def get_out_channels(self):
        return self.out_channels


class Freq_Time_Pooled_Extractor(nn.Module):
    def __init__(self, input_image_channels, out_channels, nb_blocks=4):
        super(Freq_Time_Pooled_Extractor, self).__init__()

        self.out_channels = out_channels
        self.blocks = nn.ModuleList()

        in_channels = input_image_channels
        for i in range(nb_blocks):
            self.blocks.append(Freq_Time_Depthwise_Block(in_channels, out_channels,
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
            conv_out = append_spatial_location(conv_out, axis=self.config['stem']['spatial_location'])

        # FIXME : Move the stem logic inside our own feature extractor
        # FIXME : Probably need to be resnet style, otherwise we'll get vanishing gradients
        # FIXME : We should keep a stem layer after our feature extractor (I guess ?)

        conv_out = self.stem_conv(conv_out)
        conv_out = self.max_pool_in_freq(conv_out)

        if self.config['stem']['spatial_location']:
            conv_out = append_spatial_location(conv_out, axis=self.config['stem']['spatial_location'])

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

        spatial_location_extra_channels = len(config['stem']['spatial_location'])

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
