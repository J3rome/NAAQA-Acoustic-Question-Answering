from collections import OrderedDict

import torch.nn as nn

from models.utils import Conv2d_padded, append_spatial_location

class Conv_classifier(nn.Module):
    def __init__(self, in_channels, output_size, pooling_type, projection_size=None, spatial_location_layer=True, dropout_drop_prob=0):
        super(Conv_classifier, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.pooling_type = pooling_type
        self.spatial_location_layer = spatial_location_layer
        self.projection_size = projection_size

        in_channels = in_channels + (2 if spatial_location_layer else 0)
        # Classification (Via 1x1 conv & GlobalPooling)

        if projection_size:
            self.classif_prof_conv = nn.Sequential(OrderedDict([
                ('conv', Conv2d_padded(in_channels=in_channels,
                                   out_channels=projection_size,
                                   kernel_size=[1, 1], stride=1, padding="VALID", dilation=1)),
                ('batchnorm', nn.BatchNorm2d(projection_size, eps=0.001)),
                ('relu', nn.ReLU(inplace=True))
            ]))

        self.logits = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels=projection_size if projection_size else in_channels,
                               out_channels=output_size,
                               kernel_size=[1, 1], stride=1)),
            #('batchnorm', nn.BatchNorm2d(output_size)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_features):
        if self.spatial_location_layer:
            input_features = append_spatial_location(input_features)

        out = input_features
        if self.projection_size:
            out = self.classif_prof_conv(out)

        logits_maps = self.logits(out)

        # Global Avg/Max Pooling
        if self.pooling_type == 'max':
            logits, _ = logits_maps.view(logits_maps.shape[0], logits_maps.shape[1], -1).max(dim=2)
        else:
            logits = logits_maps.mean(dim=[2, 3])

        logits_softmaxed = self.softmax(logits)

        return logits, logits_softmaxed


class Fcn_classifier(nn.Module):
    def __init__(self, in_channels, projection_size, output_size, pooling_type, spatial_location_layer, dropout_drop_prob):
        super(Fcn_classifier, self).__init__()

        # Used for text summary
        self.summary_level = 1

        self.pooling_type = pooling_type
        self.spatial_location_layer = spatial_location_layer

        first_conv_out = 512

        self.classif_conv = nn.Sequential(OrderedDict([
            ('conv', Conv2d_padded(in_channels=in_channels + (2 if spatial_location_layer else 0),
                               out_channels=first_conv_out,
                               kernel_size=[1, 1], stride=1, padding="VALID", dilation=1, bias=False)),
            ('batchnorm', nn.BatchNorm2d(first_conv_out, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        self.hidden_layer = nn.Sequential(OrderedDict([
            ('linear', nn.Linear(first_conv_out, projection_size, bias=False)),
            ('batchnorm', nn.BatchNorm1d(projection_size, eps=0.001)),
            ('relu', nn.ReLU(inplace=True))
        ]))

        # FIXME : Why we don't have batchnorm & relu here ?
        self.logits = nn.Linear(projection_size, output_size)

        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(p=dropout_drop_prob)

    def forward(self, input_features):
        if self.spatial_location_layer:
            input_features = append_spatial_location(input_features)

        conv_out = self.classif_conv(input_features)

        # Global Max Pooling
        if self.pooling_type == 'max':
            logits, _ = conv_out.view(conv_out.shape[0], conv_out.shape[1], -1).max(dim=2)
        else:
            logits = conv_out.mean(dim=[2, 3])

        hidden_out = self.hidden_layer(logits)
        #hidden_out = self.dropout(hidden_out)

        logits = self.logits(hidden_out)

        logits_softmaxed = self.softmax(logits)

        return logits, logits_softmaxed