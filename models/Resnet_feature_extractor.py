import torch.nn as nn
import torchvision

from utils.Reproducibility_Handler import Reproductible_Block

class Resnet_feature_extractor(nn.Module):
    def __init__(self, resnet_version=101, layer_index=6, no_grads=True):      # TODO : Add Parameter to unfreeze some layers
        super(Resnet_feature_extractor, self).__init__()
        # FIXME : Try with other ResNet

        # Resnet 101 layers
        # 0 -> Conv2d
        # 1 -> BatchNorm2d
        # 2 -> ReLU
        # 3 -> MaxPool2d
        # 4 -> First Block
            # 0 -> First Bottleneck
                # conv1
                # bn1
                # conv2
                # bn2
                # conv 3
                # bn3
                # relu
                # downsample (Only in first block)
                    # 0 -> Conv2d
                    # 1 -> BatchNorm2d
            # 1 -> Second Bottleneck
            # 2 -> Third Bottleneck

        # 5 -> Second Block
            # 0 to 2 -> Bottlenecks

        # 6 -> Third Block
            # 0 to 22 -> Bottlenecks

        # 7 -> Fourth Block
            # 0 to 2 -> Bottlenecks

        # 8 -> AdaptiveAvgPool2d
        # 9 -> Linear

        assert resnet_version == 101, 'Only Resnet-101 is implemented.'

        # Prevent change to the Random State when instantiating Resnet Model
        with Reproductible_Block(reset_state_after=True):
            resnet = torchvision.models.resnet101(pretrained=True)

            self.extractor = nn.Sequential(*list(resnet.children())[:layer_index+1])

            if no_grads:
                for param in self.extractor.parameters():
                    param.requires_grad = False

    def forward(self, image, spatial_location):
        return self.extractor(image)

    def get_last_bottleneck(self):
        last_elem = self.extractor[-1]

        # The chosen layer must be a bottleneck layer, otherwise this will result in an infinite loop
        while not isinstance(last_elem, torchvision.models.resnet.Bottleneck):
            last_elem = last_elem[-1]

        return last_elem

    def get_out_channels(self):
        last_bottleneck = self.get_last_bottleneck()

        return last_bottleneck.bn3.num_features

    def get_output_shape(self, input_image, channel_first=True):
        output = self(input_image)
        output_size = output.size()[1:]     # Remove batch size dimension

        if channel_first:
            return list(output_size)

        return [output_size[1], output_size[2], output_size[0]]