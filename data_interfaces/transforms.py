import torch
import numpy as np
import torchvision.transforms.functional as vis_F
import torch.nn.functional as F


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    Move channel dimension first (Pytorch Format)
    Cast samples to correct types
    """

    def __call__(self, sample):
        # swap color axis because (RGB/BGR)
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.array(sample['image']).transpose((2, 0, 1))      # FIXME : Transpose in pytorch
        return {
            'image': torch.from_numpy(image).float(),
            'question': torch.from_numpy(sample['question']).int(),
            'answer': torch.from_numpy(sample['answer']),
            'id': sample['id'],             # Not processed by the network, no need to transform to tensor.. Seems to be transfered to tensor in collate function anyways
            'scene_id': sample['scene_id']  # Not processed by the network, no need to transform to tensor
        }


class ImgBetweenZeroOne(object):
    """ Normalize the image between 0 and 1 """

    def __init__(self, max_val=255, min_val=0):
        self.max_val = max_val
        self.min_val = min_val

    def __call__(self, sample):
        sample['image'] = (sample['image'] - self.min_val) / (self.max_val - self.min_val)

        return sample


class ResizeImg(object):
    """ Resize PIL image to 'output_shape' """
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        sample['image'] = vis_F.resize(sample['image'], self.output_shape)
        return sample


class ResizeImgBasedOnHeight(object):
    """ Resize PIL image to 'output_height' while keeping image ration """
    def __init__(self, output_height):
        self.output_height = output_height

    def __call__(self, sample):
        output_width = int(self.output_height * sample['image'].width / sample['image'].height)

        if output_width + self.output_height != sample['image'].width + sample['image'].height:
            sample['image'] = vis_F.resize(sample['image'], (self.output_height, output_width))

        return sample


class ResizeImgBasedOnWidth(object):
    """ Resize PIL image to 'output_width' while keeping image ration"""
    def __init__(self, output_width):
        self.output_width = output_width

    def __call__(self, sample):
        output_height = int(self.output_width * sample['image'].height / sample['image'].width)

        if output_height + self.output_width != sample['image'].height + sample['image'].width:
            sample['image'] = vis_F.resize(sample['image'], (output_height, self.output_width))

        return sample


class PadTensor(object):
    """ Pad image Tensor to output_height x output_width. Pad to right & bottom of image"""
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def __call__(self, sample):
        width_to_pad = self.output_shape[1] - sample['image'].shape[2]
        height_to_pad = self.output_shape[0] - sample['image'].shape[1]

        if width_to_pad + height_to_pad > 0:
            sample['image'] = F.pad(sample['image'], [0, width_to_pad, 0, height_to_pad])

        sample['image_padding'] = torch.tensor([height_to_pad, width_to_pad], dtype=torch.int)

        return sample
