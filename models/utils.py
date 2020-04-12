import torch
from torch import nn
import torch.nn.functional as F

# Replicate tensorflow 'SAME' padding (Taken from https://github.com/mlperf/inference/blob/master/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40)
class Conv2d_padded(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_padded, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(
            0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size
        )
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


def append_spatial_location(features, start=-1, end=1):

    batch_size, _, width, height = features.size()
    device = features.device

    x_coords = torch.linspace(start, end, steps=height, device=device).unsqueeze(0).expand(width, height).unsqueeze(0)
    x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, width, height)
    y_coords = torch.linspace(start, end, steps=width, device=device).unsqueeze(1).expand(width, height).unsqueeze(0)
    y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return torch.cat([features, x_coords, y_coords], 1)


def pad2d_and_cat_tensors(tensors, pad_mode="end"):
    """
    Take a list of tensors of shape [Batch_size, Channels, Height, Width]
    Will pad height & width of each tensors to make them equal
    Concatenate the padded tensors accross the channel dimension
    """
    dim_to_pad = [2, 3]
    sizes = [list(t.size()) for t in tensors]
    max_size = [None, None]

    for dim in dim_to_pad:
        max_size.append(max(sizes, key=lambda s: s[dim])[dim])

    for i, (tensor, size) in enumerate(zip(tensors, sizes)):
        to_pad = []

        for dim in dim_to_pad:
            size_diff = size[dim] - max_size[dim]
            if size_diff < 0:
                to_pad.append(abs(size_diff))
            else:
                to_pad.append(0)

        if pad_mode == "end":
            # F.pad arguments : [left, right, top, down] which is the reverse order of to_pad variable
            padding = [0, to_pad[1], 0, to_pad[0]]
        else:
            left_pad = to_pad[1] // 2
            right_pad = left_pad + (to_pad[1] % 2)
            top_pad = to_pad[0] // 2
            down_pad = top_pad + (to_pad[0] % 2)

            padding = [left_pad, right_pad, top_pad, down_pad]

        tensors[i] = F.pad(tensor, padding)

    return torch.cat(tensors, dim=1)


def get_trainable_childs(model):
    to_init = []

    for child in model.children():
        if hasattr(child, 'weight'):
            to_init.append(child)
        else:
            to_init += get_trainable_childs(child)

    return to_init


class global_avg_pooling(nn.Module):
    def __init__(self, dim):
        super(global_avg_pooling, self).__init__()

        self.dim = dim

    def forward(self, input_maps):
        return input_maps.mean(self.dim)


class global_max_pooling(nn.Module):
    def __init__(self, dim):
        super(global_max_pooling, self).__init__()

        self.dim = dim

    def forward(self, input_maps):
        return input_maps.max(self.dim)