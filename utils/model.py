
import torch
import torch.nn as nn
import torch.nn.functional as F


def append_spatial_location(features, start=-1, end=1):

    batch_size, _, width, height = features.size()
    device = features.device

    x_coords = torch.linspace(start, end, steps=height, device=device).unsqueeze(0).expand(width, height).unsqueeze(0)
    x_coords = x_coords.unsqueeze(0).expand(batch_size, -1, width, height)
    y_coords = torch.linspace(start, end, steps=width, device=device).unsqueeze(1).expand(width, height).unsqueeze(0)
    y_coords = y_coords.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return torch.cat([features, x_coords, y_coords], 1)

# Replicate tensorflow 'SAME' padding (Taken from https://github.com/mlperf/inference/blob/master/others/edge/object_detection/ssd_mobilenet/pytorch/utils.py#L40)
class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
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
