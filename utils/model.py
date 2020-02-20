from collections import OrderedDict

import os
import numpy as np
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


def tf_weight_transfer(model, weights_path, output_path):
    # TODO : Word embedding weights & GRU Weights & Batchnorm ?
    for filename in os.listdir(weights_path):
        filepath = f"{weights_path}/{filename}"
        if os.path.isfile(filepath) and ".npy" in filename:
            if "image_conv_feat" in filename:
                if 'weights' in filename:
                    if "conv_1" in filename:
                        model.image_pipeline.conv1.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "conv_2" in filename:
                        model.image_pipeline.conv2.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "conv3" in filename:
                        model.image_pipeline.conv3.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "conv4" in filename:
                        model.image_pipeline.conv4.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
            elif "image_film_stack" in filename:
                if "stem_conv" in filename and "weights" in filename:
                    model.image_pipeline.stem_conv.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                if "resblocks" in filename:
                    if "ResBlock_0" in filename:
                        resblock_idx = 0
                    elif "ResBlock_1" in filename:
                        resblock_idx = 1
                    elif "ResBlock_2" in filename:
                        resblock_idx = 2
                    elif "ResBlock_3" in filename:
                        resblock_idx = 3
                    elif "ResBlock_4" in filename:
                        resblock_idx = 4

                    if "conv1" in filename:
                        if "weights" in filename:
                            model.resblocks[resblock_idx].conv1.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                        elif "biases" in filename:
                            model.resblocks[resblock_idx].conv1.conv.bias = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "conv2" in filename:
                        if "weights" in filename:
                            model.resblocks[resblock_idx].conv2.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                        elif "biases" in filename:
                            model.resblocks[resblock_idx].conv2.conv.bias = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "film_projection" in filename:
                        if "weights" in filename:
                            model.resblocks[resblock_idx].film_layer.film.params_vector.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                        elif "biases" in filename:
                            model.resblocks[resblock_idx].film_layer.film.params_vector.bias = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))

                elif "head_conv" in filename and 'weights' in filename:
                    model.classifier.classif_conv.conv.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))

            elif "classifier" in filename:
                if "hidden_layer" in filename and "weights" in filename:
                    model.classifier.hidden_layer.linear.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                elif "softmax_layer" in filename:
                    if "weights" in filename:
                        model.classifier.logits.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "biases" in filename:
                        model.classifier.logits.bias = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
            elif "word_embedding" in filename and "weights" in filename:
                model.question_pipeline.word_emb.weight = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
            elif "gru_cell" in filename:
                if "candidate" in filename:
                    if 'weights' in filename:
                        model.question_pipeline.rnn_state.weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "biases" in filename:
                        model.question_pipeline.rnn_state.bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))

                elif "gates" in filename:
                    if 'weights' in filename:
                        model.question_pipeline.rnn_state.weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))
                    elif "biases" in filename:
                        model.question_pipeline.rnn_state.bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.load(filepath)))


    # Finished transfering weights
    # Save training weights
    checkpoint = {
        'model_state_dict': model.get_cleaned_state_dict()
    }

    torch.save(checkpoint, '%s/model.pt.tar' % output_path)

