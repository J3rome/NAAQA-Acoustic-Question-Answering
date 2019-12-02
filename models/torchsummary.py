import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np


def summary(model, input_infos, batch_size=-1, device="cuda:0"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()

            if isinstance(input[0], tuple):
                input_shape = []
                for inp in input[0]:
                    shape = list(inp.size())
                    shape[0] = batch_size
                    input_shape.append(shape)
            else:
                input_shape = list(input[0].size())
                input_shape[0] = batch_size

            summary[m_key]["input_shape"] = input_shape
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
            and not (hasattr(module, "is_container") and module.is_container)
        ):
            hooks.append(module.register_forward_hook(hook))

    assert 'cpu' in device or (torch.cuda.is_available() and 'cuda' in device), \
        "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device.startswith("cuda"):
        dtype_prefix = torch.cuda
    else:
        dtype_prefix = torch

    # multiple inputs to the network
    #if isinstance(input_size, tuple):
    #    input_size = [input_size]

    #model.to(device)

    # batch_size of 2 for batchnorm
    x = []
    for input_size, input_type in input_infos:
        # FIXME : Test on GPU
        x.append(torch.ones(2, *input_size).type(input_type).to(device))

    if len(x) == 1:
        x = x[0]

    #x.append(torch.ones(2, *input_size[0]).type(dtype_prefix.LongTensor).to(device))
    #x.append(torch.ones(2).type(dtype_prefix.LongTensor).to(device) + 1)
    #x.append(torch.ones(2, *input_size[2]).type(dtype_prefix.FloatTensor).to(device))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x, pack_sequence=False)

    # remove these hooks
    for h in hooks:
        h.remove()

    line_new = "{:>20} {:>35} {:>30} {:>15}".format("Layer (type)", "Input Shape", "Output Shape", "Param #")
    nb_char_per_line = len(line_new)
    print("-" * nb_char_per_line)
    print(line_new)
    print("=" * nb_char_per_line)
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20} {:>35} {:>30} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    #total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_input_size = abs(np.sum([np.prod(in_tuple) for in_tuple in input_size]) * batch_size * 4. / (1024 ** 2.))

    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("=" * nb_char_per_line)
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("-" * nb_char_per_line)
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("-" * nb_char_per_line)
    # return summary
