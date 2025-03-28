import os

import numpy as np
import torch

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