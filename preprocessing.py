import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

from data_interfaces.CLEAR_dataset import CLEARTokenizer, CLEAR_collate_fct, CLEAR_dataset
from utils.file import create_folder_if_necessary, save_json, read_json
from utils.processing import calc_dataset_stats

import torch
import torch.nn as nn

from models.tools.lr_finder import LRFinder
import matplotlib.pyplot as plt


def get_lr_finder_curves(model, device, train_dataloader, output_dated_folder, num_iter, optimizer, val_dataloader=None,
                         loss_criterion=nn.CrossEntropyLoss(), weight_decay_list=None, min_lr=1e-10, show_fig=False):
    if type(weight_decay_list) != list:
        weight_decay_list = [0., 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7]

    # TODO : The order of the data probably affect the LR curves. Shuffling and doing multiple time should help

    # FIXME : What momentum value should we use for SGD ???
    # Force momentum to 0
    initial_optimizer_state_dict = optimizer.state_dict()
    optimizer.param_groups[0]['momentum'] = 0
    optimizer.param_groups[0]['lr'] = min_lr

    fig, ax = plt.subplots()
    lr_finder = LRFinder(model, optimizer, loss_criterion, device=device)

    num_iter_val = int(num_iter * 0.20)

    for weight_decay in weight_decay_list:
        # Reset LR Finder and change weight decay
        lr_finder.reset(weight_decay=weight_decay)

        print(f"Learning Rate finder -- Running for {num_iter} batches with weight decay : {weight_decay:.5}")
        # FIXME : Should probably run with validation data?
        losses_per_lr = lr_finder.range_test(train_dataloader, val_loader=val_dataloader, end_lr=num_iter_val,
                                             num_iter=num_iter, num_iter_val=num_iter_val)

        fig, ax = lr_finder.plot(fig_ax=(fig, ax), legend_label=f"Weight Decay : {weight_decay:.5}", show_fig=False)

        save_json(losses_per_lr, output_dated_folder, f'lr_finder_weight_decay_{weight_decay:.5}.json')

    if show_fig:
        plt.show()

    filepath = "%s/%s" % (output_dated_folder, 'lr_finder_plot.png')
    fig.savefig(filepath)

    # Reset optimiser config
    optimizer.load_state_dict(initial_optimizer_state_dict)

    return fig, ax


def get_dataset_stats_and_write(dataset, device, stats_filepath=None, recalculate=False, batch_size=1,
                                nb_dataloader_worker=0):
    if stats_filepath is None:
        stats_filepath = f"{dataset.root_folder_path}/{dataset.preprocessed_folder_name}/clear_stats.json"

    if os.path.exists(stats_filepath) and not recalculate:
        print(f"Loading CLEAR stats from {stats_filepath}")
        return read_json(stats_filepath)

    dataset_copy = dataset.__class__.from_dataset_object(dataset, dataset.games)

    dataset_copy.keep_1_game_per_scene()

    dataloader = torch.utils.data.DataLoader(dataset_copy, batch_size=batch_size, shuffle=False, num_workers=nb_dataloader_worker,
                                             collate_fn=CLEAR_collate_fct(padding_token=
                                                                          dataset_copy.get_padding_token()))

    mean, std, min_val, max_val = calc_dataset_stats(dataloader, device=device)

    stats = {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val
    }
    save_json(stats, stats_filepath)

    print(f"Clear stats written to '{stats_filepath}'")

    return stats


# >>> Feature Extraction
def extract_features(device, feature_extractor, dataloaders, output_folder_name='preprocessed'):
    dataloaders_first_key = list(dataloaders.keys())[0]
    first_dataloader = dataloaders[dataloaders_first_key]

    assert first_dataloader.dataset.is_raw_img(), 'Input must be set to RAW in config to pre extract features.'

    data_path = first_dataloader.dataset.root_folder_path
    batch_size = first_dataloader.batch_size
    output_folder_path = '%s/%s' % (data_path, output_folder_name)

    # NOTE : When testing multiple dataset configurations, Images and questions are generated in separate folder and
    #        linked together so we don't have multiple copies of the dataset (And multiple preprocessing runs)
    #
    #        We use the default symlink to create the new folder at the correct destination so it is available
    #        to other configuration of the dataset (When extracting using different value of 'output_folder_name')
    #
    #        If "preprocessed" is not a symlink, 'output_folder_name' will be created in requested 'data_path'

    output_exist = os.path.exists(output_folder_path)
    preprocessed_default_folder_path = '%s/preprocessed' % data_path
    if not output_exist and os.path.exists(preprocessed_default_folder_path) and \
            os.path.islink(preprocessed_default_folder_path):

        # Retrieve paths from symlink
        default_link_value = os.readlink(preprocessed_default_folder_path)
        new_link_value = default_link_value.replace('preprocessed', output_folder_name)

        # Create folder in appropriate directory
        create_folder_if_necessary("%s/%s" % (data_path, new_link_value))

        # Create symlink in requested directory
        if not output_exist:
            os.symlink(new_link_value, output_folder_path)
    else:
        create_folder_if_necessary(output_folder_path)

    # Set model to eval mode
    feature_extractor.eval()

    for set_type, dataloader in dataloaders.items():
        print("Extracting features from '%s' set" % set_type)
        output_filepath = '%s/%s_features.h5' % (output_folder_path, set_type)

        # Retrieve min & max dims of images
        max_width_id, height, max_width = dataloader.dataset.get_max_width_image_dims(return_scene_id=True)
        game_id = dataloader.dataset.get_random_game_for_scene(max_width_id)
        max_width_img = dataloader.dataset[game_id]['image'].unsqueeze(0).to(device)
        feature_extractor_output_shape = feature_extractor.get_output_shape(max_width_img, channel_first=True)

        # Keep only 1 game per scene (We want to process every image only once)
        dataloader.dataset.keep_1_game_per_scene()

        nb_games = len(dataloader.dataset)

        with h5py.File(output_filepath, 'w') as f:
            h5_dataset = f.create_dataset('features', shape=[nb_games] + feature_extractor_output_shape, dtype=np.float32)
            h5_idx2img = f.create_dataset('idx2img', shape=[nb_games], dtype=np.int32)
            h5_img_padding = f.create_dataset('img_padding', shape=[nb_games, 2], dtype=np.int32)
            h5_idx = 0
            for batch in tqdm(dataloader):
                images = batch['image'].to(device)

                with torch.set_grad_enabled(False):
                    features = feature_extractor(images).detach().cpu().numpy()

                h5_dataset[h5_idx: h5_idx + batch_size] = features
                h5_img_padding[h5_idx: h5_idx + batch_size] = batch['image_padding']

                for i, scene_id in enumerate(batch['scene_id']):
                    h5_idx2img[h5_idx + i] = scene_id

                h5_idx += batch_size
        print("Features extracted succesfully to '%s'" % output_filepath)


def images_to_h5(device, dataloaders, output_folder_name='preprocessed', feature_extractor=None):
    dataloaders_first_key = list(dataloaders.keys())[0]
    first_dataloader = dataloaders[dataloaders_first_key]

    assert first_dataloader.dataset.is_raw_img(), 'Input must be set to RAW in config to pre extract features.'

    data_path = first_dataloader.dataset.root_folder_path
    batch_size = first_dataloader.batch_size
    output_folder_path = '%s/%s' % (data_path, output_folder_name)

    # NOTE : When testing multiple dataset configurations, Images and questions are generated in separate folder and
    #        linked together so we don't have multiple copies of the dataset (And multiple preprocessing runs)
    #
    #        We use the default symlink to create the new folder at the correct destination so it is available
    #        to other configuration of the dataset (When extracting using different value of 'output_folder_name')
    #
    #        If "preprocessed" is not a symlink, 'output_folder_name' will be created in requested 'data_path'

    output_exist = os.path.exists(output_folder_path)
    preprocessed_default_folder_path = '%s/preprocessed' % data_path
    if not output_exist and os.path.exists(preprocessed_default_folder_path) and \
            os.path.islink(preprocessed_default_folder_path):

        # Retrieve paths from symlink
        default_link_value = os.readlink(preprocessed_default_folder_path)
        new_link_value = default_link_value.replace('preprocessed', output_folder_name)

        # Create folder in appropriate directory
        create_folder_if_necessary("%s/%s" % (data_path, new_link_value))

        # Create symlink in requested directory
        if not output_exist:
            os.symlink(new_link_value, output_folder_path)
    else:
        create_folder_if_necessary(output_folder_path)

    if feature_extractor:
        # Set feature extractor to eval mode
        feature_extractor.to(device)
        feature_extractor.eval()
    else:
        input_channels = first_dataloader.dataset.get_input_shape()[0]

    for set_type, dataloader in dataloaders.items():
        print("Creating H5 file from '%s' set" % set_type)
        output_filepath = '%s/%s_features.h5' % (output_folder_path, set_type)

        # Retrieve min & max dims of images
        max_width_id, height, max_width = dataloader.dataset.get_max_width_image_dims(return_scene_id=True)

        if feature_extractor:
            game_id = dataloader.dataset.get_random_game_for_scene(max_width_id)
            max_width_img = dataloader.dataset[game_id]['image'].unsqueeze(0).to(device)
            image_dim = feature_extractor.get_output_shape(max_width_img, channel_first=True)
        else:
            image_dim = [input_channels, height, max_width]

        # Keep only 1 game per scene (We want to process every image only once)
        dataloader.dataset.keep_1_game_per_scene()

        nb_games = len(dataloader.dataset)

        with h5py.File(output_filepath, 'w') as f:
            # FIXME : Change dataset name ?
            h5_dataset = f.create_dataset('features', shape=[nb_games] + image_dim, dtype=np.float32)
            h5_idx2img = f.create_dataset('idx2img', shape=[nb_games], dtype=np.int32)
            h5_img_padding = f.create_dataset('img_padding', shape=[nb_games, 2], dtype=np.int32)
            h5_idx = 0
            for batch in tqdm(dataloader):
                features = batch['image']
                if feature_extractor:
                    features.to(device)
                    with torch.set_grad_enabled(False):
                        features = feature_extractor(features, spatial_location=[])

                if features.device != 'cpu':
                    features = features.detach().cpu()

                h5_dataset[h5_idx: h5_idx + batch_size] = features

                # FIXME : Padding is not good when using resnet
                h5_img_padding[h5_idx: h5_idx + batch_size] = batch['image_padding']

                for i, scene_id in enumerate(batch['scene_id']):
                    h5_idx2img[h5_idx + i] = scene_id

                h5_idx += batch_size

        print("Images extracted successfully to '%s'" % output_filepath)


# >>> Dictionary Creation (For word tokenization)
def create_dict_from_questions(dataset, word_min_occurence=1, dict_filename='dict.json', force_all_answers=False,
                               output_folder_name='preprocessed', start_end_tokens=True):
    word2i = {'<padding>': 0,
              '<unk>': 1
              }
    word_index = max(word2i.values()) + 1

    if start_end_tokens:
        word2i['<start>'] = word_index
        word2i['<end>'] = word_index + 1
        word_index += 2

    answer2i = {  #'<padding>': 0,        # FIXME : Why would we need padding in the answers ?
        #'<unk>': 0  # FIXME : We have no training example with unkonwn answer. Add Switch to remove unknown answer
    }

    if len(answer2i) > 0:
        answer_index = max(answer2i.values()) + 1
    else:
        answer_index = 0

    answers = [k.lower() for k in dataset.answer_counter.keys()]
    word2occ = defaultdict(int)

    tokenizer = CLEARTokenizer.get_tokenizer_inst()
    #forbidden_tokens = [",", "?"]
    forbidden_tokens = []

    # Tokenize questions
    for i in dataset.games.keys():
        game = dataset.get_game(i)
        input_tokens = [t.lower() for t in tokenizer.tokenize(game['question'])]
        for tok in input_tokens:
            if tok not in forbidden_tokens:
                word2occ[tok] += 1

    # Sort tokens then shuffle then to keep control over the order (to enhance reproducibility)
    sorted_words = sorted(word2occ.items(), key=lambda x: x[0])
    shuffle(sorted_words)

    for word_occ in sorted_words:
        if word_occ[1] >= word_min_occurence:
            word2i[word_occ[0]] = word_index
            word_index += 1

    if force_all_answers:
        all_answers = read_json(dataset.root_folder_path, 'attributes.json')

        all_answers = [a.lower() for answers in all_answers.values() for a in answers]

        padded_answers = []

        for answer in all_answers:
            if answer not in answers:
                answers.append(answer)
                padded_answers.append(answer)

        print("Padded dict with %d missing answers : " % len(padded_answers))
        print(padded_answers)

    sorted_answers = sorted(answers)
    shuffle(sorted_answers)
    # parse the answers
    for answer in sorted_answers:
        answer2i[answer] = answer_index
        answer_index += 1

    print("Number of words: {}".format(len(word2i)))
    print("Number of answers: {}".format(len(answer2i)))

    preprocessed_folder_path = os.path.join(dataset.root_folder_path, output_folder_name)

    if not os.path.isdir(preprocessed_folder_path):
        os.mkdir(preprocessed_folder_path)

    save_json({
            'word2i': word2i,
            'answer2i': answer2i
        }, preprocessed_folder_path, dict_filename)


if __name__ == "__main__":
    print("To run preprocessing, use main.py --preprocessing (or --create_dict, --feature_extract for individual steps)")
    exit(1)
