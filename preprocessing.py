import os
import h5py
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

from data_interfaces.CLEAR_dataset import CLEARTokenizer
from utils import create_folder_if_necessary, save_json, read_json, set_random_state, get_random_state

from torch.utils.data import DataLoader
import torch


# >>> Feature Extraction
def extract_features(device, feature_extractor, dataloaders, output_folder_name='preprocessed'):
    dataloaders_first_key = list(dataloaders.keys())[0]
    first_dataloader = dataloaders[dataloaders_first_key]

    assert first_dataloader.dataset.is_raw_img(), 'Input must be set to RAW in config to pre extract features.'

    data_path = first_dataloader.dataset.root_folder_path
    batch_size = first_dataloader.batch_size
    output_folder_path = '%s/%s' % (data_path, output_folder_name)
    create_folder_if_necessary(output_folder_path)

    # Set model to eval mode
    feature_extractor.eval()

    for set_type, dataloader in dataloaders.items():
        print("Extracting features from '%s' set" % set_type)
        output_filepath = '%s/%s_features.h5' % (output_folder_path, set_type)

        # Retrieve min & max dims of images
        max_width_id, height, max_width = dataloader.dataset.get_max_width_image_dims(return_scene_id=True)
        game_id = dataloader.dataset.get_game_id_for_scene(max_width_id)
        max_width_img = dataloader.dataset[game_id]['image'].unsqueeze(0)
        feature_extractor_output_shape = feature_extractor.get_output_shape(max_width_img, channel_first=False)

        # Keep only 1 game per scene (We want to process every image only once)
        dataloader.dataset.keep_1_game_per_scene()

        nb_games = len(dataloader.dataset)

        with h5py.File(output_filepath, 'w') as f:
            # FIXME : Find a way to have variable size. MaxShape is not the answer
            #         We can use --pad_to_largest image, save a dataset of padding and remove padding when retrieving <<-- This won't work, the output shape is different after passing throught the feature extractor. Padding at this level will have a big impact
            h5_dataset = f.create_dataset('features', shape=[nb_games] + feature_extractor_output_shape, dtype=np.float32)
            h5_idx2img = f.create_dataset('idx2img', shape=[nb_games], dtype=np.int32)
            h5_idx = 0
            for batch in tqdm(dataloader):
                images = batch['image'].to(device)

                with torch.set_grad_enabled(False):
                    features = feature_extractor(images).detach().cpu().numpy()

                # swap axis
                # numpy image: H x W x C
                # torch image: C X H X W
                # We want to save in numpy format
                features = features.transpose((0, 2, 3, 1))

                h5_dataset[h5_idx: h5_idx + batch_size] = features

                for i, scene_id in enumerate(batch['scene_id']):
                    h5_idx2img[h5_idx + i] = scene_id

                h5_idx += batch_size
        print("Features extracted succesfully to '%s'" % output_filepath)

    # Save the extracted feature shape
    save_json({"extracted_feature_shape": feature_extractor_output_shape}, output_folder_path,
              filename='feature_shape.json')


# >>> Dictionary Creation (For word tokenization)
def create_dict_from_questions(dataset, word_min_occurence=1, dict_filename='dict.json', force_all_answers=False,
                               output_folder_name='preprocessed'):
    # FIXME : Should we use the whole dataset to create the dictionary ?
    games = dataset.games

    word2i = {'<padding>': 0,
              '<unk>': 1
              }
    word_index = max(word2i.values()) + 1

    answer2i = {  #'<padding>': 0,        # FIXME : Why would we need padding in the answers ?
        '<unk>': 0  # FIXME : We have no training example with unkonwn answer. Add Switch to remove unknown answer
    }
    answer_index = max(answer2i.values()) + 1

    answer2occ = dataset.answer_counter
    word2occ = defaultdict(int)

    tokenizer = CLEARTokenizer.get_tokenizer_inst()

    # Tokenize questions
    for i in range(len(games)):
        game = dataset.get_game(i)
        input_tokens = tokenizer.tokenize(game['question'])
        for tok in input_tokens:
            word2occ[tok] += 1

    # Sort tokens then shuffle then to keep control over the order (to enhance reproducibility)
    sorted_words = sorted(word2occ.items(), key=lambda x: x[0])
    shuffle(sorted_words)

    for word_occ in sorted_words:
        if word_occ[1] >= word_min_occurence:
            word2i[word_occ[0]] = word_index
            word_index += 1

    sorted_answers = sorted(answer2occ.keys())
    shuffle(sorted_answers)
    # parse the answers
    for answer in sorted_answers:
        answer2i[answer] = answer_index
        answer_index += 1

    if force_all_answers:
        all_answers = read_json(dataset.root_folder_path, 'attributes.json')

        all_answers = [a for answers in all_answers.values() for a in answers]

        padded_answers = []

        for answer in all_answers:
            if answer not in answer2i:
                answer2i[answer] = answer_index
                answer_index += 1
                padded_answers.append(answer)

        print("Padded dict with %d missing answers : " % len(padded_answers))
        print(padded_answers)

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
