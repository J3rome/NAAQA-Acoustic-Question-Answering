import ujson
import os
import h5py
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from random import shuffle

from data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from utils import create_folder_if_necessary, save_json

# >>> Feature Extraction
def preextract_features(sess, dataset, network_wrapper, resnet_ckpt_path, sets=['train', 'val', 'test'], output_folder_name="preprocessed"):

    # FIXME: Config should be set automatically to raw when this option is used
    assert dataset.is_raw_img(), "Config must be set to raw image"

    input_image = network_wrapper.get_input_image()
    feature_extractor = network_wrapper.get_feature_extractor()
    feature_extractor_output_shape = [int(dim) for dim in feature_extractor.get_shape()[1:]]
    output_folder = "%s/%s" % (dataset.root_folder_path, output_folder_name)

    create_folder_if_necessary(output_folder)

    # We want to process each scene only one time (Keep only game per scene)
    dataset.keep_1_game_per_scene()

    sess.run(tf.global_variables_initializer())

    network_wrapper.restore_feature_extractor_weights(sess, resnet_ckpt_path)

    for set_type in sets:
        print("Extracting feature for set '%s'" % set_type)
        batches = dataset.get_batches(set_type)
        nb_games = batches.get_nb_games()
        output_filepath = "%s/%s_features.h5" % (output_folder, set_type)
        batch_size = batches.batch_size

        # TODO : Add check to see if file already exist
        with h5py.File(output_filepath, 'w') as f:
            h5_dataset = f.create_dataset('features', shape=[nb_games] + feature_extractor_output_shape, dtype=np.float32)
            h5_idx2img = f.create_dataset('idx2img', shape=[nb_games], dtype=np.int32)
            h5_idx = 0

            for batch in tqdm(batches):
                feed_dict = {
                    input_image: np.array(batch['image'])
                }
                features = sess.run(feature_extractor, feed_dict=feed_dict)

                h5_dataset[h5_idx: h5_idx + batch_size] = features

                for i, game in enumerate(batch['raw']):
                    h5_idx2img[h5_idx + i] = game.image.id

                h5_idx += batch_size

        print("%s set features extracted to '%s'." % (set_type, output_filepath))

    save_json({"extracted_feature_shape" : feature_extractor_output_shape},
              output_folder, 'feature_shape.json', indented=True)


# >>> Dictionary Creation (For word tokenization)
def create_dict_from_questions(dataset, word_min_occurence=1, dict_filename='dict.json'):
    # FIXME : Should we use the whole dataset to create the dictionary ?
    games = dataset.games['train']

    word2i = {'<padding>': 0,
              '<unk>': 1
              }
    word_index = max(word2i.values()) + 1

    answer2i = {  #'<padding>': 0,        # FIXME : Why would we need padding in the answers ?
        '<unk>': 0  # FIXME : We have no training example with unkonwn answer. Add Switch to remove unknown answer
    }
    answer_index = max(answer2i.values()) + 1

    answer2occ = dataset.answer_counter['train']
    word2occ = defaultdict(int)

    tokenizer = CLEARTokenizer.get_tokenizer_inst()

    # Tokenize questions
    for game in games:
        input_tokens = tokenizer.tokenize(game.question)
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

    print("Number of words: {}".format(len(word2i)))
    print("Number of answers: {}".format(len(answer2i)))

    preprocessed_folder_path = os.path.join(dataset.root_folder_path, 'preprocessed')

    if not os.path.isdir(preprocessed_folder_path):
        os.mkdir(preprocessed_folder_path)

    save_json({
            'word2i': word2i,
            'answer2i': answer2i
        }, preprocessed_folder_path, dict_filename)


if __name__ == "__main__":
    print("To run preprocessing, use main.py --preprocessing (or --create_dict, --feature_extract for individual steps)")
    exit(1)
