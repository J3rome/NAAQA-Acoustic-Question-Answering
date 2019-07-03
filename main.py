import argparse
from collections import defaultdict
from _datetime import datetime
import os
import subprocess

import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
import ujson

from utils import set_random_seed, create_folder_if_necessary, get_config, process_predictions
from utils import create_symlink_to_latest_folder, save_training_stats, save_inference_results
from utils import is_tensor_optimizer, is_tensor_prediction, is_tensor_scalar

from models.film_network_wrapper import FiLM_Network_Wrapper
from data_interfaces.CLEAR_dataset import CLEARDataset
from data_interfaces.CLEAR_tokenizer import CLEARTokenizer

parser = argparse.ArgumentParser('FiLM model for CLEAR Dataset (Acoustic Question Answering)', fromfile_prefix_chars='@')

parser.add_argument("--training", help="FiLM model training", action='store_true')
parser.add_argument("--inference", help="FiLM model inference", action='store_true')
parser.add_argument("--preprocessing", help="Data preprocessing (Word tokenization & Feature Pre-Extraction",
                    action='store_true')
parser.add_argument("--feature_extract", help="Feature Pre-Extraction", action='store_true')
parser.add_argument("--create_dict", help="Create word dictionary (for tokenization)", action='store_true')

# Input parameters
parser.add_argument("--data_root_path", type=str, default='data', help="Directory with data")
parser.add_argument("--version_name", type=str, help="Name of the dataset version")
parser.add_argument("--resnet_ckpt_path", type=str, default=None, help="Path to resnet-101 ckpt file")
parser.add_argument("--film_ckpt_path", type=str, default=None, help="Path to Film pretrained ckpt file")
parser.add_argument("--config_path", type=str, default='config/film.json', help="Path to Film pretrained ckpt file")         # FIXME : Add default value
parser.add_argument("--inference_set", type=str, default='test', help="Define on which set the inference should be runned")
# TODO : Add option for custom test file

# Output parameters
parser.add_argument("-output_root_path", type=str, default='output', help="Directory with image")

# Other parameters
parser.add_argument("--nb_epoch", type=int, default=5, help="Nb of epoch for training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (For training and inference)")
parser.add_argument("--random_seed", type=int, default=None, help="Random seed used for the experiment")


# TODO : Arguments Handling
#   Tasks :
#       Train from extracted features (No resnet - preprocessed)
#       Train from Raw Images (Resnet with fixed weights)
#       Train from Raw Images (Resnet or similar retrained (Only firsts couple layers))
#       Run from extracted features (No resnet - preprocessed)
#       Run from Raw Images (Resnet With fixed weights)
#       Run from Raw Images with VISUALIZATION
#


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

    with open('%s/feature_shape.json' % output_folder, 'w') as f:
        ujson.dump({
            "extracted_feature_shape" : feature_extractor_output_shape
        }, f, indent=2)


# >>> Training
def do_one_epoch(sess, batchifier, network_wrapper, outputs_var, keep_results=False):
    # check for optimizer to define training/eval mode
    is_training = any([is_tensor_optimizer(x) for x in outputs_var])

    aggregated_outputs = defaultdict(lambda : [])
    processed_predictions = []

    for batch in tqdm(batchifier):

        feed_dict = network_wrapper.get_feed_dict(is_training, batch['question'], batch['answer'], batch['image'], batch['seq_length'])

        results = sess.run(outputs_var, feed_dict=feed_dict)

        for var, result in zip(outputs_var, results):
            if is_tensor_scalar(var) and var in outputs_var:
                aggregated_outputs[var].append(result)

            elif is_tensor_prediction(var) and keep_results:

                aggregated_outputs[var] = True

                processed_predictions += process_predictions(network_wrapper.get_dataset(), result, batch['raw'])

    for var in aggregated_outputs.keys():
        if is_tensor_scalar(var):
            aggregated_outputs[var] = np.mean(aggregated_outputs[var]).item()
        elif is_tensor_prediction(var):
            aggregated_outputs[var] = processed_predictions

    to_return = list(aggregated_outputs.values())

    if len(to_return) < 3:
        # This method should always return 3 results (Because of the way we unpack it)
        to_return += []

    return to_return


def do_film_training(sess, dataset, network_wrapper, optimizer_config, resnet_ckpt_path, nb_epoch, output_folder, keep_results=True):
    stats_file_path = "%s/stats.json" % output_folder

    # Setup optimizer (For training)
    optimize_step, [loss, accuracy] = network_wrapper.create_optimizer(optimizer_config, var_list=None)  # TODO : Var_List should contain only film variables

    prediction = network_wrapper.get_network_prediction()

    # FIXME : This should be wrapped inside the wrapper (Maybe a initializer function ?)
    # We update the film variables because the adam variables weren't there when we first defined them
    network_wrapper.film_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="clear")
    optimizer_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="beta")

    sess.run(tf.variables_initializer(network_wrapper.film_variables))
    sess.run(tf.variables_initializer(optimizer_variables))
    #sess.run(tf.global_variables_initializer())

    if dataset.is_raw_img():
        sess.run(tf.variables_initializer(network_wrapper.feature_extractor_variables))
        network_wrapper.restore_feature_extractor_weights(sess, resnet_ckpt_path)

    ##################################
    # TESTING
    #stem_conv_weights = [v for v in tf.trainable_variables() if v.name == 'clear/stem/stem_conv/weights:0'][0]
    #stem_conv_weights = sess.run(stem_conv_weights)
    #stem_conv_weights = stem_conv_weights.tolist()

    #with open('saved_stem_conv_weights_full4.json', 'w') as f:
    #    json.dump(stem_conv_weights, f, indent=2)

    #moving_parameters = [v for v in tf.global_variables() if ('moving' in v.name) and 'Adam' not in v.name]
    #gamma_beta_parameters = [v for v in tf.global_variables() if ('gamma' in v.name or 'beta' in v.name) and 'Adam' not in v.name and 'power' not in v.name]

    ##################################

    stats = []

    # Training Loop
    for epoch in range(nb_epoch):
        epoch_output_folder_path = "%s/Epoch_%.2d" % (output_folder, epoch)
        create_folder_if_necessary(epoch_output_folder_path)

        print("Epoch %d" % epoch)
        train_loss, train_accuracy, train_predictions = do_one_epoch(sess, dataset.get_batches('train'),
                                                                     network_wrapper,
                                                                     [loss, accuracy, prediction, optimize_step],
                                                                     keep_results=keep_results)

        print("Training :")
        print("    Loss : %f  - Accuracy : %f" % (train_loss, train_accuracy))

        # FIXME : Inference of validation set doesn't yield same result.
        # FIXME : Should val batches be shuffled ?
        # FIXME : Validation accuracy is skewed by the batch padding. We could recalculate accuracy from the prediction (After removing padded ones)
        val_loss, val_accuracy, val_predictions = do_one_epoch(sess, dataset.get_batches('val', shuffled=False),
                                                               network_wrapper, [loss, accuracy, prediction],
                                                               keep_results=keep_results)

        print("Validation :")
        print("    Loss : %f  - Accuracy : %f" % (val_loss, val_accuracy))

        save_training_stats(stats_file_path, epoch, train_accuracy, train_loss, val_accuracy, val_loss)

        if keep_results:
            save_inference_results(train_predictions, epoch_output_folder_path, filename="train_inferences.json")
            save_inference_results(val_predictions, epoch_output_folder_path, filename="val_inferences.json")

        stats.append({
            'epoch' : epoch,
            'train_loss' : train_loss,
            'train_accuracy': train_accuracy,
            'val_loss' : val_loss,
            'val_accuracy' : val_accuracy
        })

        network_wrapper.save_film_checkpoint(sess, "%s/checkpoint.ckpt" % epoch_output_folder_path)

    # Create a symlink to best epoch output folder
    best_epoch = sorted(stats, key=lambda s: s['val_accuracy'], reverse=True)[0]['epoch']
    subprocess.run("cd %s && ln -s Epoch_%.2d best" % (output_folder, best_epoch), shell=True)


# >>> Inference
def do_test_inference(sess, dataset, network_wrapper, output_folder, film_ckpt_path, resnet_ckpt_path, set_name="test"):
    test_batches = dataset.get_batches(set_name, shuffled=False)

    sess.run(tf.variables_initializer(network_wrapper.film_variables))
    #sess.run(tf.global_variables_initializer())

    if dataset.is_raw_img():
        sess.run(tf.variables_initializer(network_wrapper.feature_extractor_variables))
        network_wrapper.restore_feature_extractor_weights(sess, resnet_ckpt_path)

    network_wrapper.restore_film_network_weights(sess, film_ckpt_path)

    processed_results = []

    for batch in tqdm(test_batches):
        feed_dict = network_wrapper.get_feed_dict(False, batch['question'], batch['answer'],
                                                  batch['image'], batch['seq_length'])

        results = sess.run(network_wrapper.get_network_prediction(), feed_dict=feed_dict)

        processed_results += process_predictions(dataset, results, batch['raw'])

    # Batches are required to have always the same size.
    # We don't want the batch padding to interfere with the test accuracy.
    # We removed the padded (duplicated) examples
    if test_batches.nb_padded_in_last_batch > 0:
        processed_results = processed_results[:-test_batches.nb_padded_in_last_batch]

    nb_correct = sum(1 for r in processed_results if r['correct'])
    nb_results = len(processed_results)
    accuracy = nb_correct/nb_results

    save_inference_results(processed_results, output_folder)

    print("Test set accuracy : %f" % accuracy)


# >>> Dictionary Creation (For word tokenization)
def create_dict_from_questions(dataset, word_min_occurence=1, dict_filename='dict.json'):
    # FIXME : Should we use the whole dataset to create the dictionary ?
    games = dataset.games['train']

    word2i = {'<padding>': 0,
              '<unk>': 1
              }

    answer2i = {  # '<padding>': 0,        # FIXME : Why would we need padding in the answers ?
        '<unk>': 0  # FIXME : We have no training example with unkonwn answer. Add Switch to remove unknown answer
    }

    answer2occ = dataset.answer_counter['train']
    word2occ = defaultdict(int)

    tokenizer = CLEARTokenizer.get_tokenizer_inst()

    for game in games:
        input_tokens = tokenizer.tokenize(game.question)
        for tok in input_tokens:
            word2occ[tok] += 1

    # parse the questions
    for word, occ in word2occ.items():
        if occ >= word_min_occurence:
            word2i[word] = len(word2i)

    # parse the answers
    for answer in answer2occ.keys():
        answer2i[answer] = len(answer2i)

    print("Number of words: {}".format(len(word2i)))
    print("Number of answers: {}".format(len(answer2i)))

    preprocessed_folder_path = os.path.join(dataset.root_folder_path, 'preprocessed')
    dict_file_path = os.path.join(preprocessed_folder_path, dict_filename)

    if not os.path.isdir(preprocessed_folder_path):
        os.mkdir(preprocessed_folder_path)

    with open(dict_file_path, 'w') as f:
        ujson.dump({
            'word2i': word2i,
            'answer2i': answer2i
        }, f, indent=2)


def main(args):

    if 0 < sum([args.training, args.preprocessing, args.inference, args.feature_extract, args.create_dict]) > 1:
        print("[ERROR] Can only do one task at a time (--training, --preprocessing or --inference)")
        exit(1)

    is_preprocessing = False

    if args.training:
        task = "train_film"
    elif args.inference:
        task = "inference"
    elif args.preprocessing:
        task = "full_preprocessing"
        is_preprocessing = True
    elif args.feature_extract:
        task = "feature_extract"
        is_preprocessing = True
    elif args.create_dict:
        task = "create_dict"
        is_preprocessing = True

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # Paths
    data_path = "%s/%s" % (args.data_root_path, args.version_name)

    output_task_folder = "%s/%s" % (args.output_root_path, task)
    output_experiment_folder = "%s/%s" %(output_task_folder, args.version_name)
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%Hh%M")
    output_dated_folder = "%s/%s" % (output_experiment_folder, current_datetime_str)

    if args.resnet_ckpt_path is None:
        args.resnet_ckpt_path = "%s/resnet/resnet_v1_101.ckpt" % args.data_root_path

    if args.film_ckpt_path is None:
        #experiment_date = "2019-06-23_16h37"
        experiment_date = "latest"
        args.film_ckpt_path = "%s/train_film/%s/%s/best/checkpoint.ckpt" % (args.output_root_path, args.version_name, experiment_date)

    film_model_config = get_config(args.config_path)

    create_output_folder = not is_preprocessing

    if create_output_folder:
        # TODO : See if this is optimal file structure
        create_folder_if_necessary(args.output_root_path)
        create_folder_if_necessary(output_task_folder)
        create_folder_if_necessary(output_experiment_folder)
        create_folder_if_necessary(output_dated_folder)
        create_symlink_to_latest_folder(output_experiment_folder, current_datetime_str)

    ########################################################
    ################### Data Loading #######################
    ########################################################
    dataset = CLEARDataset(data_path, film_model_config['input'],
                           batch_size=args.batch_size, tokenize_text=not is_preprocessing)

    ########################################################
    ################## Network Setup #######################
    ########################################################
    network_wrapper = FiLM_Network_Wrapper(film_model_config, dataset, preprocessing=is_preprocessing)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
        # Debugging Tools
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "T480s:8076")
        tensorboard_writer = tf.summary.FileWriter('test_resnet_logs', sess.graph)   #FIXME : Make the path parametrable ?

        if task == "train_film":
            do_film_training(sess, dataset, network_wrapper, film_model_config['optimizer'],
                             args.resnet_ckpt_path, args.nb_epoch, output_dated_folder)
        elif task == "inference":
            do_test_inference(sess, dataset, network_wrapper, output_dated_folder,
                              args.film_ckpt_path, args.resnet_ckpt_path, set_name=args.inference_set)
        elif task == "feature_extract":
            preextract_features(sess, dataset, network_wrapper, args.resnet_ckpt_path)
        elif task == "create_dict":
            create_dict_from_questions(dataset)
        elif task == "full_preprocessing":
            create_dict_from_questions(dataset)
            preextract_features(sess, dataset, network_wrapper, args.resnet_ckpt_path)

        time_elapsed = str(datetime.now() - current_datetime)

        print("Execution took %s" % time_elapsed)

        # TODO : Export Gamma & Beta
        # TODO : Export visualizations

    if create_output_folder:
        with open('%s/config_%s.json' % (output_dated_folder, film_model_config['input']['type']), 'w') as f:
            ujson.dump(film_model_config, f, indent=2)

        with open('%s/timing.json' % output_dated_folder, 'w') as f:
            ujson.dump({'time_elapsed': time_elapsed}, f, indent=2)

        print("All Done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

        # TODO : Extract Beta And Gamma Parameters + T-SNE
        # TODO : Feed RAW image directly to the FiLM network
        # TODO : Quantify time cost of using raw images vs preprocessed conv features
        # TODO : Resize images ? Do we absolutely need 224x224 for the resnet preprocessing ?
        #        Since we extract a layer in resnet, we can feed any size, its fully convolutional up to that point
        #        When feeding directly to FiLM, we can use original size ?
        # TODO : What is the optimal size for our spectrograms ?
        # TODO : Train with different amount of residual blocks. Other modifications to the architecture ?



