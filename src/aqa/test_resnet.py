import tensorflow as tf
from tqdm import tqdm
import numpy as np
import h5py
from collections import defaultdict
from _datetime import datetime
import json
import os
import subprocess
import random
from collections import OrderedDict

from tensorflow.python import debug as tf_debug

from aqa.models.film_network import FiLM_Network
from aqa.data_interfaces.CLEAR_tokenizer import CLEARTokenizer
from aqa.models.film_network_wrapper import FiLM_Network_Wrapper
from aqa.data_interfaces.CLEAR_dataset import CLEARDataset, CLEARBatchifier
from aqa.model_handlers.optimizer import create_optimizer
from aqa.models.resnet import create_resnet




# TODO : Arguments Handling
#   Tasks :
#       Train from extracted features (No resnet - preprocessed)
#       Train from Raw Images (Resnet with fixed weights)
#       Train from Raw Images (Resnet or similar retrained (Only firsts couple layers))
#       Run from extracted features (No resnet - preprocessed)
#       Run from Raw Images (Resnet With fixed weights)
#       Run from Raw Images with VISUALIZATION
#
#   Parameters (USE THE CONFIG FILE, keep number of parameters down) :
#       experiment_name
#       resnet_cpkt
#       resnet_chosen_layer
#       dict_path
#
#       config_file_path
#       mode (Task...train,run,visualization,etc)
#       tensorboard_log_dir (If None, no logging)
#       Output Path (Trained models, visualization, etc)
#

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)


# TODO : Move some place else
# TODO check if optimizers are always ops? Maybe there is a better check
def is_optimizer(x):
    return hasattr(x, 'op_def')

def is_summary(x):
    return (isinstance(x, tf.Tensor) and x.dtype is tf.string)


def is_float(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32


def is_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0


def is_list_int(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.int64 and len(x.shape) == 1


def is_prediction(x):
    return isinstance(x, tf.Tensor) and 'predicted_answer' in x.name


def create_folder_if_necessary(folder_path, overwrite_folder=False):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    elif overwrite_folder:
        os.rmdir(folder_path)
        os.mkdir(folder_path)


def save_training_stats(stats_output_file, epoch_nb, train_accuracy, train_loss, val_accuracy, val_loss):
    """
    Will read the stats file from disk and append new epoch stats (Will create the file if not present)
    """
    if os.path.isfile(stats_output_file):
        with open(stats_output_file, 'r') as f:
            stats = json.load(f)
    else:
        stats = []

    stats.append({
        'epoch': "epoch_%.3d" % (epoch_nb + 1),
        'train_acc': train_accuracy,
        'train_loss': train_loss,
        'val_accuracy': val_accuracy,
        'val_loss': val_loss
    })

    stats = sorted(stats, key=lambda e: e['val_loss'])

    with open(stats_output_file, 'w') as f:
        json.dump(stats, f, indent=2, sort_keys=True)


def do_one_epoch(sess, batchifier, network_wrapper, outputs_var, keep_results=False):
    # check for optimizer to define training/eval mode
    is_training = any([is_optimizer(x) for x in outputs_var])

    aggregated_outputs = defaultdict(lambda : [])
    processed_predictions = []

    for batch in tqdm(batchifier):

        feed_dict = network_wrapper.get_feed_dict(is_training, batch['question'], batch['answer'], batch['image'], batch['seq_length'])

        results = sess.run(outputs_var, feed_dict=feed_dict)

        for var, result in zip(outputs_var, results):
            if is_scalar(var) and var in outputs_var:
                aggregated_outputs[var].append(result)

            elif is_prediction(var) and keep_results:

                aggregated_outputs[var] = True

                for i, res, game in zip(range(len(result)), result, batch['raw']):
                    processed_predictions.append({
                        'question_id': batch['raw'][i].id,
                        'scene_id': batch['raw'][i].image.id,
                        'answer': batchifier.tokenizer.decode_answer(res),
                        'ground_truth': batchifier.tokenizer.decode_answer(batch['raw'][i].answer),
                        'correct': bool(res == batch['raw'][i].answer)
                    })

    for var in aggregated_outputs.keys():
        if is_scalar(var):
            aggregated_outputs[var] = np.mean(aggregated_outputs[var]).item()
        elif is_prediction(var):
            aggregated_outputs[var] = processed_predictions

    to_return = list(aggregated_outputs.values())

    if len(to_return) < 3:
        # This method should always return 3 results (Because of the way we unpack it)
        to_return += []

    return to_return


def do_film_training(sess, dataset, network_wrapper, optimizer_config, nb_epoch, output_folder, keep_results=True):
    stats_file_path = "%s/stats.json" % output_folder

    # Setup optimizer (For training)
    optimize_step, [loss, accuracy] = create_optimizer(network_wrapper.get_network(), optimizer_config, var_list=None)  # TODO : Var_List should contain only film variables

    prediction = network_wrapper.get_network_prediction()

    sess.run(tf.global_variables_initializer())

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

        val_loss, val_accuracy, val_predictions = do_one_epoch(sess, dataset.get_batches('val'),
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

def save_inference_results(results, output_folder, filename="results.json"):
    with open("%s/%s" % (output_folder, filename), 'w') as f:
        json.dump(results, f, indent=2)

def do_test_inference(sess, dataset, network_wrapper, output_folder, set_name="test"):
    test_batches = dataset.get_batches(set_name)

    sess.run(tf.global_variables_initializer())

    processed_results = []

    for batch in tqdm(test_batches):
        feed_dict = network_wrapper.get_feed_dict(False, batch['question'], batch['answer'],
                                                  batch['image'], batch['seq_length'])

        results = sess.run(network_wrapper.get_network_prediction(), feed_dict=feed_dict)

        for i, result in enumerate(results):
            processed_results.append({
                'question_id' : batch['raw'][i].id,
                'scene_id': batch['raw'][i].image.id,
                'answer': dataset.tokenizer.decode_answer(result),
                'ground_truth': dataset.tokenizer.decode_answer(batch['raw'][i].answer),
                'correct': bool(result == batch['raw'][i].answer)
            })

    nb_correct = sum(1 for r in processed_results if r['correct'])
    nb_results = len(processed_results)
    accuracy = nb_correct/nb_results

    save_inference_results(processed_results, output_folder)

    print("Test set accuracy : %f" % accuracy)


def do_visualization():
    return 1


def preextract_features(sess, dataset, network_wrapper, sets=['train', 'val', 'test'], output_folder_name="preprocessed"):

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

    for set_type in sets:
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
        json.dump({
            "extracted_feature_shape" : feature_extractor_output_shape
        }, f, indent=2)


def main():
    # TODO : Seed management
    #task = "train_film"
    #task = "test_inference"
    task = "preextract_features"

    # Parameters
    nb_epoch = 5
    nb_thread = 2
    batch_size = 32
    seed = 667

    # Paths
    root_folder = "data"
    experiment_name = "v2.0.0_1k_scenes_1_inst_per_scene"
    experiment_path = "%s/%s" % (root_folder, experiment_name)
    resnet_ckpt_path = "%s/resnet/resnet_v1_101.ckpt" % root_folder

    output_root_folder = "output"
    output_task_folder = "%s/%s" % (output_root_folder, task)
    output_experiment_folder = "%s/%s" %(output_task_folder, experiment_name)
    now = datetime.now()
    output_dated_folder = "%s/%s" % (output_experiment_folder, now.strftime("%Y-%m-%d_%Hh%M"))

    experiment_date = "2019-06-13_02h46"
    film_ckpt_path = "%s/train_film/%s/%s/best/checkpoint.ckpt" % (output_root_folder, experiment_name, experiment_date)

    # TODO : Output folder should contains info about the config used
    # TODO : Read config from file
    film_model_config = {
        "input": {
            "type": "raw",
            "dim": [224, 224, 3],
            #"type": "conv",
            #"dim": [224, 224, 3],   # TODO : Those should be infered from the shape of input tensor
        },
        "feature_extractor": {
            "type": "resnet",
            "version": 101,
            "output_layer": "block3/unit_22/bottleneck_v1"
        },
        "question": {
            "word_embedding_dim": 200,
            "rnn_state_size": 4096
        },
        "stem": {
            "spatial_location": True,
            "conv_out": 128,
            "conv_kernel": [3, 3]
        },
        "resblock": {
            "no_resblock": 4,
            "spatial_location": True,
            "kernel1": [1, 1],
            "kernel2": [3, 3]
        },
        "classifier": {
            "spatial_location": True,
            "conv_out": 512,
            "conv_kernel": [1, 1],
            "no_mlp_units": 1024
        },
        'optimizer' : {
            "learning_rate": 3e-4,
            "clip_val": 0.0,
            "weight_decay": 1e-5
        }
    }



    restore_feature_extractor_weights = True if (task == "train_film" and film_model_config['input']['type'] == 'raw') or "inference" in task else False
    restore_film_weights = True if "inference" in task else False
    create_output_folder = True if not 'pre' in task else False

    if seed is not None:
        set_random_seed(seed)

    if create_output_folder:
        # TODO : See if this is optimal file structure
        # Creating output folders       # TODO : Might not want to creat all the folders all the time
        create_folder_if_necessary(output_root_folder)
        create_folder_if_necessary(output_task_folder)
        create_folder_if_necessary(output_experiment_folder)
        create_folder_if_necessary(output_dated_folder)

    ########################################################
    ################### Data Loading #######################
    ########################################################
    dataset = CLEARDataset(experiment_path, film_model_config['input'], batch_size=batch_size)

    ########################################################
    ################## Network Setup #######################
    ########################################################
    network_wrapper = FiLM_Network_Wrapper(film_model_config, dataset)

    # GPU Options
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        # Debugging Tools
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "T480s:8076")
        tensorboard_writer = tf.summary.FileWriter('test_resnet_logs', sess.graph)

        if restore_feature_extractor_weights:
            network_wrapper.restore_feature_extractor_weights(sess, resnet_ckpt_path)

        if restore_film_weights:
            network_wrapper.restore_film_network_weights(sess, film_ckpt_path)

        if task == "train_film":
            do_film_training(sess, dataset, network_wrapper, film_model_config['optimizer'], nb_epoch, output_dated_folder)
        elif task == "test_inference":
            do_test_inference(sess, dataset, network_wrapper,output_dated_folder)
        elif task == "preextract_features":
            preextract_features(sess, dataset, network_wrapper)

        # TODO : Export Gamma & Beta
        # TODO : Export visualizations

    if create_output_folder:
        with open('%s/config_%s.json' % (output_dated_folder, film_model_config['input']['type']), 'w') as f:
            json.dump(film_model_config, f, indent=2)

        print("All Done")

if __name__ == "__main__":
    main()



        # TODO : Restore pretrained FiLM network
        # TODO : Extract Beta And Gamma Parameters + T-SNE
        # TODO : Feed RAW image directly to the FiLM network
        # TODO : Options for preprocessing (Feature Exctraction) to minimize training time ? ( Quantify the increase in training time)
        # TODO : MAKE SURE WE GET THE SAME OUTPUT WHEN USING PREPROCESSED FEATURES AND RAW IMAGES
        # TODO : See how using the full resnet + FiLM impact batch size (More parameters than with preprocessing)
        # TODO : Resize images ? Do we absolutely need 224x224 for the resnet preprocessing ?
        #        Since we extract a layer in resnet, we can feed any size, its fully convolutional up to that point
        #        When feeding directly to FiLM, we can use original size ?
        # TODO : What is the optimal size for our spectrograms ?
        # TODO : Train with different amount of residual blocks. Other modifications to the architecture ?
    #
    #     for one_set in set_type:
    #
    #         print("Load dataset -> set: {}".format(one_set))
    #         dataset_args["which_set"] = one_set
    #         dataset = dataset_cstor(**dataset_args)
    #
    #         # hack dataset to only keep one game by image
    #         image_id_set = {}
    #         games = []
    #         for game in dataset.games:
    #             if game.image.id not in image_id_set:
    #                 games.append(game)
    #                 image_id_set[game.image.id] = 1
    #
    #         dataset.games = games
    #         no_images = len(games)
    #
    #         source_name = os.path.basename(img_input.name[:-2])
    #         dummy_tokenizer = DummyTokenizer()
    #         batchifier = batchifier_cstor(tokenizer=dummy_tokenizer, sources=[source_name])
    #         iterator = Iterator(dataset,
    #                             batch_size=batch_size,
    #                             pool=cpu_pool,
    #                             batchifier=batchifier)
    #
    #         ############################
    #         #  CREATE FEATURES
    #         ############################
    #         print("Start computing image features...")
    #         filepath = os.path.join(out_dir, "{}_features.h5".format(one_set))
    #         with h5py.File(filepath, 'w') as f:
    #
    #             ft_shape = [int(dim) for dim in ft_output.get_shape()[1:]]
    #             ft_dataset = f.create_dataset('features', shape=[no_images] + ft_shape, dtype=np.float32)
    #             idx2img = f.create_dataset('idx2img', shape=[no_images], dtype=np.int32)
    #             pt_hd5 = 0
    #
    #             for batch in tqdm(iterator):
    #                 feat = sess.run(ft_output, feed_dict={img_input: numpy.array(batch[source_name])})
    #
    #                 # Store dataset
    #                 batch_size = len(batch["raw"])
    #                 ft_dataset[pt_hd5: pt_hd5 + batch_size] = feat
    #
    #                 # Store idx to image.id
    #                 for i, game in enumerate(batch["raw"]):
    #                     idx2img[pt_hd5 + i] = game.image.id
    #
    #                 # update hd5 pointer
    #                 pt_hd5 += batch_size
    #             print("Start dumping file: {}".format(filepath))
    #         print("Finished dumping file: {}".format(filepath))


