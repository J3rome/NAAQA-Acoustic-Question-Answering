import tensorflow as tf
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from _datetime import datetime
import json
import os
from collections import OrderedDict

from tensorflow.python import debug as tf_debug

from aqa.models.film_network import FiLM_Network
from aqa.data_interfaces.CLEAR_tokenizer import CLEARTokenizer
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

def do_one_epoch(sess, batchifier, outputs_var, network, image_var):
    # check for optimizer to define training/eval mode
    is_training = any([is_optimizer(x) for x in outputs_var])

    aggregated_outputs = defaultdict(lambda : [])

    for batch in tqdm(batchifier):

        feed_dict = network.get_feed_dict(image_var, is_training, batch['question'], batch['answer'], batch['image'], batch['seq_length'])

        results = sess.run(outputs_var, feed_dict=feed_dict)

        for var, result in zip(outputs_var, results):
            if is_scalar(var) and var in outputs_var:
                aggregated_outputs[var].append(result)

            elif is_list_int(var):
                # Inference mode (Answer tokens)
                aggregated_outputs[var] += result

    for var in aggregated_outputs.keys():
        if is_scalar(var):
            aggregated_outputs[var] = np.mean(aggregated_outputs[var])

    return list(aggregated_outputs.values())




if __name__ == "__main__":

    task = "train"

    # Parameters
    nb_epoch = 3
    nb_thread = 2
    batch_size = 32

    # Paths
    root_folder = "data"
    experiment_name = "v2.0.0_1k_scenes_1_inst_per_scene"
    experiment_path = "%s/%s" % (root_folder, experiment_name)
    resnet_ckpt_path = "%s/resnet/resnet_v1_101.ckpt" % root_folder
    resnet_chosen_layer = "block3/unit_22/bottleneck_v1"
    dict_path = "%s/preprocessed/dict.json" % experiment_path

    output_root_folder = "output"
    output_task_folder = "%s/%s" % (output_root_folder, task)
    now = datetime.now()
    output_experiment_folder = "%s/%s" %(output_task_folder, experiment_name)
    output_dated_folder = now.strftime("%Y-%m-%d_%H:%M")
    stats_file_path = "%s/stats.json" % output_dated_folder
    checkpoint_save_path = "%s/checkpoint.ckpt" % output_dated_folder


    film_model_config = {
        "image": {
            "type": "raw",
            "dim": [224, 224, 3],
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
        }
    }

    optimizer_config = {
        "learning_rate": 3e-4,
        "clip_val": 0.0,
        "weight_decay": 1e-5
    }

    ########################################################
    ################### Data Loading #######################
    ########################################################
    dataset = CLEARDataset(experiment_path, film_model_config['image'], batch_size=batch_size)

    ########################################################
    ################## Network Setup #######################
    ########################################################
    # TODO : There should be 2 Image placeholder : One for Raw Image, One for extracted features
    # Input Image
    images = tf.placeholder(tf.float32, [None, 224, 224, 3], name='clear/image')      # FIXME : would it be better to use a fixed batch_size instead of None ?

    # Feature extractor (Resnet 101)
    feature_extractor_chosen_layer = create_resnet(images, resnet_version=101,
                                                   chosen_layer=resnet_chosen_layer, is_training=False)

    # Adding the FiLM network after the chosen resnet layer
    network = FiLM_Network(film_model_config, input_image_tensor=feature_extractor_chosen_layer,
                           no_words=dataset.tokenizer.no_words, no_answers=dataset.tokenizer.no_answers, device=0)  # FIXME : Not sure that device 0 is appropriate for CPU

    # Setup optimizer (For training)
    optimize_step, [loss, accuracy] = create_optimizer(network, optimizer_config, var_list=None)  # TODO : Var_List should contain only film variables

    # GPU Options
    gpu_options = tf.GPUOptions(allow_growth=True)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "T480s:8076")

        tensorboard_writer = tf.summary.FileWriter('test_resnet_logs', sess.graph)

        sess.run(tf.global_variables_initializer())

        # Restore the pretrained weight of resnet
        resnet_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v1_101")
        resnet_saver = tf.train.Saver(var_list=resnet_variables)
        resnet_saver.restore(sess, resnet_ckpt_path)

        # Restore pretrained weight for FiLM network
        film_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="clear")    # TODO : Change scope
        film_saver = tf.train.Saver(max_to_keep=None, pad_step_number=3, var_list=film_variables)


        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        all_variables_name = [v.name for v in all_variables]
        resnet_variables_name = [v.name for v in resnet_variables]
        film_variables_name = [v.name for v in film_variables]

        # Training Loop
        for epoch in range(nb_epoch):
            print("Epoch %d" % epoch)
            train_loss, train_accuracy = do_one_epoch(sess, dataset.get_batches('train'),
                                                      [loss, accuracy, optimize_step], network, images)

            val_loss, val_accuracy = do_one_epoch(sess, dataset.get_batches('val'),
                                                      [loss, accuracy, optimize_step], network, images)

            save_training_stats(stats_file_path, epoch, train_accuracy, train_loss, val_accuracy,
                                val_loss)

            film_saver.save(sess, checkpoint_save_path, global_step=epoch)

            # TODO : Export Gamma & Beta
            # TODO : Export visualizations

            print("Training :")
            print("    Loss : %f  - Accuracy : %f" % (train_loss, train_accuracy))

        print("All Done")




        # TODO : Restore pretrained FiLM network
        # TODO : Train only FiLM part
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

