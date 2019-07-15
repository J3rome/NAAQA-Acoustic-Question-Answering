import argparse
from collections import defaultdict
from _datetime import datetime
import subprocess
import shutil

import tensorflow as tf
from tensorboard.plugins.beholder import Beholder
from tqdm import tqdm
import numpy as np
import ujson

from utils import set_random_seed, create_folder_if_necessary, get_config, process_predictions, process_gamma_beta
from utils import create_symlink_to_latest_folder, save_training_stats, save_json
from utils import is_tensor_optimizer, is_tensor_prediction, is_tensor_scalar, is_tensor_beta_list, is_tensor_gamma_list, is_tensor_summary

from models.film_network_wrapper import FiLM_Network_Wrapper
from data_interfaces.CLEAR_dataset import CLEARDataset
from visualization import grad_cam_visualization
from preprocessing import preextract_features, create_dict_from_questions

parser = argparse.ArgumentParser('FiLM model for CLEAR Dataset (Acoustic Question Answering)', fromfile_prefix_chars='@')

parser.add_argument("--training", help="FiLM model training", action='store_true')
parser.add_argument("--inference", help="FiLM model inference", action='store_true')
parser.add_argument("--visualize", help="FiLM model visualization", action='store_true')
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
parser.add_argument("--dict_file_path", type=str, default=None, help="Define what dictionnary file should be used")

# TODO : Add option for custom test file

# Output parameters
parser.add_argument("-output_root_path", type=str, default='output', help="Directory with image")

# Other parameters
parser.add_argument("--nb_epoch", type=int, default=15, help="Nb of epoch for training")
parser.add_argument("--nb_epoch_stats_to_keep", type=int, default=5, help="Nb of epoch stats to keep for training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (For training and inference)")
parser.add_argument("--random_seed", type=int, default=None, help="Random seed used for the experiment")
parser.add_argument("--force_dict_all_answer", help="Will make sure that all answers are included in the dict" +
                                                    "(not just the one appearing in the train set)", action='store_true')


# TODO : Arguments Handling
#   Tasks :
#       Train from extracted features (No resnet - preprocessed)
#       Train from Raw Images (Resnet with fixed weights)
#       Train from Raw Images (Resnet or similar retrained (Only firsts couple layers))
#       Run from extracted features (No resnet - preprocessed)
#       Run from Raw Images (Resnet With fixed weights)
#       Run from Raw Images with VISUALIZATION
#

# >>> Training
def do_one_epoch(sess, batchifier, network_wrapper, outputs_var, epoch_index, writer=None, beholder=None):
    # check for optimizer to define training/eval mode
    is_training = any([is_tensor_optimizer(x) for x in outputs_var])

    aggregated_outputs = defaultdict(lambda : [])
    gamma_beta = {'gamma': [], 'beta': []}
    processed_predictions = []
    processed_gamma_beta = []

    summary_index = epoch_index * batchifier.batch_size

    for batch in tqdm(batchifier):

        feed_dict = network_wrapper.get_feed_dict(is_training, batch['question'], batch['answer'], batch['image'], batch['seq_length'])

        results = sess.run(outputs_var, feed_dict=feed_dict)

        if beholder is not None:
            print("Beholder Update")
            beholder.update(session=sess)

        for var, result in zip(outputs_var, results):
            if is_tensor_scalar(var):
                aggregated_outputs[var].append(result)

            elif is_tensor_prediction(var):
                aggregated_outputs[var] = True
                processed_predictions.append(process_predictions(network_wrapper.get_dataset(), result, batch['raw']))

            elif is_tensor_gamma_list(var):
                gamma_beta['gamma'].append(result)
            elif is_tensor_beta_list(var):
                gamma_beta['beta'].append(result)

            elif is_tensor_summary(var) and writer is not None:
                writer.add_summary(result, summary_index)
                summary_index += 1

        if writer is not None and summary_index > epoch_index * batchifier.batch_size:
            writer.flush()

    for var in aggregated_outputs.keys():
        if is_tensor_scalar(var):
            aggregated_outputs[var] = np.mean(aggregated_outputs[var]).item()
        elif is_tensor_prediction(var):
            aggregated_outputs[var] = [pred for epoch in processed_predictions for pred in epoch]

    for batch_index, gamma_vectors_per_batch, beta_vectors_per_batch in zip(range(len(gamma_beta['gamma'])), gamma_beta['gamma'], gamma_beta['beta']):
        processed_gamma_beta += process_gamma_beta(processed_predictions[batch_index], gamma_vectors_per_batch, beta_vectors_per_batch)

    to_return = list(aggregated_outputs.values())
    if len(processed_gamma_beta) > 0:
        # FIXME : Not keeping the order here. Not dependent on the operation order anymore
        to_return.append(processed_gamma_beta)

    return to_return


def do_film_training(sess, dataset, network_wrapper, optimizer_config, resnet_ckpt_path, nb_epoch, output_folder,
                     train_writer=None, val_writer=None, beholder=None, nb_epoch_to_keep=None):
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

    removed_epoch = []

    gamma_vector_tensors, beta_vector_tensors = network_wrapper.get_gamma_beta()

    op_to_run = [tf.summary.merge_all(), loss, accuracy, prediction, gamma_vector_tensors, beta_vector_tensors]

    # Training Loop
    for epoch in range(nb_epoch):
        epoch_output_folder_path = "%s/Epoch_%.2d" % (output_folder, epoch)
        create_folder_if_necessary(epoch_output_folder_path)

        print("Epoch %d" % epoch)
        time_before_epoch = datetime.now()
        train_loss, train_accuracy, train_predictions, train_gamma_beta_vectors = do_one_epoch(sess,
                                                                                        dataset.get_batches('train'),
                                                                                        network_wrapper,
                                                                                        op_to_run + [optimize_step],
                                                                                        epoch,
                                                                                        writer=train_writer,
                                                                                        beholder=beholder)

        epoch_train_time = datetime.now() - time_before_epoch

        print("Training :")
        print("    Loss : %f  - Accuracy : %f" % (train_loss, train_accuracy))

        # FIXME : Inference of validation set doesn't yield same result.
        # FIXME : Should val batches be shuffled ?
        # FIXME : Validation accuracy is skewed by the batch padding. We could recalculate accuracy from the prediction (After removing padded ones)
        val_loss, val_accuracy, val_predictions, val_gamma_beta_vectors = do_one_epoch(sess,
                                                                            dataset.get_batches('val', shuffled=False),
                                                                            network_wrapper, op_to_run,
                                                                            epoch, writer=None)#val_writer)

        print("Validation :")
        print("    Loss : %f  - Accuracy : %f" % (val_loss, val_accuracy))

        stats = save_training_stats(stats_file_path, epoch, train_accuracy, train_loss,
                            val_accuracy, val_loss, epoch_train_time)

        save_json(train_predictions, epoch_output_folder_path, filename="train_inferences.json")
        save_json(val_predictions, epoch_output_folder_path, filename="val_inferences.json")
        save_json(train_gamma_beta_vectors, epoch_output_folder_path, filename='train_gamma_beta.json')
        save_json(val_gamma_beta_vectors, epoch_output_folder_path, filename='val_gamma_beta.json')

        network_wrapper.save_film_checkpoint(sess, "%s/checkpoint.ckpt" % epoch_output_folder_path)

        if nb_epoch_to_keep is not None:
            # FIXME : Definitely not the most efficient way to do this
            sorted_stats = sorted(stats, key=lambda s: s['val_accuracy'], reverse=True)

            epoch_to_remove = sorted_stats[nb_epoch_to_keep:]

            for epoch_stat in epoch_to_remove:
                if epoch_stat['epoch'] not in removed_epoch:
                    removed_epoch.append(epoch_stat['epoch'])

                    shutil.rmtree("%s/%s" % (output_folder, epoch_stat['epoch']))

    # Create a symlink to best epoch output folder
    best_epoch = sorted(stats, key=lambda s: s['val_accuracy'], reverse=True)[0]['epoch']
    subprocess.run("cd %s && ln -s %s best" % (output_folder, best_epoch), shell=True)


# >>> Inference
def do_batch_inference(sess, dataset, network_wrapper, output_folder, film_ckpt_path, resnet_ckpt_path, set_name="test"):
    test_batches = dataset.get_batches(set_name, shuffled=False)

    sess.run(tf.variables_initializer(network_wrapper.film_variables))
    #sess.run(tf.global_variables_initializer())

    if dataset.is_raw_img():
        sess.run(tf.variables_initializer(network_wrapper.feature_extractor_variables))
        network_wrapper.restore_feature_extractor_weights(sess, resnet_ckpt_path)

    network_wrapper.restore_film_network_weights(sess, film_ckpt_path)

    network_predictions = network_wrapper.get_network_prediction()
    gamma_vector_tensors, beta_vector_tensors = network_wrapper.get_gamma_beta()

    processed_predictions = []

    processed_gamma_beta = []

    for batch in tqdm(test_batches):
        feed_dict = network_wrapper.get_feed_dict(False, batch['question'], batch['answer'],
                                                  batch['image'], batch['seq_length'])

        predictions, gamma_vectors, beta_vectors = sess.run([network_predictions,
                                                            gamma_vector_tensors,
                                                            beta_vector_tensors],
                                                            feed_dict=feed_dict)

        batch_processed_predictions = process_predictions(dataset, predictions, batch['raw'])
        processed_predictions += batch_processed_predictions

        processed_gamma_beta += process_gamma_beta(batch_processed_predictions, gamma_vectors, beta_vectors)

    # Batches are required to have always the same size.
    # We don't want the batch padding to interfere with the test accuracy.
    # We removed the padded (duplicated) examples
    if test_batches.nb_padded_in_last_batch > 0:
        processed_predictions = processed_predictions[:-test_batches.nb_padded_in_last_batch]
        processed_gamma_beta = processed_gamma_beta[:-test_batches.nb_padded_in_last_batch]

    nb_correct = sum(1 for r in processed_predictions if r['correct'])
    nb_results = len(processed_predictions)
    accuracy = nb_correct/nb_results

    save_json(processed_predictions, output_folder, filename='results.json')
    save_json(processed_gamma_beta, output_folder, filename='gamma_beta.json')

    print("Test set accuracy : %f" % accuracy)


def main(args):

    mutually_exclusive_params = [args.training, args.preprocessing, args.inference,
                                 args.feature_extract, args.create_dict, args.visualize]

    if 0 < sum(mutually_exclusive_params) > 1:
        print("[ERROR] Can only do one task at a time (--training, --inference, --visualize," +
              " --preprocessing, --create_dict, --feature_extract)")
        exit(1)

    is_preprocessing = False

    if args.training:
        task = "train_film"
    elif args.inference:
        task = "inference"
    elif args.visualize:
        task = "visualize_grad_cam"
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

    tensorboard_folder = "%s/tensorboard" % args.output_root_path

    train_writer_folder = '%s/train/%s' % (tensorboard_folder, current_datetime_str)
    val_writer_folder = '%s/val/%s' % (tensorboard_folder, current_datetime_str)
    test_writer_folder = '%s/test/%s' % (tensorboard_folder, current_datetime_str)
    beholder_folder = '%s/beholder' % tensorboard_folder

    if args.resnet_ckpt_path is None:
        args.resnet_ckpt_path = "%s/resnet/resnet_v1_101.ckpt" % args.data_root_path

    if args.film_ckpt_path is None:
        #experiment_date = "2019-06-23_16h37"
        experiment_date = "latest"
        args.film_ckpt_path = "%s/train_film/%s/%s/best/checkpoint.ckpt" % (args.output_root_path, args.version_name, experiment_date)

    if args.dict_file_path is None:
        args.dict_file_path = "%s/preprocessed/dict.json" % data_path

    film_model_config = get_config(args.config_path)

    create_output_folder = not is_preprocessing

    if create_output_folder:
        # TODO : See if this is optimal file structure
        create_folder_if_necessary(args.output_root_path)
        create_folder_if_necessary(output_task_folder)
        create_folder_if_necessary(output_experiment_folder)
        create_folder_if_necessary(output_dated_folder)
        create_symlink_to_latest_folder(output_experiment_folder, current_datetime_str)

        # Save arguments & config to output folder
        save_json(args, output_dated_folder, filename="arguments.json")
        save_json(film_model_config, output_dated_folder, filename='config_%s.json' % film_model_config['input']['type'])

        # Copy dictionary file used
        shutil.copyfile(args.dict_file_path, "%s/dict.json" % output_dated_folder)

    ########################################################
    ################### Data Loading #######################
    ########################################################
    dataset = CLEARDataset(data_path, film_model_config['input'], batch_size=args.batch_size,
                           tokenize_text=not is_preprocessing, dict_file_path=args.dict_file_path)

    ########################################################
    ################## Network Setup #######################
    ########################################################
    network_wrapper = FiLM_Network_Wrapper(film_model_config, dataset, preprocessing=is_preprocessing)

    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True)) as sess:
        # Debugging Tools
        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "T480s:8076")
        train_writer = tf.summary.FileWriter(train_writer_folder, sess.graph)  # FIXME : Make the path parametrable ?
        val_writer = tf.summary.FileWriter(val_writer_folder, sess.graph)  # FIXME : Make the path parametrable ?

        beholder = Beholder(beholder_folder)

        if task == "train_film":
            do_film_training(sess, dataset, network_wrapper, film_model_config['optimizer'],
                             args.resnet_ckpt_path, args.nb_epoch, output_dated_folder,
                             train_writer=train_writer, val_writer=val_writer, beholder=beholder,
                             nb_epoch_to_keep=args.nb_epoch_stats_to_keep)

        elif task == "inference":
            do_batch_inference(sess, dataset, network_wrapper, output_dated_folder,
                              args.film_ckpt_path, args.resnet_ckpt_path, set_name=args.inference_set)

        elif task == "visualize_grad_cam":
            grad_cam_visualization(sess, network_wrapper, args.film_ckpt_path, args.resnet_ckpt_path)

        elif task == "feature_extract":
            preextract_features(sess, dataset, network_wrapper, args.resnet_ckpt_path)

        elif task == "create_dict":
            create_dict_from_questions(dataset, force_all_answers=args.force_dict_all_answer)

        elif task == "full_preprocessing":
            create_dict_from_questions(dataset, force_all_answers=args.force_dict_all_answer)
            preextract_features(sess, dataset, network_wrapper, args.resnet_ckpt_path)

        time_elapsed = str(datetime.now() - current_datetime)

        print("Execution took %s" % time_elapsed)

        # TODO : Export Gamma & Beta
        # TODO : Export visualizations

    if create_output_folder:
        save_json({'time_elapsed': time_elapsed}, output_dated_folder, filename='timing.json')

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



