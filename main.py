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
from utils import create_symlink_to_latest_folder, save_training_stats, save_json, sort_stats, is_date_string
from utils import is_tensor_optimizer, is_tensor_prediction, is_tensor_scalar, is_tensor_beta_list, is_tensor_gamma_list, is_tensor_summary

from models.film_network_wrapper import FiLM_Network_Wrapper
from data_interfaces.CLEAR_dataset import CLEARDataset
from visualization import grad_cam_visualization
from preprocessing import create_dict_from_questions, extract_features

# NEW IMPORTS
from models.torch_film_model import CLEAR_FiLM_model
from data_interfaces.torch_dataset import CLEAR_dataset, ToTensor, ResizeImg  # FIXME : ToTensor should be imported from somewhere else. Utils ?
from models.torchsummary import summary     # Custom version of torchsummary to fix bugs with input
import torch
import time
import copy
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms

parser = argparse.ArgumentParser('FiLM model for CLEAR Dataset (Acoustic Question Answering)', fromfile_prefix_chars='@')

parser.add_argument("--training", help="FiLM model training", action='store_true')
parser.add_argument("--inference", help="FiLM model inference", action='store_true')
parser.add_argument("--visualize", help="FiLM model visualization", action='store_true')
parser.add_argument("--feature_extract", help="Feature Pre-Extraction", action='store_true')
parser.add_argument("--create_dict", help="Create word dictionary (for tokenization)", action='store_true')

# Input parameters
parser.add_argument("--data_root_path", type=str, default='data', help="Directory with data")
parser.add_argument("--version_name", type=str, help="Name of the dataset version")
parser.add_argument("--film_model_weight_path", type=str, default=None, help="Path to Film pretrained weight file")
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
parser.add_argument("--continue_training", help="Will use the --film_model_weight_path as  training starting point",
                    action='store_true')
parser.add_argument("--random_seed", type=int, default=None, help="Random seed used for the experiment")
parser.add_argument("--use_cpu", help="Model will be run/train on CPU", action='store_true')
parser.add_argument("--force_dict_all_answer", help="Will make sure that all answers are included in the dict" +
                                                    "(not just the one appearing in the train set)" +
                                                    " -- Preprocessing option" , action='store_true')


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
def do_one_epoch_TF(sess, batchifier, network_wrapper, outputs_var, epoch_index, writer=None, beholder=None):
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


def do_film_training_TF(sess, dataset, network_wrapper, optimizer_config, resnet_ckpt_path, nb_epoch, output_folder,
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
        train_loss, train_accuracy, train_predictions, train_gamma_beta_vectors = do_one_epoch_TF(sess,
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
        val_loss, val_accuracy, val_predictions, val_gamma_beta_vectors = do_one_epoch_TF(sess,
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
def do_batch_inference_TF(sess, dataset, network_wrapper, output_folder, film_ckpt_path, resnet_ckpt_path, set_name="test"):
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


# TODO : Interactive mode
def run_one_game(device, model, games, data_path, input_config, transforms_list=[]):

    # TODO : Should be able to run from a specific image (Not necessarly inside data/experiment_name/images/set_type)

    set_type = 'test'
    game = {
            "question_index" : 1,
            "question": "How many loud things can we hear ?",
            "answer": 3,
            "scene_index": 3
        }

    game['scene_filename'] = "CLEAR_%s_%06d.png" % (set_type, game['scene_index'])

    games = [
        game
    ]

    test_dataset = CLEAR_dataset(data_path, input_config, set_type, questions=games,
                                 dict_file_path=args.dict_file_path,
                                 transforms=transforms.Compose(transforms_list + [ToTensor()]))

    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                  num_workers=1, collate_fn=test_dataset.CLEAR_collate_fct)

    _, accuracy, predictions, gamma_beta = process_dataloader(False, device, model, test_dataloader)

    print("Accuracy is : %f" % accuracy)


def process_dataloader(is_training, device, model, dataloader, criterion=None, optimizer=None):
    # Model should already be copied to GPU at this point
    #assert (is_training and criterion and optimizer)

    if is_training:
        model.train()       #FIXME :Verify, is there a cost to calling train multiple time ? Is there a way to check if already set ?
    else:
        model.eval()

    dataset_size = len(dataloader.dataset)
    running_loss = 0.0
    running_corrects = 0

    processed_predictions = []
    processed_gammas_betas = []

    for batch in tqdm(dataloader):
        # mem_trace.report('Batch %d' % i)
        images = batch['image'].to(device)  # .type(torch.cuda.FloatTensor)
        questions = batch['question'].to(device)  # .type(torch.cuda.LongTensor)
        answers = batch['answer'].to(device)  # .type(torch.cuda.LongTensor)
        seq_lengths = batch['seq_length'].to(device)

        # Those are not processed by the network, only used to create statistics. Therefore, no need to copy to GPU
        questions_id = batch['id']
        scenes_id = batch['scene_id']

        if is_training:
            # zero the parameter gradients
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):  # FIXME : Do we need to set this to false when evaluating validation ?
            outputs = model(questions, seq_lengths, images)
            _, preds = torch.max(outputs, 1)
            if criterion:
                loss = criterion(outputs, answers)

            if is_training:
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

        batch_processed_predictions = process_predictions(dataloader.dataset, preds.tolist(), answers.tolist(),
                                                          questions_id.tolist(), scenes_id.tolist())

        processed_predictions += batch_processed_predictions

        gammas, betas = model.get_gammas_betas()
        processed_gammas_betas += process_gamma_beta(batch_processed_predictions, gammas, betas)

        # statistics
        if criterion:
            running_loss += loss.item() * dataloader.batch_size
        running_corrects += torch.sum(preds == answers.data).item()

    # Todo : accumulate preds & create processed result

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    return epoch_loss, epoch_acc, processed_predictions, processed_gammas_betas


def set_inference(device, model, dataloader, output_folder):
    set_type = dataloader.dataset.set

    # TODO : load pretrained weights

    _, acc, predictions, gammas_betas = process_dataloader(False, device, model, dataloader)

    save_json(predictions, output_folder, filename='%s_predictions.json' % set_type)
    save_json(gammas_betas, output_folder, filename='%s_predictions.json' % set_type)

    print("%s Accuracy : %.5f" % (set_type, acc))


def train_model(device, model, dataloaders, output_folder, criterion=None, optimizer=None, scheduler=None,
                nb_epoch=25, nb_epoch_to_keep=None, start_epoch=0):

    stats_file_path = "%s/stats.json" % output_folder
    removed_epoch = []

    since = time.time()

    for epoch in range(start_epoch, start_epoch + nb_epoch):
        epoch_output_folder_path = "%s/Epoch_%.2d" % (output_folder, epoch)
        create_folder_if_necessary(epoch_output_folder_path)
        print('Epoch {}/{}'.format(epoch, start_epoch + nb_epoch - 1))
        print('-' * 10)

        time_before_train = datetime.now()
        train_loss, train_acc, train_predictions, train_gammas_betas = process_dataloader(True, device, model,
                                                                                          dataloaders['train'],
                                                                                          criterion, optimizer)
        epoch_train_time = datetime.now() - time_before_train

        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Train', train_loss, train_acc))

        val_loss, val_acc, val_predictions, val_gammas_betas = process_dataloader(False, device, model,
                                                                                  dataloaders['val'], criterion,
                                                                                  optimizer)
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Val', val_loss, val_acc))

        # TODO : Resnet Preprocessing
        # TODO : Visualize gradcam
        # TODO : T-SNE plots

        stats = save_training_stats(stats_file_path, epoch, train_acc, train_loss, val_acc, val_loss, epoch_train_time)

        save_json(train_predictions, epoch_output_folder_path, filename="train_predictions.json")
        save_json(val_predictions, epoch_output_folder_path, filename="val_predictions.json")
        save_json(train_gammas_betas, epoch_output_folder_path, filename="train_gamma_beta.json")
        save_json(val_gammas_betas, epoch_output_folder_path, filename="val_gamma_beta.json")

        print("Training took %s" % str(epoch_train_time))

        # Save training weights
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.get_cleaned_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss
        }, '%s/model.pt.tar' % epoch_output_folder_path)

        if nb_epoch_to_keep is not None:
            # FIXME : Definitely not the most efficient way to do this
            sorted_stats = sort_stats(stats, reverse=True)

            epoch_to_remove = sorted_stats[nb_epoch_to_keep:]

            for epoch_stat in epoch_to_remove:
                if epoch_stat['epoch'] not in removed_epoch:
                    removed_epoch.append(epoch_stat['epoch'])

                    shutil.rmtree("%s/%s" % (output_folder, epoch_stat['epoch']))
        print()

    # FIXME : Should probably keep the symlink updated at each epoch in case we shutdown the process midway
    # Create a symlink to best epoch output folder
    if nb_epoch_to_keep is None:
        sorted_stats = sort_stats(stats, reverse=True)

    best_epoch = sorted_stats[0]
    subprocess.run("cd %s && ln -s %s best" % (output_folder, best_epoch['epoch']), shell=True)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {}'.format(best_epoch['val_acc']))

    # TODO : load best model weights ?
    #model.load_state_dict(best_model_state)
    return model


def main(args):

    mutually_exclusive_params = [args.training, args.inference,
                                 args.feature_extract, args.create_dict, args.visualize]

    if 0 < sum(mutually_exclusive_params) > 1:
        print("[ERROR] Can only do one task at a time (--training, --inference, --visualize," +
              " --create_dict, --feature_extract)")
        exit(1)

    if args.training:
        task = "train_film"
    elif args.inference:
        task = "inference"
    elif args.visualize:
        task = "visualize_grad_cam"
    elif args.feature_extract:
        task = "feature_extract"
    elif args.create_dict:
        task = "create_dict"

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

    # If path specified is a date, we construct the path to the best model weights for the specified run
    if args.film_model_weight_path is not None:

        base_path = "%s/train_film/%s/%s" % (args.output_root_path, args.version_name, args.film_model_weight_path)
        suffix = "best/model.pt.tar"        # FIXME : We might redo some epoch when continuing training because the 'best' epoch is not necessarely the last

        if is_date_string(args.film_model_weight_path):
            args.film_model_weight_path = "%s/%s" % (base_path, suffix)
        elif args.film_model_weight_path == 'latest':
            # The 'latest' symlink will be overriden by this run (If continuing training).
            # Use real path of latest experiment
            symlink_value = os.readlink(base_path)
            clean_base_path = base_path[:-(len(args.film_model_weight_path) + 1)]
            args.film_model_weight_path = '%s/%s/%s' % (clean_base_path, symlink_value, suffix)

    restore_model_weights = args.inference or (args.training and args.continue_training)

    if args.dict_file_path is None:
        args.dict_file_path = "%s/preprocessed/dict.json" % data_path

    film_model_config = get_config(args.config_path)

    create_output_folder = not args.create_dict and not args.feature_extract

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

    device = 'cuda:0' if torch.cuda.is_available() and not args.use_cpu else 'cpu'

    ####################################
    #   Dataloading
    ####################################

    transforms_list = []
    # FIXME : When preprocessing features, we should force input type to raw
    if film_model_config['input']['type'] == 'raw':
        feature_extractor_config = {'version': 101, 'layer_index': 6}   # Idx 6 -> Block3/unit22
        transforms_list = [ResizeImg((224, 224))]     # TODO : Take size as parameter ?
    else:
        feature_extractor_config = None

    transforms_to_apply = transforms.Compose(transforms_list + [ToTensor()])

    print("Creating Datasets")
    dict_file_path = None if not args.create_dict else args.dict_file_path

    train_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'train', dict_file_path=dict_file_path,
                                  transforms=transforms_to_apply, tokenize_text=not args.create_dict)

    val_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'val', dict_file_path=dict_file_path,
                                transforms=transforms_to_apply, tokenize_text=not args.create_dict)

    test_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'test', dict_file_path=dict_file_path,
                                 transforms=transforms_to_apply, tokenize_text=not args.create_dict)

    #trickytest_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'trickytest',
    #                             dict_file_path=args.dict_file_path,
    #                             transforms=transforms.Compose(transforms_list + [ToTensor()]))

    print("Creating Dataloaders")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=train_dataset.CLEAR_collate_fct)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=train_dataset.CLEAR_collate_fct)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, collate_fn=train_dataset.CLEAR_collate_fct)

    #trickytest_dataloader = DataLoader(trickytest_dataset, batch_size=args.batch_size, shuffle=False,
    #                             num_workers=4, collate_fn=train_dataset.CLEAR_collate_fct)


    ####################################
    #   Model Definition
    ####################################

    if not args.create_dict:
        print("Creating model")
        # Retrieve informations to instantiate model
        nb_words, nb_answers = train_dataset.get_token_counts()
        input_image_torch_shape = train_dataset.get_input_shape(channel_first=True)  # Torch size have Channel as first dimension
        padding_token = train_dataset.get_padding_token()

        film_model = CLEAR_FiLM_model(film_model_config, input_image_channels=input_image_torch_shape[0],
                                      nb_words=nb_words, nb_answers=nb_answers,
                                      sequence_padding_idx=padding_token,
                                      feature_extraction_config=feature_extractor_config)

        if restore_model_weights:
            assert args.film_model_weight_path is not None, 'Must provide path to model weights to ' \
                                                            'do inference or to continue training.'

            checkpoint = torch.load(args.film_model_weight_path, map_location=device)
            # We need non-strict because feature extractor weight are not included in the saved state dict
            film_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if device != 'cpu':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = True

        film_model.to(device)

        print("Model ready to run")

        summary(film_model, [(22,), (1,), input_image_torch_shape], device=device)

    if task == "train_film":
        trainable_parameters = filter(lambda p: p.requires_grad, film_model.parameters())

        # FIXME : Not sure what is the current behavious when specifying weight decay to Adam Optimizer. INVESTIGATE THIS
        optimizer = torch.optim.Adam(trainable_parameters, lr=film_model_config['optimizer']['learning_rate'],
                                     weight_decay=film_model_config['optimizer']['weight_decay'])

        start_epoch = 0
        if args.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1

        # scheduler = torch.optim.lr_scheduler   # FIXME : Using a scheduler give the ability to decay only each N epoch.

        train_model(device=device, model=film_model, dataloaders={'train': train_dataloader, 'val': val_dataloader},
                    output_folder=output_dated_folder, criterion=nn.CrossEntropyLoss(), optimizer=optimizer,
                    nb_epoch=args.nb_epoch, nb_epoch_to_keep=args.nb_epoch_stats_to_keep,
                    start_epoch=start_epoch)

    elif task == "inference":
        set_inference(device=device, model=film_model, dataloader=test_dataloader, output_folder=output_dated_folder)

    elif task == "create_dict":
        create_dict_from_questions(train_dataset, force_all_answers=args.force_dict_all_answer)

    elif task == "feature_extract":
        extract_features(device=device, feature_extractor=film_model.feature_extractor,
                         dataloaders={'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader})

    elif task == "visualize_grad_cam":
        grad_cam_visualization(sess, network_wrapper, args.film_ckpt_path, args.resnet_ckpt_path)

    time_elapsed = str(datetime.now() - current_datetime)

    print("Execution took %s" % time_elapsed)

    if create_output_folder:
        save_json({'time_elapsed': time_elapsed}, output_dated_folder, filename='timing.json')

        print("All Done")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

        # TODO : Extract Beta And Gamma Parameters + T-SNE
        # TODO : Feed RAW image directly to the FiLM network
        # TODO : Resize images ? Do we absolutely need 224x224 for the resnet preprocessing ?
        #        Since we extract a layer in resnet, we can feed any size, its fully convolutional up to that point
        #        When feeding directly to FiLM, we can use original size ?
        # TODO : What is the optimal size for our spectrograms ?
        # TODO : Train with different amount of residual blocks. Other modifications to the architecture ?



