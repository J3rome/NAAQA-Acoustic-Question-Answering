import argparse
from _datetime import datetime
import subprocess
import shutil
import os
import random

from tqdm import tqdm

from utils import set_random_seed, create_folder_if_necessary, get_config, process_predictions, process_gamma_beta
from utils import create_symlink_to_latest_folder, save_training_stats, save_json, sort_stats, is_date_string
from utils import save_gamma_beta_h5, save_git_revision, get_random_state, set_random_state, close_tensorboard_writers
from utils import calc_mean_and_std, update_mean_in_config, calc_f1_score

from visualization import visualize_gamma_beta, grad_cam_visualization
from preprocessing import create_dict_from_questions, extract_features, images_to_h5

# NEW IMPORTS
from models.CLEAR_film_model import CLEAR_FiLM_model
from data_interfaces.CLEAR_dataset import CLEAR_dataset, CLEAR_collate_fct
from data_interfaces.transforms import ToTensor, ImgBetweenZeroOne, ResizeImgBasedOnHeight, ResizeImgBasedOnWidth, PadTensor, NormalizeSample, ResizeTensor
from models.torchsummary import summary     # Custom version of torchsummary to fix bugs with input
import torch
import time
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR
import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from models.lr_finder import LRFinder
import matplotlib.pyplot as plt


# TODO : Add option for custom test file --> Already available by specifying different inference_set ? The according dataset & dataloader should be created..
#       Maybe not a good idea to instantiate everything out of the "task" functions.. Or maybe we could just instantiate it inside for the test inference
parser = argparse.ArgumentParser('FiLM model for CLEAR Dataset (Acoustic Question Answering)', fromfile_prefix_chars='@')

parser.add_argument("--training", help="FiLM model training", action='store_true')
parser.add_argument("--inference", help="FiLM model inference", action='store_true')
parser.add_argument("--visualize_grad_cam", help="Class Activation Maps - GradCAM", action='store_true')
parser.add_argument("--visualize_gamma_beta", help="FiLM model parameters visualization (T-SNE)", action='store_true')
parser.add_argument("--prepare_images", help="Save images in h5 file for faster retrieving", action='store_true')
parser.add_argument("--feature_extract", help="Feature Pre-Extraction", action='store_true')
parser.add_argument("--create_dict", help="Create word dictionary (for tokenization)", action='store_true')
parser.add_argument("--random_answer_baseline", help="Spit out a random answer for each question", action='store_true')
parser.add_argument("--random_weight_baseline", help="Use randomly initialised Neural Network to answer the question",
                    action='store_true')
parser.add_argument("--lr_finder", help="Create LR Finder plot", action='store_true')
parser.add_argument("--write_clear_mean_to_config", help="Will calculate the mean and std of the dataset and write it "
                                                         "to the config file", action='store_true')

# Input parameters
parser.add_argument("--data_root_path", type=str, default='data', help="Directory with data")
parser.add_argument("--version_name", type=str, help="Name of the dataset version")
parser.add_argument("--film_model_weight_path", type=str, default=None, help="Path to Film pretrained weight file")
parser.add_argument("--config_path", type=str, default='config/film.json', help="Path to Film pretrained ckpt file")
parser.add_argument("--inference_set", type=str, default='test', help="Define on which set the inference will run")
parser.add_argument("--dict_file_path", type=str, default=None, help="Define what dictionnary file should be used")
parser.add_argument("--normalize_with_imagenet_stats", help="Will normalize input images according to"
                                                       "ImageNet mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--normalize_with_clear_stats", help="Will normalize input images according to"
                                                         "CLEAR mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--no_img_resize", help="Disable RAW image resizing", action='store_true')
parser.add_argument("--raw_img_resize_val", type=int, default=224,
                    help="Specify the size to which the image will be resized (when working with RAW img)"
                         "The width is calculated according to the height in order to keep the ratio")
parser.add_argument("--raw_img_resize_based_on_height", action='store_true',
                    help="If set (with --raw_img_resize_val), the width of the image will be calculated according to "
                         "the height in order to keep the ratio. [Default option if neither "
                         "--raw_img_resize_based_on_height and --raw_img_resize_based_on_width are set]")
parser.add_argument("--raw_img_resize_based_on_width", action='store_true',
                    help="If set (with --raw_img_resize_val), the height of the image will be calculated according to "
                         "the width in order to keep the ratio")


parser.add_argument("--keep_image_range", help="Will NOT scale the image between 0-1 (RAW img)", action='store_true')
parser.add_argument("--pad_to_largest_image", help="If set, images will be padded to meet the largest image in the set."
                                                   "All input will have the same size.", action='store_true')
parser.add_argument("--pad_to_square_images", help="If set, all images will be padded to make them square",
                    action='store_true')
parser.add_argument("--resize_to_square_images", help="If set, all images will be resized to make them square",
                    action='store_true')
parser.add_argument("--gamma_beta_path", type=str, default=None, help="Path where gamma_beta values are stored "
                                                                          "(when using --visualize_gamma_beta)")
parser.add_argument("--no_early_stopping", help="Override the early stopping config", action='store_true')
parser.add_argument("--feature_extractor_layer_index", type=int, default=6, help="Layer id of the pretrained Resnet")
parser.add_argument("--no_feature_extractor", help="Raw images won't go through Resnet feature extractor before "
                                                    "training", action='store_true')


# Output parameters
parser.add_argument("--output_root_path", type=str, default='output', help="Directory with image")
parser.add_argument("--preprocessed_folder_name", type=str, default='preprocessed',
                    help="Directory where to store/are stored extracted features and token dictionary")
parser.add_argument("--output_name_suffix", type=str, default='', help="Suffix that will be appended to the version "
                                                                       "name (output & tensorboard)")
parser.add_argument("--no_start_end_tokens", help="Constants tokens won't be added to the question "
                                                  "(<start> & <end> tokens)", action='store_true')
parser.add_argument("--dict_folder", type=str, default=None,
                    help="Directory where to store/retrieve generated dictionary. "
                         "If --dict_file_path is used, this will be ignored")
parser.add_argument("--tensorboard_folder", type=str, default='tensorboard',
                    help="Path where tensorboard data should be stored.")
parser.add_argument("--tensorboard_save_graph", help="Save model graph to tensorboard", action='store_true')
parser.add_argument("--tensorboard_save_images", help="Save input images to tensorboard", action='store_true')
parser.add_argument("--tensorboard_save_texts", help="Save input texts to tensorboard", action='store_true')
parser.add_argument("--gpu_index", type=str, default='0', help="Index of the GPU to use")


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
parser.add_argument("--no_model_summary", help="Will hide the model summary", action='store_true')
parser.add_argument("--perf_over_determinist", help="Will let torch use nondeterministic algorithms (Better "
                                                    "performance but less reproductibility)", action='store_true')
parser.add_argument("--overwrite_clear_mean", help="Will overwrite the Mean and Std of the CLEAR dataset stored in "
                                                   "config file", action='store_true')
parser.add_argument("--f1_score", help="Use f1 score in loss calculation", action='store_true')
parser.add_argument("--cyclical_lr", help="Will use cyclical learning rate (Bounds in config.json)",
                    action='store_true')


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


def set_inference(device, model, dataloader, criterion, output_folder, save_gamma_beta=True):
    set_type = dataloader.dataset.set

    if save_gamma_beta:
        gamma_beta_path = '%s/%s_gamma_beta.h5' % (output_folder, set_type)
    else:
        gamma_beta_path = None

    _, acc, predictions = process_dataloader(False, device, model, dataloader, criterion=criterion, gamma_beta_path=None)#gamma_beta_path)

    save_json(predictions, output_folder, filename='%s_predictions.json' % set_type)

    print("%s Accuracy : %.5f" % (set_type, acc))


def train_model(device, model, dataloaders, output_folder, criterion=None, optimizer=None, scheduler=None,
                nb_epoch=25, nb_epoch_to_keep=None, start_epoch=0, tensorboard=None):

    if tensorboard is None:
        tensorboard = {'writers': {'train': None, 'val': None}, 'options': None}
    else:
        assert 'train' in tensorboard['writers'] and 'val' in tensorboard['writers'], 'Must provide all tensorboard writers.'

    tensorboard_per_set = {'writer': None, 'options': tensorboard['options']}

    stats_file_path = "%s/stats.json" % output_folder
    removed_epoch = []

    since = time.time()

    # Early stopping (Only enable when we are running at least 20 epoch)
    early_stopping = model.early_stopping is not None and nb_epoch > 20
    if early_stopping:
        if type(model.early_stopping['wait_first_n_epoch']) == float:
            wait_first_n_epoch = int(nb_epoch * model.early_stopping['wait_first_n_epoch'])
        else:
            wait_first_n_epoch = model.early_stopping['wait_first_n_epoch']

        if type(model.early_stopping['stop_threshold']) == float:
            stop_threshold = int(nb_epoch*model.early_stopping['stop_threshold'])
        else:
            stop_threshold = model.early_stopping['stop_threshold']

        wait_first_n_epoch += start_epoch           # Apply grace period even when continuing training
        stop_threshold = max(stop_threshold, 1)
        best_val_loss = 9999
        early_stop_counter = 0

    # TODO : Write hyperparams to tensorboard

    for epoch in range(start_epoch, start_epoch + nb_epoch):
        epoch_output_folder_path = "%s/Epoch_%.2d" % (output_folder, epoch)
        create_folder_if_necessary(epoch_output_folder_path)
        print('Epoch {}/{}'.format(epoch, start_epoch + nb_epoch - 1))
        print('-' * 10)

        epoch_time = datetime.now()
        tensorboard_per_set['writer'] = tensorboard['writers']['train']
        train_loss, train_acc, train_predictions = process_dataloader(True, device, model,
                                                                      dataloaders['train'],
                                                                      criterion, optimizer, scheduler=scheduler,
                                                                      epoch_id=epoch, tensorboard=tensorboard_per_set,
                                                                      gamma_beta_path="%s/train_gamma_beta.h5" % epoch_output_folder_path)
        epoch_train_time = datetime.now() - epoch_time

        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Train', train_loss, train_acc))

        tensorboard_per_set['writer'] = tensorboard['writers']['val']
        val_loss, val_acc, val_predictions = process_dataloader(False, device, model,
                                                                dataloaders['val'], criterion,
                                                                epoch_id=epoch, tensorboard=tensorboard_per_set,
                                                                gamma_beta_path="%s/val_gamma_beta.h5" % epoch_output_folder_path)
        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format('Val', val_loss, val_acc))

        stats = save_training_stats(stats_file_path, epoch, train_acc, train_loss, val_acc, val_loss, epoch_train_time)

        save_json(train_predictions, epoch_output_folder_path, filename="train_predictions.json")
        save_json(val_predictions, epoch_output_folder_path, filename="val_predictions.json")

        # Save training weights
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.get_cleaned_state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'rng_state': get_random_state(),
        }

        torch.save(checkpoint, '%s/model.pt.tar' % epoch_output_folder_path)

        sorted_stats = sort_stats(stats)

        if nb_epoch_to_keep is not None:
            # FIXME : Probably not the most efficient way to do this
            epoch_to_remove = sorted_stats[nb_epoch_to_keep:]

            for epoch_stat in epoch_to_remove:
                if epoch_stat['epoch'] not in removed_epoch:
                    removed_epoch.append(epoch_stat['epoch'])

                    shutil.rmtree("%s/%s" % (output_folder, epoch_stat['epoch']))

        print("Epoch took %s" % str(datetime.now() - epoch_time))

        # Create a symlink to best epoch output folder
        best_epoch = sorted_stats[0]
        print("Best Epoch is {} with Loss: {} Acc: {}".format(best_epoch['epoch'],
                                                              best_epoch['val_loss'],
                                                              best_epoch['val_acc']))
        best_epoch_symlink_path = '%s/best' % output_folder
        subprocess.run("ln -snf %s %s" % (best_epoch['epoch'], best_epoch_symlink_path), shell=True)

        # Early Stopping
        if early_stopping:
            if val_loss < best_val_loss - model.early_stopping['min_step']:
                best_val_loss = val_loss
                early_stop_counter = 0
            elif epoch > wait_first_n_epoch:
                early_stop_counter += 1
                print("Early Stopping count : %d/%d" % (early_stop_counter, stop_threshold))

                if early_stop_counter >= stop_threshold:
                    print("Early Stopping at epoch %d on %d" % (epoch, nb_epoch))
                    break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {}'.format(best_epoch['val_acc']))

    # TODO : load best model weights ?
    #model.load_state_dict(best_model_state)
    return model


def get_lr_finder_curves(model, device, train_dataloader, output_dated_folder, num_iter, optimizer, val_dataloader=None,
                         loss_criterion=nn.CrossEntropyLoss(), weight_decay_list=None, min_lr=1e-10):
    if type(weight_decay_list) != list:
        weight_decay_list = [0., 3e-1, 3e-2, 3e-3, 3e-4, 3e-5, 3e-6, 3e-7, 3e-8, 3e-9]

    # TODO : The order of the data probably affect the LR curves. Shuffling and doing multiple time should help

    # FIXME : What momentum value should we use for SGD ???
    # Force momentum to 0
    initial_optimizer_state_dict = optimizer.state_dict()
    zero_momentum_state_dict = optimizer.state_dict()
    zero_momentum_state_dict['param_groups'][0]['momentum'] = 0
    optimizer.load_state_dict(zero_momentum_state_dict)
    # FIXME : Should force SGD when running lr_finder ?

    fig, ax = plt.subplots()
    lr_finder = LRFinder(model, optimizer, loss_criterion, device=device)

    # There is probably a better way to set weight decay and learning rate that modifying state_dict and reloading
    lr_finder.reset(weight_decay=weight_decay_list[0], learning_rate=min_lr)

    for weight_decay in weight_decay_list:
        lr_finder.reset(weight_decay=weight_decay)
        print(f"Learning Rate finder -- Running for {num_iter} batches with weight decay : {weight_decay:.5}")
        # FIXME : Should probably run with validation data?
        lr_finder.range_test(train_dataloader, val_loader=None, end_lr=100, num_iter=num_iter,
                             num_iter_val=100)

        fig, ax = lr_finder.plot(fig_ax=(fig, ax), legend_label=f"Weight Decay : {weight_decay:.5}", show_fig=False)

    filepath = "%s/%s" % (output_dated_folder, 'lr_finder_plot.png')
    fig.savefig(filepath)

    # Reset optimiser config
    optimizer.load_state_dict(initial_optimizer_state_dict)


def write_clear_mean_to_config(dataloader, device, current_config, config_file_path, overwrite_mean=False):
    assert os.path.isfile(config_file_path), f"Config file '{config_file_path}' doesn't exist"

    key = "clear_stats"

    if not overwrite_mean and 'preprocessing' in current_config and key in current_config['preprocessing'] \
       and type(current_config['preprocessing'][key]) == list:
        assert False, "CLEAR mean is already present in config."

    dataloader.dataset.keep_1_game_per_scene()

    mean, std = calc_mean_and_std(dataloader, device=device)

    update_mean_in_config(mean, std, config_file_path, current_config=current_config, key=key)


def process_dataloader(is_training, device, model, dataloader, criterion=None, optimizer=None, scheduler=None, gamma_beta_path=None,
                       write_to_file_every=500, epoch_id=0, tensorboard=None):
    # Model should already have been copied to the GPU at this point (If using GPU)
    assert (is_training and criterion is not None and optimizer is not None) or not is_training

    if is_training:
        model.train()
    else:
        model.eval()

    dataset_size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    running_loss = 0.0
    running_corrects = 0

    processed_predictions = []
    processed_gammas_betas = []
    all_questions = []
    nb_written = 0

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        #mem_trace.report('Batch %d/%d - Epoch %d' % (i, dataloader.batch_size, epoch))
        images = batch['image'].to(device)
        questions = batch['question'].to(device)
        answers = batch['answer'].to(device)
        seq_lengths = batch['seq_length'].to(device)

        # Those are not processed by the network, only used to create statistics. Therefore, no need to copy to GPU
        questions_id = batch['id']
        scenes_id = batch['scene_id']
        images_padding = batch['image_padding']

        if is_training:
            # zero the parameter gradients
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs, outputs_softmax = model(questions, seq_lengths, images, pack_sequence=True)
            _, preds = torch.max(outputs, 1)
            if criterion:
                loss = criterion(outputs, answers)

            if is_training:
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

                if scheduler:
                    scheduler.step()

        batch_processed_predictions = process_predictions(dataloader.dataset, preds.tolist(), answers.tolist(),
                                                          questions_id.tolist(), scenes_id.tolist(),
                                                          outputs_softmax.tolist(), images_padding.tolist())

        processed_predictions += batch_processed_predictions

        # TODO : Add config to log only specific things
        if tensorboard and tensorboard['writer'] and tensorboard['options'] and tensorboard['options']['save_images']:
            # FIXME: Find a way to show original input images in tensorboard (Could save a list of scene ids and add them to tensorboard after the epoch, check performance cost -- Image loading etc)
            if dataloader.dataset.is_raw_img():
                # TODO : Tag img before adding to tensorboard ? -- This can be done via .add_image_with_boxes()
                for image in batch['image']:
                    tensorboard['writer'].add_image('Inputs/images', image, epoch_id)

            all_questions += batch['question'].tolist()

        if gamma_beta_path is not None:
            gammas, betas = model.get_gammas_betas()
            processed_gammas_betas += process_gamma_beta(batch_processed_predictions, gammas, betas)

            if batch_idx % write_to_file_every == 0 and batch_idx != 0:
                nb_written += save_gamma_beta_h5(processed_gammas_betas, dataloader.dataset.set, gamma_beta_path,
                                                 nb_vals=dataset_size, start_idx=nb_written)
                processed_gammas_betas = []

        # statistics
        if criterion:
            running_loss += loss.item() * dataloader.batch_size
        running_corrects += torch.sum(preds == answers.data).item()

    nb_left_to_write = len(processed_gammas_betas)
    if gamma_beta_path is not None and nb_left_to_write > 0:
        save_gamma_beta_h5(processed_gammas_betas, dataloader.dataset.set, gamma_beta_path, nb_vals=dataset_size,
                           start_idx=nb_written)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size

    # TODO : Add config to log only specific things
    if tensorboard and tensorboard['writer']:
        if tensorboard['options'] and tensorboard['options']['save_texts']:
            log_text = ""
            for question, processed_prediction in zip(all_questions, processed_predictions):
                # FIXME: Tokenizer might not be instantiated --- We probably wouldn't be logging in tensorboard..
                decoded_question = dataloader.dataset.tokenizer.decode_question(question, remove_padding=True)
                log_text += f"{processed_prediction['correct']}//{processed_prediction['correct_answer_family']} "
                log_text += f"{decoded_question} -- {processed_prediction['ground_truth']} "
                log_text += f"[[{processed_prediction['prediction']} - {processed_prediction['confidence']}]]  \n"

            tensorboard['writer'].add_text('Inputs/Text', log_text, epoch_id)

        tensorboard['writer'].add_scalar('Results/Loss', epoch_loss, global_step=epoch_id)
        tensorboard['writer'].add_scalar('Results/Accuracy', epoch_acc, global_step=epoch_id)

    return epoch_loss, epoch_acc, processed_predictions


def random_answer_baseline(dataloader, output_folder=None):
    answers = list(dataloader.dataset.answer_counter.keys())
    set_type = dataloader.dataset.set
    correct_answer = 0
    incorrect_answer = 0
    for batch in tqdm(dataloader):
        ground_truths = batch['answer'].tolist()
        for ground_truth in ground_truths:
            random_answer = random.choice(answers)
            if random_answer == ground_truth:
                correct_answer += 1
            else:
                incorrect_answer += 1

    accuracy = correct_answer / (correct_answer + incorrect_answer)

    print(f"Random answer baseline accuracy for set {set_type} : {accuracy}")

    if output_folder:
        save_json({'accuracy': accuracy}, output_folder, f'{set_type}_results.json')


def random_weight_baseline(model, device, dataloader, output_folder=None):

    set_type = dataloader.dataset.set
    loss, accuracy, _ = process_dataloader(False, device, model, dataloader)

    print(f"Random weight baseline for set {set_type}. Accuracy : {accuracy}  Loss : {loss}")

    if output_folder:
        save_json({'accuracy': accuracy}, output_folder, f'{set_type}_results.json')


def validate_arguments(args):
    
    mutually_exclusive_params = [args.training, args.inference, args.feature_extract, args.create_dict,
                                 args.visualize_gamma_beta, args.visualize_grad_cam, args.lr_finder,
                                 args.write_clear_mean_to_config, args.random_answer_baseline,
                                 args.random_weight_baseline, args.prepare_images]

    assert sum(mutually_exclusive_params) == 1, \
        "[ERROR] Can only do one task at a time " \
        "(--training, --inference, --visualize_gamma_beta, --create_dict, --feature_extract --visualize_grad_cam " \
        "--prepare_images, --lr_finder, --write_clear_mean_to_config, --random_answer_baseline, --random_weight_baseline)"

    mutually_exclusive_params = [args.raw_img_resize_based_on_height, args.raw_img_resize_based_on_width]
    assert sum(mutually_exclusive_params) < 2, "[ERROR] Image resize can be either --raw_img_resize_based_on_height " \
                                               "or --raw_img_resize_based_on_width but not both"

    mutually_exclusive_params = [args.pad_to_square_images, args.resize_to_square_images]
    assert sum(mutually_exclusive_params) < 2, "[ERROR] Can either --pad_to_square_images or --resize_to_square_images"


def get_task_from_args(args):
    if args.training:
        return "train_film"
    elif args.inference:
        return "inference"
    elif args.visualize_gamma_beta:
        return "visualize_gamma_beta"
    elif args.visualize_grad_cam:
        return "visualize_grad_cam"
    elif args.feature_extract:
        return "feature_extract"
    elif args.prepare_images:
        return "prepare_images"
    elif args.create_dict:
        return "create_dict"
    elif args.lr_finder:
        return 'lr_finder'
    elif args.write_clear_mean_to_config:
        return 'write_clear_mean_to_config'
    elif args.random_weight_baseline:
        return 'random_weight_baseline'
    elif args.random_answer_baseline:
        return 'random_answer_baseline'

    assert False, "Arguments don't specify task"

def main(args):
    # Parameter validation
    validate_arguments(args)
    task = get_task_from_args(args)

    output_name = args.version_name + "_" + args.output_name_suffix if args.output_name_suffix else args.version_name
    print("\nTask '%s' for version '%s'\n" % (task.replace('_', ' ').title(), output_name))

    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    # Paths
    data_path = "%s/%s" % (args.data_root_path, args.version_name)

    output_task_folder = "%s/%s" % (args.output_root_path, task)
    output_experiment_folder = "%s/%s" % (output_task_folder, output_name)
    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y-%m-%d_%Hh%M")
    output_dated_folder = "%s/%s" % (output_experiment_folder, current_datetime_str)

    # Flags
    continuing_training = args.training and args.continue_training
    restore_model_weights = args.inference or continuing_training or args.visualize_grad_cam
    create_output_folder = not args.create_dict and not args.feature_extract and not args.write_clear_mean_to_config
    instantiate_model = not args.create_dict and not args.write_clear_mean_to_config and \
                        'gamma_beta' not in task and 'random_answer' not in task
    use_tensorboard = 'train' in task
    create_loss_criterion = args.training or args.lr_finder
    create_optimizer = args.training or args.lr_finder

    if continuing_training and args.film_model_weight_path is None:
        args.film_mode_weight_path = 'latest'

    # Make sure we are not normalizing beforce calculating mean and std
    if args.write_clear_mean_to_config:
        args.normalize_with_imagenet_stats = False
        args.normalize_with_clear_stats = False

    args.dict_folder = args.preprocessed_folder_name if args.dict_folder is None else args.dict_folder
    if args.dict_file_path is None:
        args.dict_file_path = "%s/%s/dict.json" % (data_path, args.dict_folder)

    film_model_config = get_config(args.config_path)

    early_stopping = not args.no_early_stopping and film_model_config['early_stopping']['enable']
    film_model_config['early_stopping']['enable'] =  early_stopping

    if create_output_folder:
        # TODO : See if this is optimal file structure
        create_folder_if_necessary(args.output_root_path)
        create_folder_if_necessary(output_task_folder)
        create_folder_if_necessary(output_experiment_folder)
        create_folder_if_necessary(output_dated_folder)

        # Save arguments & config to output folder
        save_json(args, output_dated_folder, filename="arguments.json")
        save_git_revision(output_dated_folder)

        if instantiate_model:
            save_json(film_model_config, output_dated_folder, filename='config_%s.json' % film_model_config['input']['type'])

            # Copy dictionary file used
            shutil.copyfile(args.dict_file_path, "%s/dict.json" % output_dated_folder)

    if args.no_img_resize or film_model_config['input']['type'].lower() != 'raw':
        args.raw_img_resize_val = None

    device = f'cuda:{args.gpu_index}' if torch.cuda.is_available() and not args.use_cpu else 'cpu'
    print("Using device '%s'" % device)

    ####################################
    #   Dataloading
    ####################################

    transforms_list = []

    # Bundle together ToTensor and ImgBetweenZeroOne, need to be one after the other for other transforms to work
    to_tensor_transform = [ToTensor()]
    if not args.keep_image_range:
        to_tensor_transform.append(ImgBetweenZeroOne())

    if film_model_config['input']['type'] == 'raw':

        if args.no_feature_extractor:
            feature_extractor_config = None
        else:
            feature_extractor_config = {'version': 101, 'layer_index': args.feature_extractor_layer_index}   # Idx 6 -> Block3/unit22

        if args.raw_img_resize_val:
            if args.raw_img_resize_based_on_width:
                resize_transform = ResizeImgBasedOnWidth
            else:
                # By default, we resize according to height
                resize_transform = ResizeImgBasedOnHeight
            transforms_list.append(resize_transform(args.raw_img_resize_val))

        # TODO : Add data augmentation ?

        transforms_list += to_tensor_transform

        if args.normalize_with_imagenet_stats or args.normalize_with_clear_stats:
            if args.normalize_with_imagenet_stats:
                stats = film_model_config['preprocessing']['imagenet_stats']
            else:
                stats = film_model_config['preprocessing']['clear_stats']
                
            transforms_list.append(NormalizeSample(mean=stats['mean'], std=stats['std'], inplace=True))

    else:
        transforms_list += to_tensor_transform
        feature_extractor_config = None

    transforms_to_apply = transforms.Compose(transforms_list)

    print("Creating Datasets")
    train_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'train',
                                  dict_file_path=args.dict_file_path, transforms=transforms_to_apply,
                                  tokenize_text=not args.create_dict,
                                  preprocessed_folder_name=args.preprocessed_folder_name)

    val_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'val',
                                dict_file_path=args.dict_file_path, transforms=transforms_to_apply,
                                tokenize_text=not args.create_dict,
                                preprocessed_folder_name=args.preprocessed_folder_name)

    test_dataset = CLEAR_dataset(args.data_root_path, args.version_name, film_model_config['input'], 'test',
                                 dict_file_path=args.dict_file_path, transforms=transforms_to_apply,
                                 tokenize_text=not args.create_dict,
                                 preprocessed_folder_name=args.preprocessed_folder_name)

    #trickytest_dataset = CLEAR_dataset(data_path, film_model_config['input'], 'trickytest',
    #                             dict_file_path=args.dict_file_path,
    #                             transforms=transforms.Compose(transforms_list + [ToTensor()]))

    if args.pad_to_largest_image or args.pad_to_square_images:
        # We need the dataset object to retrieve images dims so we have to manually add transforms
        max_train_img_dims = train_dataset.get_max_width_image_dims()
        max_val_img_dims = val_dataset.get_max_width_image_dims()
        max_test_img_dims = test_dataset.get_max_width_image_dims()

        if args.pad_to_largest_image:
            train_dataset.add_transform(PadTensor(max_train_img_dims))
            val_dataset.add_transform(PadTensor(max_val_img_dims))
            test_dataset.add_transform(PadTensor(max_test_img_dims))

        if args.pad_to_square_images or args.resize_to_square_images:
            train_biggest_dim = max(max_train_img_dims)
            val_biggest_dim = max(max_val_img_dims)
            test_biggest_dim = max(max_test_img_dims)

            if args.resize_to_square_images:
                to_square_transform = ResizeTensor
            else:
                to_square_transform = PadTensor

            train_dataset.add_transform(to_square_transform((train_biggest_dim, train_biggest_dim)))
            val_dataset.add_transform(to_square_transform((val_biggest_dim, val_biggest_dim)))
            test_dataset.add_transform(to_square_transform((test_biggest_dim, test_biggest_dim)))


    print("Creating Dataloaders")
    collate_fct = CLEAR_collate_fct(padding_token=train_dataset.get_padding_token())

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=collate_fct)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=4, collate_fn=collate_fct)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, collate_fn=collate_fct)

    #trickytest_dataloader = DataLoader(trickytest_dataset, batch_size=args.batch_size, shuffle=False,
    #                             num_workers=4, collate_fn=train_dataset.CLEAR_collate_fct)

    # This should only be used once to get the dataset mean & std. We could write it to the data folder in json format
    # FIXME : THIS IS AFFECTING THE RESULTS --> Should restore rng state if doing this
    #mean, std = calc_mean_and_std(train_test_dataloader, device=device)


    ####################################
    #   Model Definition
    ####################################
    if instantiate_model:
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

            # If path specified is a date, we construct the path to the best model weights for the specified run
            base_path = "%s/train_film/%s/%s" % (args.output_root_path, output_name, args.film_model_weight_path)
            # Note : We might redo some epoch when continuing training because the 'best' epoch is not necessarely the last
            suffix = "best/model.pt.tar"

            if is_date_string(args.film_model_weight_path):
                args.film_model_weight_path = "%s/%s" % (base_path, suffix)
            elif args.film_model_weight_path == 'latest':
                # The 'latest' symlink will be overriden by this run (If continuing training).
                # Use real path of latest experiment
                symlink_value = os.readlink(base_path)
                clean_base_path = base_path[:-(len(args.film_model_weight_path) + 1)]
                args.film_model_weight_path = '%s/%s/%s' % (clean_base_path, symlink_value, suffix)

            save_json({'restored_film_weight_path': args.film_model_weight_path},
                      output_dated_folder, 'restored_from.json')

            checkpoint = torch.load(args.film_model_weight_path, map_location=device)

            if device != 'cpu':
                if 'torch' in checkpoint['rng_state']:
                    checkpoint['rng_state']['torch'] = checkpoint['rng_state']['torch'].cpu()

                if 'torch_cuda' in checkpoint['rng_state']:
                    checkpoint['rng_state']['torch_cuda'] = checkpoint['rng_state']['torch_cuda'].cpu()

            # We need non-strict because feature extractor weight are not included in the saved state dict
            film_model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        trainable_parameters = filter(lambda p: p.requires_grad, film_model.parameters())

        if create_optimizer:
            if film_model_config['optimizer'].get('type', '') == 'sgd':
                optimizer = torch.optim.SGD(trainable_parameters, lr=film_model_config['optimizer']['learning_rate'],
                                            momentum=film_model_config['optimizer']['momentum'],
                                            weight_decay=film_model_config['optimizer']['weight_decay'])
            else:
                optimizer = torch.optim.Adam(trainable_parameters, lr=film_model_config['optimizer']['learning_rate'],
                                             weight_decay=film_model_config['optimizer']['weight_decay'])

        if create_loss_criterion:
            loss_criterion_tmp = nn.CrossEntropyLoss()

            if args.f1_score:
                def loss_criterion(outputs, answers):
                    loss = loss_criterion_tmp(outputs, answers)
                    _, preds = torch.max(outputs, 1)

                    return loss + (1 - calc_f1_score(preds, answers))
            else:
                loss_criterion = loss_criterion_tmp

        if device != 'cpu':
            if args.perf_over_determinist:
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True

        film_model.to(device)

        print("Model ready to run")

        if not args.no_model_summary:
            # Printing summary affects the random state (Raw Vs Pre-Extracted Features).
            # We restore it to ensure reproducibility between input type
            random_state = get_random_state()
            summary(film_model, [(22,), (1,), input_image_torch_shape], device=device)
            set_random_state(random_state)

        if use_tensorboard:
            # FIXME : What happen with test set? I guess we don't really care, we got our own visualisations for test run
            # Create tensorboard writer
            base_writer_path = '%s/%s/%s' % (args.tensorboard_folder, output_name, current_datetime_str)

            # TODO : Add 'comment' param with more infos on run. Ex : Raw vs Conv
            tensorboard = {
                'writers': {
                    'train': SummaryWriter('%s/train' % base_writer_path),
                    'val': SummaryWriter('%s/val' % base_writer_path)
                },
                'options': {
                    'save_images': args.tensorboard_save_images,
                    'save_texts': args.tensorboard_save_texts
                }
            }

            if args.tensorboard_save_graph:
                # FIXME : For now we are ignoring TracerWarnings. Not sure the saved graph is 100% accurate...
                import warnings
                warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)

                # FIXME : Test on GPU
                dummy_input = [torch.ones(2, 22, dtype=torch.long),
                               torch.ones(2, 1, dtype=torch.long),
                               torch.ones(2, *input_image_torch_shape, dtype=torch.float)]
                tensorboard['writers']['train'].add_graph(film_model, dummy_input)
        else:
            tensorboard = None

    if create_output_folder:
        # We create the symlink here so that bug in initialisation won't create a new 'latest' folder
        create_symlink_to_latest_folder(output_experiment_folder, current_datetime_str)

    if task == "train_film":
        if args.cyclical_lr:
            total_nb_steps = args.nb_epoch * len(train_dataloader)

            cycle_length = film_model_config['optimizer']['cyclical']['cycle_length']

            if type(cycle_length) == int:
                # Cycle length define the number of step in the cycle
                cycle_step = cycle_length
            elif type(cycle_length) == float:
                # Cycle length is a ratio of the total nb steps
                cycle_step = int(total_nb_steps * cycle_length)

            scheduler = CyclicLR(optimizer, base_lr=film_model_config['optimizer']['cyclical']['base_learning_rate'],
                                 max_lr=film_model_config['optimizer']['cyclical']['max_learning_rate'],
                                 step_size_up=cycle_step//2,
                                 base_momentum=film_model_config['optimizer']['cyclical']['base_momentum'],
                                 max_momentum=film_model_config['optimizer']['cyclical']['max_momentum'])

        else:
            scheduler = None

        start_epoch = 0
        if args.continue_training:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            set_random_state(checkpoint['rng_state'])

        train_model(device=device, model=film_model, dataloaders={'train': train_dataloader, 'val': val_dataloader},
                    output_folder=output_dated_folder, criterion=loss_criterion, optimizer=optimizer,
                    scheduler=scheduler, nb_epoch=args.nb_epoch,
                    nb_epoch_to_keep=args.nb_epoch_stats_to_keep, start_epoch=start_epoch, tensorboard=tensorboard)

    elif task == "inference":
        inference_dataloader = test_dataloader
        set_inference(device=device, model=film_model, dataloader=inference_dataloader, criterion=nn.CrossEntropyLoss(),
                      output_folder=output_dated_folder)

    elif task == "create_dict":
        create_dict_from_questions(train_dataset, force_all_answers=args.force_dict_all_answer,
                                   output_folder_name=args.dict_folder, start_end_tokens=not args.no_start_end_tokens)

    elif task == "prepare_images":
        images_to_h5(dataloaders={'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader},
                     square_image=args.pad_to_square_images or args.resize_to_square_images,
                     output_folder_name=args.preprocessed_folder_name)

    elif task == "feature_extract":
        extract_features(device=device, feature_extractor=film_model.feature_extractor,
                         dataloaders={'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader},
                         output_folder_name=args.preprocessed_folder_name)

    elif task == "visualize_gamma_beta":
        visualize_gamma_beta(args.gamma_beta_path,
                             datasets={'train': train_dataset, 'val': val_dataset, 'test': test_dataset},
                             output_folder=output_dated_folder)

    elif task == "visualize_grad_cam":
        grad_cam_visualization(device=device, model=film_model, dataloader=train_dataloader,
                               output_folder=output_dated_folder)

    elif task == "lr_finder":
        get_lr_finder_curves(film_model, device, train_dataloader, output_dated_folder, args.nb_epoch, optimizer,
                             val_dataloader=val_dataloader, loss_criterion=loss_criterion)

    elif task == "write_clear_mean_to_config":
        write_clear_mean_to_config(train_dataloader, device, film_model_config, args.config_path,
                                   args.overwrite_clear_mean)

    elif task == 'random_answer_baseline':
        random_answer_baseline(train_dataloader, output_dated_folder)
        random_answer_baseline(val_dataloader, output_dated_folder)

    elif task == 'random_weight_baseline':
        random_weight_baseline(film_model, device, train_dataloader, output_dated_folder)
        random_weight_baseline(film_model, device, val_dataloader, output_dated_folder)

    if use_tensorboard:
        close_tensorboard_writers(tensorboard['writers'])

    time_elapsed = str(datetime.now() - current_datetime)

    print("Execution took %s" % time_elapsed)
    print()

    if create_output_folder:
        save_json({'time_elapsed': time_elapsed}, output_dated_folder, filename='timing.json')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
