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

from runner import train_model, set_inference, prepare_model
from visualization import visualize_gamma_beta, grad_cam_visualization, print_model_summary, save_graph_to_tensorboard
from preprocessing import create_dict_from_questions, extract_features, images_to_h5, get_lr_finder_curves
from preprocessing import write_clear_mean_to_config
from baselines import random_answer_baseline, random_weight_baseline

# NEW IMPORTS
from models.CLEAR_film_model import CLEAR_FiLM_model
from data_interfaces.CLEAR_dataset import CLEAR_dataset, CLEAR_collate_fct
from data_interfaces.transforms import ToTensor, ImgBetweenZeroOne, ResizeImgBasedOnHeight, ResizeImgBasedOnWidth, PadTensor, NormalizeSample, ResizeTensor
from models.torchsummary import summary
import torch
import time
from torch.utils.data import DataLoader

import torch.nn as nn
from torchvision import transforms
from tensorboardX import SummaryWriter


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
parser.add_argument("--h5_image_input", help="If set, images will be read from h5 file in preprocessed folder",
                    action='store_true')
parser.add_argument("--conv_feature_input", help="If set, conv feature will be read from h5 file in preprocessed folder",
                    action='store_true')
parser.add_argument("--inference_set", type=str, default='test', help="Define on which set the inference will run")
parser.add_argument("--dict_file_path", type=str, default=None, help="Define what dictionnary file should be used")
parser.add_argument("--normalize_with_imagenet_stats", help="Will normalize input images according to"
                                                       "ImageNet mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--normalize_with_clear_stats", help="Will normalize input images according to"
                                                         "CLEAR mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--raw_img_resize_val", type=int, default=None,
                    help="Specify the size to which the image will be resized (when working with RAW img)"
                         "The width is calculated according to the height in order to keep the ratio")
parser.add_argument("--raw_img_resize_based_on_height", action='store_true',
                    help="If set (with --raw_img_resize_val), the width of the image will be calculated according to "
                         "the height in order to keep the ratio. [Default option if neither "
                         "--raw_img_resize_based_on_height and --raw_img_resize_based_on_width are set]")
parser.add_argument("--raw_img_resize_based_on_width", action='store_true',
                    help="If set (with --raw_img_resize_val), the height of the image will be calculated according to "
                         "the width in order to keep the ratio")


parser.add_argument("--keep_image_range", help="Will NOT scale the image between 0-1 (Reverse --normalize_zero_one)",
                    action='store_true')
parser.add_argument("--normalize_zero_one", help="Will scale the image between 0-1 (This is set by default when"
                                                 " working with RAW images. Can be overridden with --keep_image_range)",
                    action='store_true')
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


# Argument handling
def validate_arguments(args):
    
    mutually_exclusive_params = [args['training'], args['inference'], args['feature_extract'], args['create_dict'],
                                 args['visualize_gamma_beta'], args['visualize_grad_cam'], args['lr_finder'],
                                 args['write_clear_mean_to_config'], args['random_answer_baseline'],
                                 args['random_weight_baseline'], args['prepare_images']]

    assert sum(mutually_exclusive_params) == 1, \
        "[ERROR] Can only do one task at a time " \
        "(--training, --inference, --visualize_gamma_beta, --create_dict, --feature_extract --visualize_grad_cam " \
        "--prepare_images, --lr_finder, --write_clear_mean_to_config, --random_answer_baseline, --random_weight_baseline)"

    mutually_exclusive_params = [args['raw_img_resize_based_on_height'], args['raw_img_resize_based_on_width']]
    assert sum(mutually_exclusive_params) < 2, "[ERROR] Image resize can be either --raw_img_resize_based_on_height " \
                                               "or --raw_img_resize_based_on_width but not both"

    mutually_exclusive_params = [args['pad_to_square_images'], args['resize_to_square_images']]
    assert sum(mutually_exclusive_params) < 2, "[ERROR] Can either --pad_to_square_images or --resize_to_square_images"


def create_flags_from_args(task, args):
    flags = {}

    flags['continuing_training'] = args['training'] and args['continue_training']
    flags['restore_model_weights'] = args['inference'] or flags['continuing_training'] or args['visualize_grad_cam']
    flags['create_output_folder'] = not args['create_dict'] and not args['feature_extract'] and not args['write_clear_mean_to_config']
    flags['use_tensorboard'] = 'train' in task
    flags['create_loss_criterion'] = args['training'] or args['lr_finder']
    flags['create_optimizer'] = args['training'] or args['lr_finder']
    flags['force_sgd_optimizer'] = args['lr_finder'] or args['cyclical_lr']
    flags['instantiate_model'] = not args['create_dict'] and not args['write_clear_mean_to_config'] and \
                                 'gamma_beta' not in task and 'random_answer' not in task and not args['prepare_images']

    return flags


def get_paths_from_args(task, args):
    paths = {}

    paths["output_name"] = args['version_name'] + "_" + args['output_name_suffix'] if args['output_name_suffix'] else args['version_name']
    paths["data_path"] = "%s/%s" % (args['data_root_path'], args['version_name'])
    paths["output_task_folder"] = "%s/%s" % (args['output_root_path'], task)
    paths["output_experiment_folder"] = "%s/%s" % (paths["output_task_folder"], paths["output_name"])
    paths["current_datetime"] = datetime.now()
    paths["current_datetime_str"] = paths["current_datetime"].strftime("%Y-%m-%d_%Hh%M")
    paths["output_dated_folder"] = "%s/%s" % (paths["output_experiment_folder"], paths["current_datetime_str"])

    return paths


def get_task_from_args(args):
    if args['training']:
        return "train_film"
    elif args['inference']:
        return "inference"
    elif args['visualize_gamma_beta']:
        return "visualize_gamma_beta"
    elif args['visualize_grad_cam']:
        return "visualize_grad_cam"
    elif args['feature_extract']:
        return "feature_extract"
    elif args['prepare_images']:
        return "prepare_images"
    elif args['create_dict']:
        return "create_dict"
    elif args['lr_finder']:
        return 'lr_finder'
    elif args['write_clear_mean_to_config']:
        return 'write_clear_mean_to_config'
    elif args['random_weight_baseline']:
        return 'random_weight_baseline'
    elif args['random_answer_baseline']:
        return 'random_answer_baseline'

    assert False, "Arguments don't specify task"


def update_arguments(args, paths, flags):
    if args['conv_feature_input']:
        args['input_image_type'] = "conv"
    elif args['h5_image_input']:
        args['input_image_type'] = "raw_h5"
    else:
        args['input_image_type'] = "raw"

    if args['input_image_type'] == 'raw':
        # Default values when in RAW mode
        args['normalize_zero_one'] = True

        if args['raw_img_resize_val'] is None:
            args['raw_img_resize_val'] = 224

    args['normalize_zero_one'] = args['normalize_zero_one'] and not args['keep_image_range']

    if flags['continuing_training'] and args['film_model_weight_path'] is None:
        args['film_model_weight_path'] = 'latest'

    # Make sure we are not normalizing beforce calculating mean and std
    if args['write_clear_mean_to_config']:
        args['normalize_with_imagenet_stats'] = False
        args['normalize_with_clear_stats'] = False

    args['dict_folder'] = args['preprocessed_folder_name'] if args['dict_folder'] is None else args['dict_folder']
    if args['dict_file_path'] is None:
        args['dict_file_path'] = "%s/%s/dict.json" % (paths["data_path"], args['dict_folder'])

    # By default the start_epoch should is 0. Will only be modified if loading from checkpoint
    args["start_epoch"] = 0


def get_transforms_from_args(args, preprocessing_config):
    transforms_list = []

    # Bundle together ToTensor and ImgBetweenZeroOne, need to be one after the other for other transforms to work
    to_tensor_transform = [ToTensor()]
    if args['normalize_zero_one']:
        to_tensor_transform.append(ImgBetweenZeroOne())

    if args['input_image_type'].startswith('raw'):

        if args['raw_img_resize_val']:
            if args['raw_img_resize_based_on_width']:
                resize_transform = ResizeImgBasedOnWidth
            else:
                # By default, we resize according to height
                resize_transform = ResizeImgBasedOnHeight
            transforms_list.append(resize_transform(args['raw_img_resize_val']))

        # TODO : Add data augmentation ?

        transforms_list += to_tensor_transform

        if args['normalize_with_imagenet_stats'] or args['normalize_with_clear_stats']:
            if args['normalize_with_imagenet_stats']:
                stats = preprocessing_config['imagenet_stats']
            else:
                stats = preprocessing_config['clear_stats']

            transforms_list.append(NormalizeSample(mean=stats['mean'], std=stats['std'], inplace=True))

    else:
        transforms_list += to_tensor_transform
        feature_extractor_config = None

    return transforms.Compose(transforms_list)


def get_feature_extractor_config_from_args(args):
    if args['no_feature_extractor']:
        return None
    else:
        return {'version': 101, 'layer_index': args['feature_extractor_layer_index']}  # Idx 6 -> Block3/unit22


# Data loading & preparation
def create_datasets(args, preprocessing_config):
    print("Creating Datasets")
    transforms_to_apply = get_transforms_from_args(args, preprocessing_config)

    datasets = {
        'train': CLEAR_dataset(args['data_root_path'], args['version_name'], args['input_image_type'], 'train',
                               dict_file_path=args['dict_file_path'], transforms=transforms_to_apply,
                               tokenize_text=not args['create_dict'],
                               preprocessed_folder_name=args['preprocessed_folder_name']),

        'val': CLEAR_dataset(args['data_root_path'], args['version_name'], args['input_image_type'], 'val',
                             dict_file_path=args['dict_file_path'], transforms=transforms_to_apply,
                             tokenize_text=not args['create_dict'],
                             preprocessed_folder_name=args['preprocessed_folder_name']),

        'test': CLEAR_dataset(args['data_root_path'], args['version_name'], args['input_image_type'], 'test',
                              dict_file_path=args['dict_file_path'], transforms=transforms_to_apply,
                              tokenize_text=not args['create_dict'],
                              preprocessed_folder_name=args['preprocessed_folder_name'])
    }

    if args['pad_to_largest_image'] or args['pad_to_square_images']:
        # We need the dataset object to retrieve images dims so we have to manually add transforms
        max_train_img_dims = datasets['train'].get_max_width_image_dims()
        max_val_img_dims = datasets['val'].get_max_width_image_dims()
        max_test_img_dims = datasets['test'].get_max_width_image_dims()

        if args['pad_to_largest_image']:
            datasets['train'].add_transform(PadTensor(max_train_img_dims))
            datasets['val'].add_transform(PadTensor(max_val_img_dims))
            datasets['test'].add_transform(PadTensor(max_test_img_dims))

        if args['pad_to_square_images'] or args['resize_to_square_images']:
            train_biggest_dim = max(max_train_img_dims)
            val_biggest_dim = max(max_val_img_dims)
            test_biggest_dim = max(max_test_img_dims)

            if args['resize_to_square_images']:
                to_square_transform = ResizeTensor
            else:
                to_square_transform = PadTensor

            datasets['train'].add_transform(to_square_transform((train_biggest_dim, train_biggest_dim)))
            datasets['val'].add_transform(to_square_transform((val_biggest_dim, val_biggest_dim)))
            datasets['test'].add_transform(to_square_transform((test_biggest_dim, test_biggest_dim)))

    return datasets


def create_dataloaders(args, datasets, nb_process=8):
    print("Creating Dataloaders")
    collate_fct = CLEAR_collate_fct(padding_token=datasets['train'].get_padding_token())

    # FIXME : Should take into account --nb_process, or at least the nb of core on the machine
    return {
        'train': DataLoader(datasets['train'], batch_size=args['batch_size'], shuffle=True,
                            num_workers=4, collate_fn=collate_fct),

        'val': DataLoader(datasets['val'], batch_size=args['batch_size'], shuffle=True,
                          num_workers=4, collate_fn=collate_fct),

        'test': DataLoader(datasets['test'], batch_size=args['batch_size'], shuffle=False,
                           num_workers=4, collate_fn=collate_fct)
    }


def execute_task(task, args, output_dated_folder, dataloaders, model, model_config, device, optimizer=None,
                 loss_criterion=None, scheduler=None, tensorboard=None):
    if task == "train_film":
        train_model(device=device, model=model, dataloaders=dataloaders,
                    output_folder=output_dated_folder, criterion=loss_criterion, optimizer=optimizer,
                    scheduler=scheduler, nb_epoch=args['nb_epoch'],
                    nb_epoch_to_keep=args['nb_epoch_stats_to_keep'], start_epoch=args['start_epoch'],
                    tensorboard=tensorboard)

    elif task == "inference":
        set_inference(device=device, model=model, dataloader=dataloaders['test'], criterion=nn.CrossEntropyLoss(),
                      output_folder=output_dated_folder)

    elif task == "create_dict":
        create_dict_from_questions(dataloaders['train'].dataset, force_all_answers=args['force_dict_all_answer'],
                                   output_folder_name=args['dict_folder'],
                                   start_end_tokens=not args['no_start_end_tokens'])

    elif task == "prepare_images":
        # TODO : Merge write_clear_mean_to_config here
        images_to_h5(dataloaders=dataloaders,
                     square_image=args['pad_to_square_images'] or args['resize_to_square_images'],
                     output_folder_name=args['preprocessed_folder_name'])

    elif task == "feature_extract":
        extract_features(device=device, feature_extractor=model.feature_extractor, dataloaders=dataloaders,
                         output_folder_name=args['preprocessed_folder_name'])

    elif task == "visualize_gamma_beta":
        visualize_gamma_beta(args['gamma_beta_path'], dataloaders=dataloaders, output_folder=output_dated_folder)

    elif task == "visualize_grad_cam":
        grad_cam_visualization(device=device, model=model, dataloader=dataloaders['train'],
                               output_folder=output_dated_folder)

    elif task == "lr_finder":
        get_lr_finder_curves(model, device, dataloaders['train'], output_dated_folder, args['nb_epoch'], optimizer,
                             val_dataloader=dataloaders['val'], loss_criterion=loss_criterion)

    elif task == "write_clear_mean_to_config":
        write_clear_mean_to_config(dataloaders['train'], device, model_config, args['config_path'],
                                   args['overwrite_clear_mean'])

    elif task == 'random_answer_baseline':
        random_answer_baseline(dataloaders['train'], output_dated_folder)
        random_answer_baseline(dataloaders['val'], output_dated_folder)

    elif task == 'random_weight_baseline':
        random_weight_baseline(model, device, dataloaders['train'], output_dated_folder)
        random_weight_baseline(model, device, dataloaders['val'], output_dated_folder)


def create_folders_save_config(args, paths, flags, model_config):
    if flags['create_output_folder']:
        # TODO : See if this is optimal file structure
        create_folder_if_necessary(args['output_root_path'])
        create_folder_if_necessary(paths["output_task_folder"])
        create_folder_if_necessary(paths["output_experiment_folder"])
        create_folder_if_necessary(paths["output_dated_folder"])

        # Save arguments & config to output folder
        save_json(args, paths["output_dated_folder"], filename="arguments.json")
        save_git_revision(paths["output_dated_folder"])

        if flags['instantiate_model']:
            save_json(model_config, paths["output_dated_folder"],
                      filename='config_%s_input.json' % args['input_image_type'])

            # Copy dictionary file used
            shutil.copyfile(args['dict_file_path'], "%s/dict.json" % paths["output_dated_folder"])


def prepare_tensorboard(args, flags, paths):
    if flags['use_tensorboard']:
        # FIXME : What happen with test set? I guess we don't really care, we got our own visualisations for test run
        # Create tensorboard writer
        base_writer_path = '%s/%s/%s' % (
        args['tensorboard_folder'], paths["output_name"], paths["current_datetime_str"])

        # TODO : Add 'comment' param with more infos on run. Ex : Raw vs Conv
        tensorboard = {
            'writers': {
                'train': SummaryWriter('%s/train' % base_writer_path),
                'val': SummaryWriter('%s/val' % base_writer_path)
            },
            'options': {
                'save_images': args['tensorboard_save_images'],
                'save_texts': args['tensorboard_save_texts']
            }
        }
    else:
        tensorboard = None

    return tensorboard


def prepare_for_task(args):
    ####################################
    #   Argument & Config parsing
    ####################################
    args = vars(args)  # Convert args object to dict
    validate_arguments(args)
    task = get_task_from_args(args)
    paths = get_paths_from_args(task, args)
    flags = create_flags_from_args(task, args)
    update_arguments(args, paths, flags)
    device = f'cuda:{args["gpu_index"]}' if torch.cuda.is_available() and not args['use_cpu'] else 'cpu'

    if args['random_seed'] is not None:
        set_random_seed(args['random_seed'])

    print("\nTask '%s' for version '%s'\n" % (task.replace('_', ' ').title(), paths["output_name"]))
    print("Using device '%s'" % device)

    film_model_config = get_config(args['config_path'])
    # FIXME : Should be in args ?
    early_stopping = not args['no_early_stopping'] and film_model_config['early_stopping']['enable']
    film_model_config['early_stopping']['enable'] = early_stopping

    # Create required folders if necessary
    create_folders_save_config(args, paths, flags, film_model_config)

    # Make sure all variables exists
    film_model, optimizer, loss_criterion, tensorboard, scheduler = None, None, None, None, None

    ####################################
    #   Dataloading
    ####################################
    datasets = create_datasets(args, film_model_config['preprocessing'])
    dataloaders = create_dataloaders(args, datasets, nb_process=8)

    ####################################
    #   Model Definition
    ####################################
    if flags['instantiate_model']:
        input_image_torch_shape = datasets['train'].get_input_shape(
            channel_first=True)  # Torch size have Channel as first dimension
        feature_extractor_config = get_feature_extractor_config_from_args(args)

        film_model, optimizer, loss_criterion, scheduler = prepare_model(args, flags, paths, dataloaders, device,
                                                                         film_model_config, input_image_torch_shape,
                                                                         feature_extractor_config)

        if not args['no_model_summary']:
            print_model_summary(film_model, input_image_torch_shape, device)

        if flags['use_tensorboard']:
            tensorboard = prepare_tensorboard(args, flags, paths)
            if args['tensorboard_save_graph']:
                save_graph_to_tensorboard(film_model, tensorboard, input_image_torch_shape)

    return (
        (task, args, flags, paths, device),
        dataloaders,
        (film_model_config, film_model, optimizer, loss_criterion, scheduler, tensorboard)
    )


def on_exit_action(flags, paths, tensorboard):
    if flags['use_tensorboard']:
        close_tensorboard_writers(tensorboard['writers'])

    time_elapsed = str(datetime.now() - paths["current_datetime"])

    print("Execution took %s\n" % time_elapsed)

    if flags['create_output_folder']:
        save_json({'time_elapsed': time_elapsed}, paths["output_dated_folder"], filename='timing.json')


def main(args):

    task_and_more, dataloaders, model_and_more = prepare_for_task(args)
    task, args, flags, paths, device = task_and_more
    film_model_config, film_model, optimizer, loss_criterion, scheduler, tensorboard = model_and_more

    ####################################
    #   Task Execution
    ####################################
    if flags['create_output_folder']:
        # We create the symlink here so that bug in initialisation won't create a new 'latest' folder
        create_symlink_to_latest_folder(paths["output_experiment_folder"], paths["current_datetime_str"])

    execute_task(task, args, paths["output_dated_folder"], dataloaders, film_model, film_model_config, device,
                 optimizer, loss_criterion, scheduler, tensorboard)

    ####################################
    #   Exit
    ####################################
    on_exit_action(flags, paths, tensorboard)


# FIXME : Args is a namespace so it is available everywhere. Not a great idea to shadow it (But we get a dict key error if we try to access it so it is easiy catchable
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
