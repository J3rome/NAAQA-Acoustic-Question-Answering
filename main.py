import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms

from runner import train_model, prepare_model, inference
from baselines import random_answer_baseline, random_weight_baseline
from preprocessing import create_dict_from_questions, extract_features, images_to_h5, get_lr_finder_curves
from preprocessing import write_clear_stats_to_file
from visualization import visualize_gamma_beta, grad_cam_visualization
from data_interfaces.CLEAR_dataset import CLEAR_dataset, CLEAR_collate_fct
from data_interfaces.transforms import ImgBetweenZeroOne, PadTensor, NormalizeSample, PadTensorHeight
from data_interfaces.transforms import ResizeTensorBasedOnMaxWidth

from utils.file import save_model_config, save_json, read_json, create_symlink_to_latest_folder
from utils.file import create_folders_save_args, fix_best_epoch_symlink_if_necessary, get_clear_stats
from utils.random import set_random_seed, get_random_state, set_random_state
from utils.visualization import save_model_summary, save_graph_to_tensorboard
from utils.argument_parsing import get_args_task_flags_paths, get_feature_extractor_config_from_args
from utils.logging import create_tensorboard_writers, close_tensorboard_writers
from utils.generic import get_imagenet_stats

from models.Resnet_feature_extractor import Resnet_feature_extractor
from models.tools.TF_weight_transfer import tf_weight_transfer


parser = argparse.ArgumentParser('FiLM model for CLEAR Dataset (Acoustic Question Answering)', fromfile_prefix_chars='@')

# Tasks
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
parser.add_argument("--notebook_data_analysis", help="Will prepare dataloaders for analysis in notebook "
                                                     "(Should not be run via main.py)", action='store_true')
parser.add_argument("--notebook_model_inference", help="Will prepare dataloaders & model for inference in notebook"
                                                     "(Should not be run via main.py)", action='store_true')
parser.add_argument("--calc_clear_mean", help="Will calculate the mean and std of the dataset and write it in a json at"
                                              "the root of the dataset", action='store_true')

parser.add_argument("--tf_weight_transfer", help="Will create a pytorch checkpoint from dumped tensorflow weights."
                                                 "path to the weights are specified by --tf_weight_path",
                    action='store_true')

# Image Preprocessing
parser.add_argument("--img_resize_height", type=int, default=224,
                    help="Specify the height to which the image will be resized")
parser.add_argument("--img_resize_width", type=int, default=224,
                    help="Specify the width to which the image with the maximum width will be resized. "
                         "Smaller images will be resized according to the same ratio as {img_resize_width}/{max_width}")
parser.add_argument("--resize_img", help="Will resize the image according to --img_resize_height and --img_resize_width",
                    action='store_true')
parser.add_argument("--resize_width_only", help="Will prevent the height from being resized (Do nothing if --pad_height)",
                    action='store_true')
parser.add_argument("--resize_img_width_no_ratio", help="Will resize images to --img_resize_width without keeping ratio",
                    action='store_true')
parser.add_argument("--pad_to_largest_image", help="If set, images will be padded to meet the largest image in the set."
                                                   "All input will have the same size.", action='store_true')
parser.add_argument("--pad_height", help="If set, the height will be padded to --img_resize_height instead of resized",
                    action='store_true')
parser.add_argument("--keep_image_range", help="Will NOT scale the image between 0-1 (Reverse --normalize_zero_one)",
                    action='store_true')
parser.add_argument("--normalize_zero_one", help="Will scale the image between 0-1 (This is set by default when"
                                                 " working with RAW images. Can be overridden with --keep_image_range)",
                    action='store_true')
parser.add_argument("--normalize_with_imagenet_stats", help="Will normalize input images according to"
                                                       "ImageNet mean & std (Only with RAW input)", action='store_true')
parser.add_argument("--normalize_with_clear_stats", help="Will normalize input images according to"
                                                         "CLEAR mean & std (Only with RAW input)", action='store_true')

# Question Preprocessing parameters
parser.add_argument("--no_start_end_tokens", help="Constants tokens won't be added to the question "
                                                  "(<start> & <end> tokens)", action='store_true')
parser.add_argument("--force_dict_all_answer", help="Will make sure that all answers are included in the dict" +
                                                    "(not just the one appearing in the train set)" +
                                                    " -- Preprocessing option" , action='store_true')
parser.add_argument("--dict_file_path", type=str, default=None, help="Define what dictionary file should be used")
parser.add_argument("--dict_folder", type=str, default=None,
                    help="Directory where to store/retrieve generated dictionary. "
                         "If --dict_file_path is used, this will be ignored")

# Model parameters
parser.add_argument("--config_path", type=str, default='config/film.json', help="Path to Film pretrained ckpt file")
parser.add_argument("--film_model_weight_path", type=str, default=None, help="Path to Film pretrained weight file")
parser.add_argument("--feature_extractor_layer_index", type=int, default=6, help="Layer id of the pretrained Resnet")
parser.add_argument("--no_feature_extractor", help="Raw images won't go through Resnet feature extractor before "
                                                    "training", action='store_true')

# Training parameters
parser.add_argument("--nb_epoch", type=int, default=15, help="Nb of epoch for training")
parser.add_argument("--nb_epoch_stats_to_keep", type=int, default=5, help="Nb of epoch stats to keep for training")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size (For training and inference)")
parser.add_argument("--continue_training", help="Will use the --film_model_weight_path as  training starting point",
                    action='store_true')
parser.add_argument("--random_seed", type=int, default=None, help="Random seed used for the experiment")
parser.add_argument("--use_cpu", help="Model will be run/train on CPU", action='store_true')
parser.add_argument("--no_early_stopping", help="Override the early stopping config", action='store_true')
parser.add_argument("--gpu_index", type=str, default='0', help="Index of the GPU to use")
parser.add_argument("--perf_over_determinist", help="Will let torch use nondeterministic algorithms (Better "
                                                    "performance but less reproductibility)", action='store_true')
parser.add_argument("--f1_score", help="Use f1 score in loss calculation", action='store_true')
parser.add_argument("--cyclical_lr", help="Will use cyclical learning rate (Bounds in config.json)",
                    action='store_true')
parser.add_argument("--enable_image_cache", help="Will enable image loading cache (In RAM)", action='store_true')
parser.add_argument("--max_image_cache_size", type=int, default=5000,
                    help="Max number of images that can be stored in cache")
parser.add_argument("--run_test_after_training", help="Will run on test set after the training is done",
                    action='store_true')
parser.add_argument("--stop_at_val_acc", type=float, default=None,
                    help="Will stop the training if validation accuracy reach this threshold. "
                         "Specified in integer percentage (95).")

# Input parameters
parser.add_argument("--h5_image_input", help="If set, images will be read from h5 file in preprocessed folder",
                    action='store_true')
parser.add_argument("--conv_feature_input", help="If set, conv feature will be read from h5 file in preprocessed folder",
                    action='store_true')
parser.add_argument("--data_root_path", type=str, default='data', help="Directory with data")
parser.add_argument("--version_name", type=str, help="Name of the dataset version")
parser.add_argument("--test_dataset_path", type=str, default=None, help="Path to the test dataset (Optional)")

# Output parameters
parser.add_argument("--output_root_path", type=str, default='output', help="Directory with image")
parser.add_argument("--preprocessed_folder_name", type=str, default='preprocessed',
                    help="Directory where to store/are stored extracted features and token dictionary")
parser.add_argument("--output_name_suffix", type=str, default='', help="Suffix that will be appended to the version "
                                                                       "name (output & tensorboard)")
parser.add_argument("--gamma_beta_path", type=str, default=None, help="Path where gamma_beta values are stored "
                                                                          "(when using --visualize_gamma_beta)")
parser.add_argument("--tensorboard_folder", type=str, default='tensorboard',
                    help="Path where tensorboard data should be stored.")
parser.add_argument("--tensorboard_save_graph", help="Save model graph to tensorboard", action='store_true')
parser.add_argument("--tensorboard_save_images", help="Save input images to tensorboard", action='store_true')
parser.add_argument("--tensorboard_save_texts", help="Save input texts to tensorboard", action='store_true')

# Other parameters
parser.add_argument("--inference_set", type=str, default='test', help="Define on which set the inference will run")
parser.add_argument("--test_set_batch_size", type=int, default=None, help="Use different batchsize for test set (optional)")
parser.add_argument("--no_model_summary", help="Will hide the model summary", action='store_true')
parser.add_argument("--tf_weight_path", type=str, help="Specify where to load dumped tensorflow weights "
                                                       "(Used with --tf_weight_transfer)")
parser.add_argument("--single_worker", help="Will limit the number of worker for each dataloader to 1 "
                                            "(Useful when running multiple runs in parallel)", action='store_true')


def get_transforms_from_args(args, data_path):
    transforms_list = []

    if args['normalize_zero_one']:
        transforms_list.append(ImgBetweenZeroOne())

    if args['input_image_type'].startswith('raw'):

        # TODO : Add data augmentation ?

        if args['normalize_with_imagenet_stats'] or args['normalize_with_clear_stats']:
            if args['normalize_with_imagenet_stats']:
                stats = get_imagenet_stats()
            else:
                stats = get_clear_stats(data_path)

            transforms_list.append(NormalizeSample(mean=stats['mean'], std=stats['std'], inplace=True))

    return transforms.Compose(transforms_list)


# Data loading & preparation
def create_datasets(args, data_path, load_dataset_extra_stats=False):
    print("Creating Datasets")
    transforms_to_apply = get_transforms_from_args(args, data_path)

    if args['test_dataset_path']:
        if '/' in args['test_dataset_path']:
            # --test_dataset_path is full path
            splitted = args['test_dataset_path'].split('/')

            test_data_root_path = '/'.join(splitted[:-1])
            test_version_name = splitted[-1]
        else:
            # --test_dataset_path is version name
            test_data_root_path = args['data_root_path']
            test_version_name = args['test_dataset_path']
    else:
        test_data_root_path = args['data_root_path']
        test_version_name = args['version_name']

    datasets = {
        'train': CLEAR_dataset(args['data_root_path'], args['version_name'], args['input_image_type'], 'train',
                               dict_file_path=args['dict_file_path'], transforms=transforms_to_apply,
                               tokenize_text=not args['create_dict'], extra_stats=load_dataset_extra_stats,
                               preprocessed_folder_name=args['preprocessed_folder_name'],
                               use_cache=args['enable_image_cache'], max_cache_size=args['max_image_cache_size']),

        'val': CLEAR_dataset(args['data_root_path'], args['version_name'], args['input_image_type'], 'val',
                             dict_file_path=args['dict_file_path'], transforms=transforms_to_apply,
                             tokenize_text=not args['create_dict'], extra_stats=load_dataset_extra_stats,
                             preprocessed_folder_name=args['preprocessed_folder_name'],
                             use_cache=args['enable_image_cache'], max_cache_size=args['max_image_cache_size']),

        'test': CLEAR_dataset(test_data_root_path, test_version_name, args['input_image_type'], 'test',
                              dict_file_path=args['dict_file_path'], transforms=transforms_to_apply,
                              tokenize_text=not args['create_dict'], extra_stats=load_dataset_extra_stats,
                              preprocessed_folder_name=args['preprocessed_folder_name'],
                              use_cache=args['enable_image_cache'], max_cache_size=args['max_image_cache_size'])
    }

    if args['resize_img'] or args['pad_to_largest_image']:
        # We need the dataset object to retrieve images dims so we have to manually add transforms
        max_train_img_dims = datasets['train'].get_max_width_image_dims()
        max_val_img_dims = datasets['val'].get_max_width_image_dims()
        max_test_img_dims = datasets['test'].get_max_width_image_dims()

        if args['resize_img']:
            resize_height = args['img_resize_height']
            max_width_ref = max_train_img_dims[1]

            if args['pad_height'] or args['resize_width_only']:
                resize_height = None

            if args['resize_img_width_no_ratio']:
                max_width_ref = None

            resize_transform = ResizeTensorBasedOnMaxWidth(output_width=args['img_resize_width'],
                                                           max_width=max_width_ref,
                                                           output_height=resize_height)

            datasets['train'].add_transform(resize_transform)
            datasets['val'].add_transform(resize_transform)
            datasets['test'].add_transform(resize_transform)

            # Update max images dims after resize
            max_train_img_dims = resize_transform.get_resized_dim(max_train_img_dims[0], max_train_img_dims[1])
            max_val_img_dims = resize_transform.get_resized_dim(max_val_img_dims[0], max_val_img_dims[1])
            max_test_img_dims = resize_transform.get_resized_dim(max_test_img_dims[0], max_test_img_dims[1])

        if args['pad_height']:
            assert max_train_img_dims[0] < args['img_resize_height'], \
                'Resize Height must be bigger than the images height to be able to pad'

            # Update max images dims after padding
            max_train_img_dims = (args['img_resize_height'], max_train_img_dims[1])
            max_val_img_dims = (args['img_resize_height'], max_val_img_dims[1])
            max_test_img_dims = (args['img_resize_height'], max_test_img_dims[1])

            pad_tensor_transform = PadTensorHeight(args['img_resize_height'])

            datasets['train'].add_transform(pad_tensor_transform)
            datasets['val'].add_transform(pad_tensor_transform)
            datasets['test'].add_transform(pad_tensor_transform)

        if args['pad_to_largest_image']:
            datasets['train'].add_transform(PadTensor(max_train_img_dims))
            datasets['val'].add_transform(PadTensor(max_val_img_dims))
            datasets['test'].add_transform(PadTensor(max_test_img_dims))

    return datasets


def create_dataloaders(args, datasets, pin_memory=True):
    print("Creating Dataloaders")
    collate_fct = CLEAR_collate_fct(padding_token=datasets['train'].get_padding_token())

    test_set_batch_size = args['test_set_batch_size'] if args['test_set_batch_size'] else args['batch_size']

    return {
        'train': DataLoader(datasets['train'], batch_size=args['batch_size'], shuffle=True,
                            num_workers=3 if not args['single_worker'] else 1, collate_fn=collate_fct,
                            pin_memory=pin_memory),

        'val': DataLoader(datasets['val'], batch_size=args['batch_size'], shuffle=True,
                          num_workers=2 if not args['single_worker'] else 1, collate_fn=collate_fct,
                          pin_memory=pin_memory),

        'test': DataLoader(datasets['test'], batch_size=test_set_batch_size, shuffle=False,
                           num_workers=2 if not args['single_worker'] else 1, collate_fn=collate_fct,
                           pin_memory=pin_memory)
    }


def execute_task(task, args, output_dated_folder, dataloaders, model, model_config, device, optimizer=None,
                 loss_criterion=None, scheduler=None, tensorboard=None):

    if task == "training":
        try:
            train_model(device=device, model=model, dataloaders=dataloaders,
                        output_folder=output_dated_folder, criterion=loss_criterion, optimizer=optimizer,
                        scheduler=scheduler, nb_epoch=args['nb_epoch'], stop_at_val_acc=args['stop_at_val_acc'],
                        nb_epoch_to_keep=args['nb_epoch_stats_to_keep'], start_epoch=args['start_epoch'],
                        tensorboard=tensorboard)
        except KeyboardInterrupt:
            print("\n\n>>> Received keyboard interrupt, Gracefully terminating training\n")

        if args['run_test_after_training']:
            inference(device=device, model=model, set_type='test',
                      dataloader=dataloaders['test'], criterion=loss_criterion,
                      output_folder=output_dated_folder)

    elif task == "inference":
        assert args['inference_set'] in dataloaders, "Invalid set name. A dataloader must exist for this set"

        inference(device=device, model=model, set_type=args['inference_set'],
                  dataloader=dataloaders[args['inference_set']], criterion=loss_criterion,
                  output_folder=output_dated_folder)

    elif task == "create_dict":
        create_dict_from_questions(dataloaders['train'].dataset, force_all_answers=args['force_dict_all_answer'],
                                   output_folder_name=args['dict_folder'],
                                   start_end_tokens=not args['no_start_end_tokens'])

    elif task == "prepare_images":
        images_to_h5(dataloaders=dataloaders, output_folder_name=args['preprocessed_folder_name'])

        # Save generation args with h5 file
        save_json(args, f"{dataloaders['train'].dataset.root_folder_path}/{args['preprocessed_folder_name']}",
                  filename="arguments.json")

    elif task == "feature_extract":
        resnet_extractor = Resnet_feature_extractor(layer_index=args['feature_extractor_layer_index'])
        extract_features(device=device, feature_extractor=resnet_extractor,
                         dataloaders=dataloaders, output_folder_name=args['preprocessed_folder_name'])
        # Save generation args with h5 file
        save_json(args, f"{dataloaders['train'].dataset.root_folder_path}/{args['preprocessed_folder_name']}",
                  filename="arguments.json")

    elif task == "visualize_gamma_beta":
        visualize_gamma_beta(args['gamma_beta_path'], dataloaders=dataloaders, output_folder=output_dated_folder)

    elif task == "visualize_grad_cam":
        grad_cam_visualization(device=device, model=model, dataloader=dataloaders['train'],
                               output_folder=output_dated_folder)

    elif task == "lr_finder":
        get_lr_finder_curves(model, device, dataloaders['train'], output_dated_folder, args['nb_epoch'], optimizer,
                             val_dataloader=dataloaders['val'], loss_criterion=loss_criterion)

    elif task == "calc_clear_mean":
        write_clear_stats_to_file(dataloaders['train'], device)

    elif task == 'random_answer_baseline':
        random_answer_baseline(dataloaders['train'], output_dated_folder)
        random_answer_baseline(dataloaders['val'], output_dated_folder)

    elif task == 'random_weight_baseline':
        random_weight_baseline(model, device, dataloaders['train'], output_dated_folder)
        random_weight_baseline(model, device, dataloaders['val'], output_dated_folder)

    elif task == 'tf_weight_transfer':
        tf_weight_transfer(model, args['tf_weight_path'], output_dated_folder)

    assert not task.startswith('notebook'), "Task not meant to be run from main.py. " \
                                            "Used to prepare dataloaders & model for analysis in notebook"


def prepare_for_task(args):
    ####################################
    #   Argument & Config parsing
    ####################################
    args, task, flags, paths = get_args_task_flags_paths(args)
    device = f'cuda:{args["gpu_index"]}' if torch.cuda.is_available() and not args['use_cpu'] else 'cpu'

    if args['random_seed'] is not None:
        set_random_seed(args['random_seed'])

    # Save initial random state, will reset to this seed after this function
    # This is done to make sure that difference in model initialisation won't cause difference in training
    initial_random_state = get_random_state()

    print("\nTask '%s' for version '%s'\n" % (task.replace('_', ' ').title(), paths["output_name"]))
    print("Using device '%s'" % device)

    # Create required folders if necessary
    if flags['create_output_folder']:
        gpu_name = torch.cuda.get_device_name(args['gpu_index']) if 'cuda' in device else None
        create_folders_save_args(args, paths, gpu_name)

    # Make sure all variables exists
    film_model, film_model_config, optimizer, loss_criterion, tensorboard, scheduler = None, None, None, None, None, None

    ####################################
    #   Dataloading
    ####################################
    datasets = create_datasets(args, paths['data_path'], flags['load_dataset_extra_stats'])
    dataloaders = create_dataloaders(args, datasets)

    ####################################
    #   Model Definition
    ####################################
    if flags['instantiate_model']:
        film_model_config = read_json(args['config_path'])
        # FIXME : Should be in args ?
        early_stopping = not args['no_early_stopping'] and film_model_config['early_stopping']['enable']
        film_model_config['early_stopping']['enable'] = early_stopping

        if flags["force_sgd_optimizer"]:
            film_model_config['optimizer']['type'] = 'sgd'

        if flags['create_output_folder'] and flags['instantiate_model']:
            save_model_config(args, paths, film_model_config)

        input_image_torch_shape = datasets['train'].get_input_shape(channel_first=True)  # Torch size have Channel as first dimension
        feature_extractor_config = get_feature_extractor_config_from_args(args)

        film_model, optimizer, loss_criterion, scheduler = prepare_model(args, flags, paths, dataloaders, device,
                                                                         film_model_config, input_image_torch_shape,
                                                                         feature_extractor_config)

        save_model_summary(paths['output_dated_folder'], film_model, input_image_torch_shape, device,
                           print_output=not args['no_model_summary'])

        if flags['use_tensorboard']:
            tensorboard = create_tensorboard_writers(args, paths)
            if args['tensorboard_save_graph']:
                save_graph_to_tensorboard(film_model, tensorboard, input_image_torch_shape)

    # Set back the random state
    set_random_state(initial_random_state)      # FIXME : This is redundant, we already reset random state after initialising the model. Actions after model initialization doesn't seem to affect random state
    return (
        (task, args, flags, paths, device),
        dataloaders,
        (film_model_config, film_model, optimizer, loss_criterion, scheduler, tensorboard)
    )


def main(args):
    args = vars(args)  # Convert args object to dict
    task_and_more, dataloaders, model_and_more = prepare_for_task(args)
    task, args, flags, paths, device = task_and_more
    film_model_config, film_model, optimizer, loss_criterion, scheduler, tensorboard = model_and_more

    ####################################
    #   Task Execution
    ####################################
    if flags['create_output_folder']:
        # We create the symlink here so that bug in initialisation won't create a new 'latest' folder
        create_symlink_to_latest_folder(paths["output_experiment_folder"], paths["current_datetime_str"])


    execute_task(task, args, paths["output_dated_folder"], dataloaders, film_model, film_model_config, device,optimizer,
                 loss_criterion, scheduler, tensorboard)


    ####################################
    #   Exit
    ####################################
    on_exit_action(args, flags, paths, tensorboard)


def on_exit_action(args, flags, paths, tensorboard):
    if flags['use_tensorboard']:
        close_tensorboard_writers(tensorboard['writers'])

    if args['continue_training']:
        fix_best_epoch_symlink_if_necessary(paths['output_dated_folder'], args['film_model_weight_path'])

    time_elapsed = str(datetime.now() - paths["current_datetime"])

    print("Execution took %s\n" % time_elapsed)

    if flags['create_output_folder']:
        save_json({'time_elapsed': time_elapsed}, paths["output_dated_folder"], filename='timing.json')


def parse_args_string(string):
    return vars(parser.parse_args(string.strip().split(' ')))


# FIXME : Args is a namespace so it is available everywhere. Not a great idea to shadow it (But we get a dict key error if we try to access it so it is easiy catchable
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
