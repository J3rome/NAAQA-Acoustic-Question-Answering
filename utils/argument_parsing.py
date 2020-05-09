from datetime import datetime
from utils.generic import is_date_string
import os


# Argument handling
def get_args_task_flags_paths(args):
    validate_arguments(args)
    task = get_task_from_args(args)
    paths = get_paths_from_args(task, args)
    flags = create_flags_from_args(task, args)
    update_arguments(args, paths, flags)

    return args, task, flags, paths


def validate_arguments(args):
    mutually_exclusive_params = [args['training'], args['inference'], args['feature_extract'], args['create_dict'],
                                 args['visualize_gamma_beta'], args['visualize_grad_cam'], args['lr_finder'],
                                 args['calc_clear_mean'], args['random_answer_baseline'],
                                 args['random_weight_baseline'], args['prepare_images'], args['notebook_data_analysis'],
                                 args['notebook_model_inference'], args['tf_weight_transfer']]

    assert sum(mutually_exclusive_params) == 1, \
        "[ERROR] Can only do one task at a time " \
        "(--training, --inference, --visualize_gamma_beta, --create_dict, --feature_extract --visualize_grad_cam " \
        "--prepare_images, --lr_finder, --calc_clear_mean, --random_answer_baseline, " \
        "--random_weight_baseline, --notebook_data_analysis, --notebook_model_inference)"

    assert not args['continue_training'] or (args['training'] and args['continue_training']), \
        "[ERROR] Must be in --training mode for --continue_training"

    assert sum([args['pad_to_largest_image'], args['pad_per_batch']]) <= 1, \
        '--pad_to_largest_image and --pad_per_batch can\'t be used together'


def create_flags_from_args(task, args):
    flags = {}

    flags['restore_model_weights'] = args['continue_training'] or args['film_model_weight_path'] is not None \
                                     or task in ['inference', 'visualize_grad_cam', 'notebook_model_inference']
    flags['use_tensorboard'] = 'train' in task
    flags['create_loss_criterion'] = task in ['training', 'lr_finder', 'inference']
    flags['create_optimizer'] = task in ['training', 'lr_finder']
    flags['force_sgd_optimizer'] = task == 'lr_finder' or args['cyclical_lr']
    flags['load_dataset_extra_stats'] = task.startswith('notebook')
    flags['create_output_folder'] = task not in ['create_dict', 'feature_extract',
                                                 'write_clear_mean_to_config'] and not task.startswith('notebook')
    flags['instantiate_model'] = task in ['training',
                                          'inference',
                                          'visualize_grad_cam',
                                          'lr_finder',
                                          'random_weight_baseline',
                                          'notebook_model_inference',
                                          'tf_weight_transfer']

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
    tasks = ['training', 'inference', 'visualize_gamma_beta', 'visualize_grad_cam', 'feature_extract', 'prepare_images',
             'create_dict', 'lr_finder', 'calc_clear_mean', 'random_weight_baseline',
             'random_answer_baseline', 'notebook_data_analysis', 'notebook_model_inference', 'tf_weight_transfer']

    for task in tasks:
        if task in args and args[task]:
            return task

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

    args['normalize_zero_one'] = args['normalize_zero_one'] and not args['keep_image_range']

    # Make sure we are not normalizing beforce calculating mean and std
    if args['calc_clear_mean']:
        args['normalize_with_imagenet_stats'] = False
        args['normalize_with_clear_stats'] = False

    args['dict_folder'] = args['preprocessed_folder_name'] if args['dict_folder'] is None else args['dict_folder']
    if args['dict_file_path'] is None:
        args['dict_file_path'] = "%s/%s/dict.json" % (paths["data_path"], args['dict_folder'])

    if flags['restore_model_weights']:
        if args['continue_training'] and args['film_model_weight_path'] is None:
            # Use latest by default when continuing training
            args['film_model_weight_path'] = 'latest'

        assert args['film_model_weight_path'] is not None, 'Must provide path to model weights to ' \
                                                           'do inference or to continue training.'

        # If path specified is a date, we construct the path to the best model weights for the specified run
        base_path = f"{args['output_root_path']}/training/{paths['output_name']}/{args['film_model_weight_path']}"
        # Note : We might redo some epoch when continuing training because the 'best' epoch is not necessarily the last
        suffix = "best/model.pt.tar"

        if is_date_string(args['film_model_weight_path']):
            args['film_model_weight_path'] = "%s/%s" % (base_path, suffix)
        elif args['film_model_weight_path'] == 'latest':
            # The 'latest' symlink will be overriden by this run (If continuing training).
            # Use real path of latest experiment
            symlink_value = os.readlink(base_path)
            clean_base_path = base_path[:-(len(args['film_model_weight_path']) + 1)]
            args['film_model_weight_path'] = '%s/%s/%s' % (clean_base_path, symlink_value, suffix)

    # By default the start_epoch should is 0. Will only be modified if loading from checkpoint
    args["start_epoch"] = 0


def get_feature_extractor_config_from_args(args):
    if args['no_feature_extractor']:
        return None
    else:
        return {'version': 101, 'layer_index': args['feature_extractor_layer_index']}  # Idx 6 -> Block3/unit22