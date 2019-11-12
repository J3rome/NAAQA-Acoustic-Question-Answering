from datetime import datetime

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
    flags['create_output_folder'] = not args['create_dict'] and not args['feature_extract'] and not args[
        'write_clear_mean_to_config']
    flags['use_tensorboard'] = 'train' in task
    flags['create_loss_criterion'] = args['training'] or args['lr_finder']
    flags['create_optimizer'] = args['training'] or args['lr_finder']
    flags['force_sgd_optimizer'] = args['lr_finder'] or args['cyclical_lr']
    flags['instantiate_model'] = not args['create_dict'] and not args['write_clear_mean_to_config'] and \
                                 'gamma_beta' not in task and 'random_answer' not in task and not args['prepare_images']

    return flags


def get_paths_from_args(task, args):
    paths = {}

    paths["output_name"] = args['version_name'] + "_" + args['output_name_suffix'] if args['output_name_suffix'] else \
    args['version_name']
    paths["data_path"] = "%s/%s" % (args['data_root_path'], args['version_name'])
    paths["output_task_folder"] = "%s/%s" % (args['output_root_path'], task)
    paths["output_experiment_folder"] = "%s/%s" % (paths["output_task_folder"], paths["output_name"])
    paths["current_datetime"] = datetime.now()
    paths["current_datetime_str"] = paths["current_datetime"].strftime("%Y-%m-%d_%Hh%M")
    paths["output_dated_folder"] = "%s/%s" % (paths["output_experiment_folder"], paths["current_datetime_str"])

    return paths


def get_task_from_args(args):
    tasks = ['training', 'inference', 'visualize_gamma_beta', 'visualize_grad_cam', 'feature_extract', 'prepare_images',
             'create_dict', 'lr_finder', 'write_clear_mean_to_config', 'random_weight_baseline',
             'random_answer_baseline']

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


def get_feature_extractor_config_from_args(args):
    if args['no_feature_extractor']:
        return None
    else:
        return {'version': 101, 'layer_index': args['feature_extractor_layer_index']}  # Idx 6 -> Block3/unit22