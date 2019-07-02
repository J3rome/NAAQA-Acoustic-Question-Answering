import random
import os
import numpy as np
import tensorflow as tf
import subprocess
import ujson


def get_config(config_path):
    with open(config_path) as f:
        return ujson.load(f)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)


def create_folder_if_necessary(folder_path, overwrite_folder=False):
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    elif overwrite_folder:
        os.rmdir(folder_path)
        os.mkdir(folder_path)


def create_symlink_to_latest_folder(experiment_folder, dated_folder_name, symlink_name='latest'):
    symlink_path = "%s/%s" % (experiment_folder, symlink_name)
    if os.path.isdir(symlink_path) or (os.path.exists(symlink_path) and not os.path.exists(os.readlink(symlink_path))):
        # Remove the previous symlink before creating a new one (We readlink to recover in case of broken symlink)
        os.remove(symlink_path)

    subprocess.run('cd %s && ln -s %s %s' % (experiment_folder, dated_folder_name, symlink_name), shell=True)


def save_training_stats(stats_output_file, epoch_nb, train_accuracy, train_loss, val_accuracy, val_loss):
    """
    Will read the stats file from disk and append new epoch stats (Will create the file if not present)
    """
    if os.path.isfile(stats_output_file):
        with open(stats_output_file, 'r') as f:
            stats = ujson.load(f)
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
        ujson.dump(stats, f, indent=2, sort_keys=True)


def save_inference_results(results, output_folder, filename="results.json"):
    with open("%s/%s" % (output_folder, filename), 'w') as f:
        ujson.dump(results, f, indent=2)


def is_tensor_optimizer(x):
    return hasattr(x, 'op_def')


def is_tensor_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0


def is_tensor_prediction(x):
    return isinstance(x, tf.Tensor) and 'predicted_answer' in x.name
