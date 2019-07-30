import random
import os
import numpy as np
import tensorflow as tf
import subprocess
import ujson
from collections import defaultdict
from dateutil.parser import parse as date_parse

import torch


def get_config(config_path):
    with open(config_path) as f:
        return ujson.load(f)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


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


def process_predictions(dataset, predictions, ground_truths, questions_id, scenes_id):
    processed_predictions = []
    for prediction, ground_truth, question_id, scene_id in zip(predictions, ground_truths, questions_id, scenes_id):
        decoded_prediction = dataset.tokenizer.decode_answer(prediction)
        decoded_ground_truth = dataset.tokenizer.decode_answer(ground_truth)
        prediction_answer_family = dataset.answer_to_family[decoded_prediction]
        ground_truth_answer_family = dataset.answer_to_family[decoded_ground_truth]
        processed_predictions.append({
            'question_id': question_id,
            'scene_id': scene_id,
            'correct': bool(prediction == ground_truth),
            'correct_answer_family': bool(prediction_answer_family == ground_truth_answer_family),
            'prediction': decoded_prediction,
            'ground_truth': decoded_ground_truth,
            'prediction_answer_family': prediction_answer_family,
            'ground_truth_answer_family': ground_truth_answer_family
        })

    return processed_predictions


def process_gamma_beta(processed_predictions, gamma_vectors_per_resblock, beta_vectors_per_resblock):
    processed_gamma_beta_vectors = []

    gamma_vectors_per_resblock = [v.tolist() for v in gamma_vectors_per_resblock]
    beta_vectors_per_resblock = [v.tolist() for v in beta_vectors_per_resblock]

    for result_index, processed_prediction in enumerate(processed_predictions):
        question_index = processed_prediction['question_id']
        processed_gamma_beta_vector = defaultdict(lambda : {})
        for resblock_index, gamma_vectors, beta_vectors in zip(range(len(gamma_vectors_per_resblock)), gamma_vectors_per_resblock, beta_vectors_per_resblock):

            processed_gamma_beta_vector['question_index'] = question_index
            processed_gamma_beta_vector['resblock_%d' % resblock_index]['gamma_vector'] = gamma_vectors[result_index]
            processed_gamma_beta_vector['resblock_%d' % resblock_index]['beta_vector'] = beta_vectors[result_index]

            # TODO : Add more attributes ? Could simply be cross loaded via question.json

        processed_gamma_beta_vectors.append(processed_gamma_beta_vector)

    return processed_gamma_beta_vectors


def sort_stats(stats, reverse=False):
    return sorted(stats, key=lambda e: float(e['val_acc']), reverse=reverse)


def save_training_stats(stats_output_file, epoch_nb, train_accuracy, train_loss, val_accuracy,
                        val_loss, epoch_train_time):
    """
    Will read the stats file from disk and append new epoch stats (Will create the file if not present)
    """
    if os.path.isfile(stats_output_file):
        with open(stats_output_file, 'r') as f:
            stats = ujson.load(f)
    else:
        stats = []

    stats.append({
        'epoch': "Epoch_%.2d" % epoch_nb,
        'train_time': str(epoch_train_time),
        'train_acc': '%.5f' % train_accuracy,
        'train_loss': '%.5f' % train_loss,
        'val_acc': '%.5f' % val_accuracy,
        'val_loss': '%.5f' % val_loss
    })

    stats = sort_stats(stats)

    with open(stats_output_file, 'w') as f:
        ujson.dump(stats, f, indent=2, sort_keys=True)

    return stats


def save_json(results, output_folder, filename, indented=True):
    with open("%s/%s" % (output_folder, filename), 'w') as f:
        ujson.dump(results, f, indent=2 if indented else None, escape_forward_slashes=False)


def read_json(folder, filename):
    with open('%s/%s' % (folder, filename), 'r') as f:
        return ujson.load(f)


def is_date_string(string):
    try:
        date_parse(string, fuzzy=True)  # Fuzzy parsing ignore unknown tokens
        return True
    except ValueError:
        return False


def is_tensor_optimizer(x):
    return hasattr(x, 'op_def')


def is_tensor_scalar(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.float32 and len(x.shape) == 0


def is_tensor_prediction(x):
    return isinstance(x, tf.Tensor) and 'predicted_answer' in x.name


def is_tensor_gamma_list(x):
    return isinstance(x, list) and isinstance(x[0], tf.Tensor) and 'gamma' in x[0].name


def is_tensor_beta_list(x):
    return isinstance(x, list) and isinstance(x[0], tf.Tensor) and 'beta' in x[0].name


def is_tensor_summary(x):
    return isinstance(x, tf.Tensor) and x.dtype is tf.string
