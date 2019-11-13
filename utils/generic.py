import re
import os
import signal

import ujson

from utils.file import read_json
from utils.logging import close_tensorboard_writers


def is_date_string(string):
    return re.match(r'\d+-\d+-\d+_\d+h\d+', string) is not None


def sort_stats(stats, reverse=False):
    return sorted(stats, key=lambda e: float(e['val_loss']), reverse=reverse)


def sort_stats_by_time(stats, reverse=False):
    return sorted(stats, key=lambda s: int(s['epoch'].split('_')[1]), reverse=reverse)


def chain_load_experiment_stats(output_experiment_dated_folder, continue_training=False, film_model_weight_path=None,
                                stats_filename="stats.json", arguments_filename="arguments.json"):
    stats_file_path = f"{output_experiment_dated_folder}/{stats_filename}"
    if os.path.isfile(stats_file_path):
        stats = sort_stats_by_time(read_json(stats_file_path))
    else:
        stats = []

    if film_model_weight_path is None:
        # Read path to model weight in arguments json file
        arguments = read_json(output_experiment_dated_folder, arguments_filename)

        continue_training = arguments['continue_training']
        film_model_weight_path = arguments['film_model_weight_path']

    if continue_training:

        if film_model_weight_path == "latest":
            # Path for fallback with old behaviour
            model_path = read_json(output_experiment_dated_folder, 'restored_from.json')['restored_film_weight_path']
        else:
            model_path = film_model_weight_path

        continued_experiment_path = '/'.join(model_path.split('/')[:-2])

        continued_experiment_stats = chain_load_experiment_stats(continued_experiment_path,
                                                                 continue_training=continue_training,
                                                                 film_model_weight_path=None,
                                                                 stats_filename=stats_filename)

        if len(stats) > 0:
            start_epoch = int(stats[0]['epoch'].split("_")[1])
            continued_experiment_stats = [s for s in continued_experiment_stats if
                                          int(s['epoch'].split('_')[1]) < start_epoch]

        stats = continued_experiment_stats + stats

    return stats


def save_training_stats(stats_output_file, epoch_nb, train_accuracy, train_loss, val_accuracy,
                        val_loss, epoch_train_time):
    """
    Will read the stats file from disk and append new epoch stats (Will create the file if not present)
    """
    if os.path.isfile(stats_output_file):
        stats = read_json(stats_output_file)
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


def get_answer_to_family_map(attributes_filepath, to_lowercase=True, reduced_text=False):
    attributes = read_json(attributes_filepath)

    answer_to_family = {"<unk>": "unknown"}  # FIXME : Quantify what is the impact of having an unknown answer

    for family, answers in attributes.items():
        for answer in answers:
            the_answer = answer
            if reduced_text and 'of the scene' in the_answer:
                the_answer = str(answer.split(' ')[0][:3])

            if to_lowercase:
                the_answer = the_answer.lower()

            # If there is duplicated answers, they will be assigned to the first occurring family
            if the_answer not in answer_to_family:
                answer_to_family[the_answer] = family

    return answer_to_family


# FIXME: Not working, cause error with dataloader if using exit, called multiple time (Multiple threads receive the signal)
def set_sigint_handler(args, task, tensorboard_writers):

    def sigint_handler(signal, frame):
        print("Ctrl+C pressed.")
        print(f"'{task}' was running. Please verify the output folder and delete incomplete data")
        if tensorboard_writers:
            print("Closing tensorboard writers")
            close_tensorboard_writers(tensorboard_writers)

    signal.signal(signal.SIGINT, sigint_handler)
