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
