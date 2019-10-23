from utils import read_json
import matplotlib.pyplot as plt
import numpy as np

from utils import get_answer_to_family_map

# TODO : Retrieve scenes ? Questions ?
# TODO : If adding more attributes (Like presence of relation), should add them in the processing instead of here. We should mainly read and arrange stuff arond, not post process


# Misc
def format_epoch_folder(epoch_folder):
    if type(epoch_folder) == int or epoch_folder.isdigit():
        epoch_folder = f"Epoch_{epoch_folder:02d}"

    assert epoch_folder.startswith('Epoch') or epoch_folder == 'best', "Invalid epoch folder provided"

    return epoch_folder


# Plotting
def show_discrete_hist_centered(data, title=None, sort_key_fct=None, show_fig=False, export_filepath=None, fig_ax=None):
    show_fig = show_fig and export_filepath is None
    labels, counts = np.unique(data, return_counts=True)

    if sort_key_fct:
        labels_counts = sorted(zip(labels, counts), key=sort_key_fct)
        labels, counts = zip(*labels_counts)

    #plt.bar(labels, counts, align='center')
    #plt.gca().set_xticks(labels)
    #plt.xticks(rotation=90)
    #plt.show()

    #return 0,0

    if fig_ax:
        # Add plot to provided figure
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

    ax.bar(labels, counts, align='center')
    #ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    #ax.xticks(rotation=90)

    if title:
        ax.set_title(title)

    if show_fig:
        plt.show()

    if export_filepath:
        assert False, "Export to file not implemented."

    return fig, ax


def plot_hist(predictions, key, filter_fct=None, title=None, label=None, norm_hist=False, show_fig=False, fig_ax=None):
    preds = predictions

    if filter_fct:
        preds = filter(filter_fct, preds)

    to_plot = [p[key] for p in preds]

    if fig_ax:
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

    if label and label.lower() == "same":
        label = title

    if norm_hist:
        length = len(to_plot)
        weights = np.ones(length)/length
    else:
        weights = None

    ax.hist(to_plot, label=label, weights=weights)

    if ax.get_title() == "":
        ax.set_title(title)

    fig.tight_layout()
    if show_fig:
        plt.show()


# Results stats helpers
def load_experiment_predictions(experiment_output_path, epoch_folder='best', set_type='val'):
    epoch_folder = format_epoch_folder(epoch_folder)

    epoch_path = f"{experiment_output_path}/{epoch_folder}"
    prediction_filename = f"{set_type}_predictions.json"

    predictions = read_json(epoch_path, prediction_filename)

    return predictions


def load_experiment_stats(experiment_output_path, set_type='val', sorted_by_time=True, only_epoch=None):
    full_stats = read_json(experiment_output_path, 'stats.json')

    if only_epoch:
        if only_epoch == 'best':
            stats = [full_stats[0]]
        else:
            epoch_folder = format_epoch_folder(only_epoch)

            stats = [s for s in full_stats if s['epoch'] == epoch_folder]
    else:
        stats = full_stats
        if sorted_by_time:
            stats = sorted(stats, key=lambda s: int(s['epoch'].split('_')[1]))

    stats = [{'epoch': s['epoch'],
              'acc': s[f'{set_type}_acc'],
              'loss': s[f'{set_type}_loss']
              } for s in stats]

    return stats


def sort_correct_incorrect_predictions(predictions):
    correct_predictions = []
    correct_family_predictions = []
    incorrect_family_predictions = []

    for prediction in predictions:
        if prediction['correct']:
            correct_predictions.append(prediction)
        elif prediction['correct_answer_family']:
            correct_family_predictions.append(prediction)
        else:
            incorrect_family_predictions.append(prediction)

    return {
        'correct': correct_predictions,
        'correct_family': correct_family_predictions,
        'incorrect_family': incorrect_family_predictions
    }


def plot_confidence(train_predictions, val_predictions, question_family=None, norm_hist=False,
                    show_fig=False, fig_ax=None):

    if fig_ax:
        fig, axs = fig_ax
        assert len(axs) >= 3, 'Subplot provided doesn\'t have 3 ax available'
    else:
        fig, axs = plt.subplots(3, 1)

    if question_family:
        filter_fct = lambda p: p['ground_truth_answer_family'] == question_family
        prefix = question_family.capitalize()
    else:
        filter_fct = None
        prefix = "All Family"

    axs[0].set_title(f"[{prefix}]Confidence in Correct Predictions")
    plot_hist(train_predictions['correct'], key="confidence", label="Train", filter_fct=filter_fct, norm_hist=norm_hist,
              fig_ax=(fig, axs[0]))
    plot_hist(val_predictions['correct'], key="confidence", label="Val", filter_fct=filter_fct, norm_hist=norm_hist,
              fig_ax=(fig, axs[0]))
    axs[0].legend()

    axs[1].set_title(f"[{prefix}]Confidence in Incorrect Predictions -- Correct Family")
    plot_hist(train_predictions['correct_family'], key="confidence", label="Train", filter_fct=filter_fct,
              norm_hist=norm_hist, fig_ax=(fig, axs[1]))
    plot_hist(val_predictions['correct_family'], key="confidence", label="Val", filter_fct=filter_fct, norm_hist=norm_hist,
              fig_ax=(fig, axs[1]))
    axs[1].legend()

    axs[2].set_title(f"[{prefix}]Confidence in Incorrect Predictions -- Incorrect Family")
    plot_hist(train_predictions['incorrect_family'], key="confidence", label="Train", filter_fct=filter_fct,
              norm_hist=norm_hist, fig_ax=(fig, axs[2]))
    plot_hist(val_predictions['incorrect_family'], key="confidence", label="Val", filter_fct=filter_fct,
              norm_hist=norm_hist, fig_ax=(fig, axs[2]))
    axs[2].legend()

    if show_fig:
        plt.show()

# Dataset stats helpers


if __name__ == "__main__":
    root_data_path = "data"
    root_output_path = "output/train_film"
    experiment_name = "v3_resnet_1k_5_inst_1024_win_50_overlap_resnet_4"
    experiment_output_path = f"{root_output_path}/{experiment_name}"
    data_name = "v3_resnet_1k_5_inst_1024_win_50_overlap"
    data_path = f"{root_data_path}/{data_name}"        # FIXME : This won't work because of the suffix in the output
    epoch_id = 25

    answer_to_family_map = get_answer_to_family_map(f'{data_path}/attributes.json', to_lowercase=False)

    train_predictions = load_experiment_predictions(experiment_output_path, epoch_id, set_type='train')
    val_predictions = load_experiment_predictions(experiment_output_path, epoch_id, set_type='val')

    correct_train_predictions = [p for p in train_predictions if p['correct']]
    incorrect_family_train_predictions = [p for p in train_predictions if not p['correct'] and not p['correct_answer_family']]
    correct_family_train_predictions = [p for p in train_predictions if not p['correct'] and p['correct_answer_family']]

    correct_val_predictions = [p for p in val_predictions if p['correct']]
    incorrect_family_val_predictions = [p for p in val_predictions if not p['correct'] and not p['correct_answer_family']]
    correct_family_val_predictions = [p for p in val_predictions if not p['correct'] and p['correct_answer_family']]

    correct_family_val_hist = [(p['prediction'], p['prediction_answer_family'], p['confidence']) for p in correct_family_val_predictions]
    correct_family_train_hist = [(p['prediction'], p['prediction_answer_family'], p['confidence']) for p in correct_family_train_predictions]

    train_confidence = [v[2] for v in correct_family_train_hist]
    val_confidence = [v[2] for v in correct_family_val_hist]

    plt.hist(train_confidence)
    plt.figure()
    plt.hist(val_confidence)
    plt.show()

    preds = [p[0] for p in correct_family_val_hist]
    train_ground_truth = [p['ground_truth'] for p in train_predictions]
    val_ground_truth = [p['ground_truth'] for p in val_predictions]
    show_discrete_hist_centered(train_ground_truth, sort_key_fct=lambda x: (answer_to_family_map[x[0]], x[0]))
    show_discrete_hist_centered(val_ground_truth, sort_key_fct=lambda x: (answer_to_family_map[x[0]], x[0]))

    train_family_ground_truth = [p['ground_truth_answer_family'] for p in train_predictions]
    val_family_ground_truth = [p['ground_truth_answer_family'] for p in val_predictions]

    show_discrete_hist_centered(train_family_ground_truth, sort_key_fct=lambda x: x[0])
    show_discrete_hist_centered(val_family_ground_truth, sort_key_fct=lambda x: x[0])
