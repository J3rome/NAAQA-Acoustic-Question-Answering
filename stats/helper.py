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


def autolabel_bar(ax, rects):
    for rect in rects:
        h = rect.get_height()
        h_str = f'{int(h)}' if h >= 1 or h == 0 else f'{float(h):.2f}'
        ax.text(rect.get_x() + rect.get_width()/2., 1.00 * h, h_str, ha='center', va='bottom', fontsize=6)


# Plotting
def show_discrete_hist(data, key=None, title=None, label=None, sort_key_fct=None, show_fig=False,
                                export_filepath=None, fig_ax=None, position='center', capitalize=True, bar_width=0.4,
                                norm_hist=False, all_x_labels=None):
    show_fig = show_fig and export_filepath is None

    if key:
        data = [d[key] for d in data]

    labels, counts = np.unique(data, return_counts=True)

    if all_x_labels:
        missing_labels = list(set(all_x_labels) - set(labels))

        labels = np.append(labels, missing_labels)
        counts = np.append(counts, np.zeros(len(missing_labels)))

    nb_labels = len(labels)

    if sort_key_fct is None:
        sort_key_fct = lambda x: x[0]

    labels_counts = sorted(zip(labels, counts), key=sort_key_fct)
    labels, counts = zip(*labels_counts)

    if norm_hist:
        total = sum(counts)
        counts = [count/total for count in counts]

    if capitalize:
        labels = [l.capitalize() for l in labels]

    if fig_ax:
        # Add plot to provided figure
        fig, ax = fig_ax
    else:
        fig, ax = plt.subplots()

    x = np.arange(nb_labels)
    original_x = x
    if position == 'left':
        x = x - (bar_width/2)
    elif position == 'right':
        x = x + (bar_width/2)

    rects = ax.bar(x, counts, width=bar_width, align='center', label=label)
    ax.set_xticks(original_x)
    ax.set_xticklabels(labels, rotation=90)

    autolabel_bar(ax, rects)

    if title:
        ax.set_title(title)

    #fig.tight_layout()

    if show_fig:
        plt.show()

    if export_filepath:
        assert False, "Export to file not implemented."

    return fig, ax


def plot_hist(predictions, key=None, filter_fct=None, title=None, label=None, norm_hist=False, show_fig=False,
              fig_ax=None):

    preds = predictions

    if filter_fct:
        preds = filter(filter_fct, preds)

    if key:
        to_plot = [p[key] for p in preds]
    else:
        to_plot = preds

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


def plot_distribution_per_question_family(train_predictions, val_predictions, all_x_labels=None, norm_hist=False,
                                          show_fig=False):
    keys_titles = [('correct', 'Correct Answer'),
                   ('correct_family', 'Incorrect Answer -- Correct Family'),
                   ('incorrect_family', 'Incorrect Answer -- Incorrect Family')]

    figs_axs = []

    for key, title in keys_titles:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        show_discrete_hist(train_predictions[key], key="ground_truth_answer_family", all_x_labels=all_x_labels,
                           fig_ax=(fig, ax), position='left', label="Train", norm_hist=norm_hist)
        show_discrete_hist(val_predictions[key], key="ground_truth_answer_family", all_x_labels=all_x_labels,
                           fig_ax=(fig, ax), position='right', label='Val', norm_hist=norm_hist)
        ax.legend()

        fig.tight_layout()

        figs_axs.append((fig, ax))

    if show_fig:
        plt.show()

    return figs_axs


def plot_predictions_confidence(train_predictions, val_predictions, question_family=None, norm_hist=False,
                    show_fig=False, fig_ax=None):
    # TODO : Add X & Y Axis labels
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


if __name__ == "__main__":
    print("Not meant to be runned as a standalone script")
