from collections import defaultdict

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
def load_experiment_predictions(experiment_output_path, epoch_folder='best', set_type='val', reduced_text=False):
    epoch_folder = format_epoch_folder(epoch_folder)

    epoch_path = f"{experiment_output_path}/{epoch_folder}"
    prediction_filename = f"{set_type}_predictions.json"

    predictions = read_json(epoch_path, prediction_filename)

    if reduced_text:
        for prediction in predictions:
            for key in ['prediction', 'ground_truth']:
                if 'of the scene' in prediction[key]:
                    prediction[key] = str(prediction[key].split(' ')[0][:3])

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


def separate_preds_ground_truth_old(processed_predictions, filter_fct=None):

    if filter_fct:
        processed_predictions = filter(filter_fct, processed_predictions)

    predictions = []
    ground_truths = []

    for processed_prediction in processed_predictions:
        predictions.append(processed_prediction['prediction'])
        ground_truths.append(processed_prediction['ground_truth'])

    return predictions, ground_truths


def separate_preds_ground_truth(processed_predictions, attribute=None):

    predictions = defaultdict(list)
    ground_truths = defaultdict(list)

    for processed_prediction in processed_predictions:
        if attribute:
            value = processed_prediction[attribute]
            predictions[value].append(processed_prediction['prediction'])
            ground_truths[value].append(processed_prediction['ground_truth'])

        predictions['all'].append(processed_prediction['prediction'])
        ground_truths['all'].append(processed_prediction['ground_truth'])

    if attribute is None:
        predictions = predictions['all']
        ground_truths = ground_truths['all']

    return predictions, ground_truths



from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def plot_confusion_matrix(predictions, ground_truth, normalize=False, title=None, show_fig=True,
                          colormap=plt.cm.Blues, add_annotations=True):

    # TODO : Sort labels
    conf_matrix = confusion_matrix(ground_truth, predictions)
    labels = unique_labels(predictions, ground_truth)

    if normalize:
        conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

    if title is None:
        title = "Confusion Matrix"

    fig, ax = plt.subplots()
    im = ax.imshow(conf_matrix, interpolation='nearest', cmap=colormap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(conf_matrix.shape[1]), yticks=np.arange(conf_matrix.shape[0]), yticklabels=labels,
           title=title, ylabel="Ground Truth", xlabel="Predictions")
    ax.set_xticklabels(labels, rotation=90)
    ax.axis('image')

    if add_annotations:
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                if conf_matrix[i, j] > 0:
                    ax.text(j, i, format(conf_matrix[i, j], fmt),
                            ha="center", va="center",
                            color="white" if conf_matrix[i, j] > thresh else "black")

    fig.tight_layout()

    if show_fig:
        plt.show()

    return fig, ax


if __name__ == "__main__":
    print("Not meant to be runned as a standalone script")
