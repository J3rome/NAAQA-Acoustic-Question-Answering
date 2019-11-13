import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

from utils.file import read_json
from utils.notebook.generic import format_epoch_folder
from utils.notebook.plot import plot_discrete_hist, plot_hist, plot_2d_matrix


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


def plot_predictions_distribution_per_question_family(train_predictions, val_predictions, all_x_labels=None, norm_hist=False,
                                          show_fig=False):
    keys_titles = [('correct', 'Correct Answer'),
                   ('correct_family', 'Incorrect Answer -- Correct Family'),
                   ('incorrect_family', 'Incorrect Answer -- Incorrect Family')]

    figs_axs = []

    for key, title in keys_titles:
        fig, ax = plt.subplots(1, 1)
        ax.set_title(title)
        plot_discrete_hist(train_predictions[key], key="ground_truth_answer_family", all_x_labels=all_x_labels,
                           fig_ax=(fig, ax), position='left', legend_label="Train", norm_hist=norm_hist)
        plot_discrete_hist(val_predictions[key], key="ground_truth_answer_family", all_x_labels=all_x_labels,
                           fig_ax=(fig, ax), position='right', legend_label='Val', norm_hist=norm_hist)
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


def plot_confusion_matrix(predictions, ground_truth, normalize=False, title=None, show_fig=True,
                          colormap=plt.cm.Blues, add_annotations=True):

    # TODO : Sort labels
    conf_matrix = confusion_matrix(ground_truth, predictions)
    labels = unique_labels(predictions, ground_truth)

    if title is None:
        title = "Confusion Matrix"

    return plot_2d_matrix(conf_matrix, labels, title=title, normalize=normalize, show_fig=show_fig, colormap=colormap,
                          xaxis_name='predictions', yaxis_name="ground truth", add_annotations=add_annotations)

