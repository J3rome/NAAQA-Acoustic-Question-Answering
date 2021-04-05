import re

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import heapq
import numpy as np
from IPython.core.display import display

from utils.file import read_json
from utils.notebook.generic import format_epoch_folder
from utils.notebook.plot import plot_discrete_hist, plot_hist, plot_2d_matrix
from utils.notebook.pandas import color_by_multi_attribute, groupby_mean


# Results stats helpers

def filter_outliers(df, groupby_columns, outlier_col='nb_epoch_trained'):
    # hacky code...
    # Return a dataframe mask where each df['nb_epoch_trained'] > mean(df['nb_epoch_trained')] - std(df['nb_epoch_trained'])
    columns_of_interest = [*groupby_columns, outlier_col]

    if not isinstance(groupby_columns, list):
        groupby_columns = [groupby_columns]

    nb_groupby_cols = len(groupby_columns)

    grouped = df[columns_of_interest].groupby(groupby_columns).agg({outlier_col: lambda x: int(np.mean(x) - np.std(x))})

    new_filters = None
    for grouped_cols, row in grouped.iterrows():
        if nb_groupby_cols == 1:
            grouped_cols = (grouped_cols,)

        new_filter = (df[outlier_col] > row[outlier_col])

        for col_name, col_val in zip(groupby_columns, grouped_cols):
            new_filter &= (df[col_name] == col_val)

        if new_filters is None:
            new_filters = new_filter
        else:
            new_filters |= new_filter

    return df[new_filters]


def keep_x_best(filtered_df, groupby_columns, nb_to_keep, discriminative_attribute='test_acc'):
    return filtered_df.sort_values([*groupby_columns, discriminative_attribute], ascending=False).groupby(
        groupby_columns, as_index=False, dropna=False).apply(lambda x: x.iloc[:nb_to_keep])


def print_missing_seeds(df, groupby_cols, all_seeds):
    if not isinstance(all_seeds, set):
        all_seeds = set(all_seeds)

    def print_by_group(x):
        missing_seeds = all_seeds - set(x['random_seed'])
        if len(missing_seeds) > 0:
            print(x.name, "  Missing : ", missing_seeds)

    df.sort_values(groupby_cols, ascending=False).groupby(groupby_cols).apply(print_by_group)


def show_table(df, filters, groupby_columns, acc_columns, extra_columns=None, format_dict=None, attribute_by_color=None,
               mean_std_col=False, display_all=False, hardcoded_cols=None,
               show_count_col=False, inplace_std=False, remove_outliers=False, print_latex=True, nb_to_keep=None,
               all_seeds=None):
    exp = df[filters]

    if extra_columns is None:
        extra_columns = []

    if attribute_by_color is None:
        attribute_by_color = {}

    if all_seeds is not None:
        print_missing_seeds(exp, groupby_columns, all_seeds)

    # Drop duplicates (Same experiment ran multiple time)
    exp = exp.sort_values('date', ascending=False).drop_duplicates(
        [*groupby_columns, 'random_seed', 'input_type', 'config'], keep='first')

    if remove_outliers:
        exp = filter_outliers(exp, groupby_columns)

    if nb_to_keep is not None:
        exp = keep_x_best(exp, groupby_columns, nb_to_keep, 'test_acc')

    # Grouping - Mean & Std calc
    acc_std_columns = [f"{c}_std" for c in acc_columns]
    columns = groupby_columns + acc_columns + extra_columns
    columns_to_show = [*groupby_columns, *acc_columns, *extra_columns]

    # All experiments
    if display_all:
        all_exp = exp.sort_values([*groupby_columns, 'random_seed'], ascending=False)[columns_to_show]
        display(color_by_multi_attribute(all_exp,
                                         main_attribute='test_acc',
                                         attributes=groupby_columns,
                                         format_dict=format_dict))

    exp_grouped = groupby_mean(exp, groupby_columns, acc_columns, columns, add_count_col=show_count_col,
                               add_std_str=True, inplace_std_str=inplace_std).sort_values('test_acc',
                                                                                          ascending=False).reset_index(
        drop=True)

    if hardcoded_cols is not None:
        for col_name, col_conf in hardcoded_cols.items():

            if col_conf['type'] == "replace_groupby":
                columns_to_show = [c for c in columns_to_show if c not in groupby_columns]

            exp_grouped[col_name] = col_conf['values']
            columns_to_show = [col_name, *columns_to_show]

    if inplace_std:
        mean_std_col = False
        columns_to_show = [c if c not in acc_columns else f'{c}_std' for c in columns_to_show]
        attribute_by_color = {(name if name not in acc_columns else f"{name}_std"): color for name, color in
                              attribute_by_color.items()}

    if mean_std_col:
        exp_grouped['mean_std'] = exp_grouped[acc_std_columns].mean(axis=1)

        if len(extra_columns) == 0:
            up_to_extra_col = columns_to_show
        else:
            up_to_extra_col = columns_to_show[:columns_to_show.index(extra_columns[0])]

        columns_to_show = [*up_to_extra_col, 'mean_std', *extra_columns]

    if show_count_col:
        columns_to_show.append('count')

    exp_grouped = exp_grouped[columns_to_show]

    main_attribute_to_color = 'test_acc' if not inplace_std else 'test_acc_std'

    # Color display
    display(color_by_multi_attribute(exp_grouped, main_attribute=main_attribute_to_color,
                                     attributes=list(attribute_by_color.keys()),
                                     cmaps=['Blues', *list([v for v in attribute_by_color.values() if v is not None])],
                                     format_dict=format_dict))

    # Latex code
    if print_latex:
        latex = exp_grouped[columns_to_show].to_latex(index=False, formatters=format_dict, escape=False).replace(
            "\\textasciitilde", "$\\approx$").replace(" ± ", " ±")
        latex = re.sub(r'&\s*', '& ', latex)

        print("\n", latex)

    return exp_grouped



def load_experiment_predictions(experiment_output_path, epoch_folder='best', set_type='val', reduced_text=False):

    if set_type != 'test':
        epoch_folder = format_epoch_folder(epoch_folder)

        epoch_path = f"{experiment_output_path}/{epoch_folder}"
    else:
        epoch_path = experiment_output_path

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


def plot_acc_loss_by_epoch(epoch_stats_per_set, show_fig=False, fig_ax=None):

    if fig_ax:
        fig, axs = fig_ax
    else:
        fig, axs = plt.subplots(2, 1)

    axs[0].set_title("Accuracy by Epoch")
    axs[1].set_title("Loss by Epoch")

    for set_type, epoch_stats in epoch_stats_per_set.items():
        axs[0].plot([s['acc'] for s in epoch_stats], label=f"{set_type.capitalize()} Accuracy")
        axs[1].plot([s['loss'] for s in epoch_stats], label=f"{set_type.capitalize()} Loss")

    axs[0].legend()
    axs[1].legend()

    fig.tight_layout()

    if show_fig:
        fig.show()


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


def plot_predictions_confidence_gap(train_predictions, val_predictions, question_family=None, norm_hist=False,
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

    #FIXME : Sooo ugly !
    for pred in train_predictions['correct']:
        top_pred = heapq.nlargest(2, pred['prediction_probs'])
        pred['gap'] = top_pred[0] - top_pred[1]

    for pred in val_predictions['correct']:
        top_pred = heapq.nlargest(2, pred['prediction_probs'])
        pred['gap'] = top_pred[0] - top_pred[1]

    axs[0].set_title(f"[{prefix}]Confidence GAP (Guess 0 & Guess 1)\nCorrect Predictions")
    plot_hist(train_predictions['correct'], key="gap", label="Train", filter_fct=filter_fct, norm_hist=norm_hist,
              fig_ax=(fig, axs[0]))
    plot_hist(val_predictions['correct'], key="gap", label="Val", filter_fct=filter_fct, norm_hist=norm_hist,
              fig_ax=(fig, axs[0]))
    axs[0].legend()

    for pred in train_predictions['correct_family']:
        top_pred = heapq.nlargest(2, pred['prediction_probs'])
        pred['gap'] = top_pred[0] - top_pred[1]

    for pred in val_predictions['correct_family']:
        top_pred = heapq.nlargest(2, pred['prediction_probs'])
        pred['gap'] = top_pred[0] - top_pred[1]

    axs[1].set_title(f"[{prefix}]Confidence GAP (Guess 0 & Guess 1)\nIncorrect Predictions -- Correct Family")
    plot_hist(train_predictions['correct_family'], key="gap", label="Train", filter_fct=filter_fct,
              norm_hist=norm_hist, fig_ax=(fig, axs[1]))
    plot_hist(val_predictions['correct_family'], key="gap", label="Val", filter_fct=filter_fct, norm_hist=norm_hist,
              fig_ax=(fig, axs[1]))
    axs[1].legend()

    for pred in train_predictions['incorrect_family']:
        top_pred = heapq.nlargest(2, pred['prediction_probs'])
        pred['gap'] = top_pred[0] - top_pred[1]

    for pred in val_predictions['incorrect_family']:
        top_pred = heapq.nlargest(2, pred['prediction_probs'])
        pred['gap'] = top_pred[0] - top_pred[1]

    axs[2].set_title(f"[{prefix}]Confidence GAP (Guess 0 & Guess 1)\nIncorrect Predictions -- Incorrect Family")
    plot_hist(train_predictions['incorrect_family'], key="gap", label="Train", filter_fct=filter_fct,
              norm_hist=norm_hist, fig_ax=(fig, axs[2]))
    plot_hist(val_predictions['incorrect_family'], key="gap", label="Val", filter_fct=filter_fct,
              norm_hist=norm_hist, fig_ax=(fig, axs[2]))
    axs[2].legend()

    if show_fig:
        plt.show()


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

