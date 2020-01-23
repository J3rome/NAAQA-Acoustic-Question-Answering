from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from utils.notebook.plot import plot_2d_matrix, plot_discrete_hist, plot_hist


# Scene analysis
def scene_object_per_position(scenes, attribute='instrument', max_scene_length=None, attribute_processing=None):
    if max_scene_length is None:
        # Assume fixed scenes. We could also rerieve the max scene length
        max_scene_length = len(scenes[0]['definition']['objects'])

    positions = defaultdict(lambda: [0] * max_scene_length)
    for scene in scenes:
        for i, obj in enumerate(scene['definition']['objects']):
            attribute_value = obj[attribute]
            if callable(attribute_processing):
                attribute_value = attribute_processing(attribute_value)
            positions[attribute_value][i] += 1

    return positions


def plot_attribute_per_position_matrix(attribute_per_position, attribute_name, title=None, show_fig=False,
                                       colormap=plt.cm.Blues, add_annotations=True, attribute_processing=None):
    """
        Column : Position
        Row : Attributes
    """
    # FIXME : Not sure the order will be the same that with items()
    attribute_values = list(attribute_per_position.keys())

    if callable(attribute_processing):
        attribute_values = [attribute_processing(attr) for attr in attribute_values]

    attribute_values = [str(val).capitalize() if val is not None else "None" for val in attribute_values]

    nb_attr = len(attribute_values)
    nb_pos = len(list(attribute_per_position.values())[0])
    matrix = np.zeros((nb_attr, nb_pos), dtype=np.int32)
    attr_idx = 0
    for attr, positions in attribute_per_position.items():
        matrix[attr_idx, :] = positions
        attr_idx += 1

    if title is None:
        title = f"[{attribute_name.capitalize()}]Scene objects per position"

    fig_ax = plot_2d_matrix(matrix, range(nb_pos), attribute_values, title=title, show_fig=show_fig, colormap=colormap,
                            xaxis_name="Position", yaxis_name=attribute_name.capitalize(),
                            add_annotations=add_annotations)

    return fig_ax


def plot_scene_distribution_per_attribute(scenes, attribute_name, title=None, legend_label=None,
                                          norm_hist=False, fig_ax=None, show_fig=False, attribute_processing=None):
    attribute_list = []
    for scene in scenes:
        for i, obj in enumerate(scene['definition']['objects']):
            value = obj[attribute_name]

            if callable(attribute_processing):
                value = attribute_processing(value)

            value = str(value) if value is not None else "None"
            attribute_list.append(value)

    if title is None:
        title = f"[{attribute_name.capitalize()}]Scene distribution"

    fig_ax = plot_discrete_hist(attribute_list, title=title,
                                fig_ax=fig_ax, legend_label=legend_label, norm_hist=norm_hist, show_fig=show_fig)

    fig_ax[0].tight_layout()

    return fig_ax


def plot_scene_duration_hist(scenes, title=None, legend_label=None, fig_ax=None, norm_hist=False, show_fig=False):
    durations = []

    for scene_info in scenes:
        scene = scene_info['definition']
        durations.append(
            scene['silence_before'] + sum(obj['duration'] + obj['silence_after'] for obj in scene['objects']))

        if title is None:
            title = "Scene durations"

    fig_ax = plot_hist(durations, title=title, fig_ax=fig_ax, label=legend_label, norm_hist=norm_hist,
                                show_fig=show_fig)

    fig_ax[0].tight_layout()

    return fig_ax


def plot_scene_total_silence_distribution(scenes, title=None, legend_label=None, fig_ax=None, norm_hist=False,
                                          show_fig=False):
    silences = []

    for scene_info in scenes:
        scene = scene_info['definition']
        silences.append(scene['silence_before'] + sum(obj['silence_after'] for obj in scene['objects']))

    if title is None:
        title = "Total silence distribution"

    fig_ax = plot_hist(silences, title=title, fig_ax=fig_ax, label=legend_label, norm_hist=norm_hist, show_fig=show_fig)

    fig_ax[0].tight_layout()

    return fig_ax


def plot_scene_silence_by_position_distribution(scenes, title=None, legend_label=None, fig_ax=None, norm_hist=False,
                                                show_fig=False):
    silences_by_position = defaultdict(list)

    for scene_info in scenes:
        scene = scene_info['definition']
        nb_obj = len(scene['objects'])
        silences_by_position[0].append(scene['silence_before'])

        for i in range(nb_obj):
            silences_by_position[i + 1].append(scene['objects'][i]['silence_after'])

    if title is None:
        title = "Silence by position"

    for i, (pos, values) in enumerate(silences_by_position.items()):
        new_fig_ax = plot_hist(values, title=f"[Position {pos}]{title}", fig_ax=fig_ax, label=legend_label,
                               norm_hist=norm_hist, show_fig=show_fig)

        new_fig_ax[0].tight_layout()

    # FIXME : We are returning the original fig_ax.. not all of them
    return fig_ax
