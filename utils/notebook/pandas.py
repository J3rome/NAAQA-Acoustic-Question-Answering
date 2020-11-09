import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from pandas.api.types import is_numeric_dtype


def grouped_scatter(dataframe, group_key, x_axis, y_axis, title=None, colormap=cm.viridis, ax=None, label_modifier=None,
                    show_label=True, colorlist=None, additional_params=None):
    group_unique_keys = dataframe[group_key].unique()

    if colorlist is None:
        colorlist = {key: colors.rgb2hex(colormap(i)) for key, i in
                     zip(group_unique_keys, np.linspace(0, 0.9, len(group_unique_keys)))}

    if ax is None:
        _, ax = plt.subplots()

    if additional_params is None:
        additional_params = {}

    for key, group in dataframe.groupby(group_key):
        if show_label:
            label = key
            if label_modifier:
                label = label_modifier(label)
        else:
            label = "_hidden"

        group.plot.scatter(ax=ax, x=x_axis, y=y_axis, c=colorlist[key], label=label, title=title, **additional_params)
        #group.plot(ax=ax, x=x_axis, y=y_axis, c=colorlist[key], label="_hidden")  # , label=key, title=title)


def sub_cols_with_cond_and_create_new_col(df, new_col_name, col_to_sub, cond1, cond2, output_cond):
    temp = df[cond1][col_to_sub].reset_index(drop=1) - df[cond2][col_to_sub].reset_index(drop=1)

    temp.index = df[output_cond].index
    df.loc[output_cond, new_col_name] = temp

    return df


def color_row_by_attribute(sample, attribute, colors_by_config):
    css = f"background-color: {colors_by_config[sample[attribute]]}"
    return [css] * len(sample.index)


def groupby_mean(df, groupby_columns, mean_columns, selected_columns, add_count_col=True):
    if type(groupby_columns) == str:
        groupby_columns = [groupby_columns]

    agg_cols = {name: 'first' for name in selected_columns if name not in groupby_columns}

    # Mean only for certain columns
    for col in mean_columns:
        agg_cols[col] = 'mean'

    grouped_df = df.groupby(groupby_columns, as_index=False)

    aggregated = grouped_df.agg(agg_cols)

    if add_count_col:
        aggregated['count'] = grouped_df.size().values
        additional_cols = ['count']
    else:
        additional_cols = []

    # We re-select the columns to set all columns in the desired order (Otherwise the grouped columns come first)
    return aggregated[selected_columns + additional_cols]


def color_by_multi_attribute(df, main_attribute, attributes=None, cmaps=None, format_dict=None, print_infos=True):
    """
    Will color the whole rows based on the main_attribute values.
    Will color specific columns based on attributes values (Over the main_attribute color)
    Data should be sorted in the desired order before passing to this function
    This will return a Styler object.
    """

    # Color map handling
    if cmaps is None:
        cmaps = ['Blues']
    elif type(cmaps) != list:
        cmaps = [cmaps]

    for i, cmap in enumerate(cmaps):
        if type(cmap) == str:
            cmaps[i] = plt.get_cmap(cmap)

    main_attribute_cmap = cmaps[0]
    other_attributes_cmaps = cmaps[1:]

    # Styler fct for both column & row styling
    def styler_fct(sample, target_attribute, min_max, categorical_values=None, cmap=None, on_column=False):
        # This is super ugly.. but got both the column and row styler together..

        if on_column:
            if sample.name not in target_attribute:
                return [""] * len(sample.index)
            else:
                values = sample
                categorical_values = categorical_values[sample.name]
                cmap = cmap[sample.name]
                min_max = min_max[sample.name]
        else:
            values = [sample[target_attribute]]

        to_return = []
        for val in values:
            if pd.isnull(val):
                val = 0

            if categorical_values:
                val = categorical_values[val]

            # Will generate a value in the range of 20,220
            val = int(((val - min_max[0]) / (min_max[1] - min_max[0])) * 200 + 20)

            color = tuple([int(c * 255) for i, c in enumerate(cmap(val)) if i < 3])
            css = f"background-color: rgb{color}; color: {text_color_from_rgb(color)};"
            to_return.append(css)

        if not on_column:
            to_return = to_return * len(sample.index)

        return to_return

    # Filling NaN values so their value can be expressed as color with cmap
    df_copy = df.fillna(value=0)

    # Prepare data normalization
    if not is_numeric_dtype(df_copy[main_attribute]):  # FIXME : Or nb unique value < X ?
        # Create range from 0 to 1 for categorical data
        unique_values = df_copy[main_attribute].unique()
        nb_values = len(unique_values)

        categorical_values = {value: i / nb_values for i, value in enumerate(unique_values)}
        min_max = (0, 1)
    else:
        min_max = df_copy[main_attribute].min(), df_copy[main_attribute].max()
        categorical_values = None

    if print_infos:
        print(f"[MAIN] Highlighting '{main_attribute}' with cmap '{main_attribute_cmap.name}'")

    styler = df_copy.style.apply(lambda x: styler_fct(sample=x, target_attribute=main_attribute, min_max=min_max,
                                                      categorical_values=categorical_values, cmap=main_attribute_cmap),
                                 axis=1)

    # Highlight specific columns over the main_attributes highlighting
    if attributes:
        if type(attributes) != list:
            attributes = [attributes]

        # We won't use reversed cmaps, if the user want it, he can manually specify it
        all_cmaps = [m for m in plt.colormaps() if '_r' not in m]

        while len(other_attributes_cmaps) < len(attributes):
            new_cmap = plt.get_cmap(random.sample(all_cmaps, 1)[0])

            if new_cmap not in other_attributes_cmaps:
                other_attributes_cmaps.append(new_cmap)

        per_att_categorical_values = {}
        per_att_min_max = {}
        per_att_cmap = {}

        for attribute, cmap in zip(attributes, other_attributes_cmaps):
            # Prepare data normalization
            if not is_numeric_dtype(df_copy[attribute]):
                # Create range from 0 to 1 for categorical data
                unique_values = sorted(df_copy[attribute].unique())
                nb_values = len(unique_values)

                per_att_categorical_values[attribute] = {value: i / nb_values for i, value in enumerate(unique_values)}
                per_att_min_max[attribute] = (0, 1)
            else:
                per_att_min_max[attribute] = df_copy[attribute].min(), df_copy[attribute].max()
                per_att_categorical_values[attribute] = None

            per_att_cmap[attribute] = cmap

            if print_infos:
                print(f"[SUB]  Highlighting colum '{attribute}' with cmap '{cmap.name}'")

        styler = styler.apply(lambda x: styler_fct(sample=x, target_attribute=attributes, min_max=per_att_min_max,
                                                   categorical_values=per_att_categorical_values, cmap=per_att_cmap,
                                                   on_column=True),
                              axis=0)

    if format_dict:
        styler = styler.format(format_dict)

    return styler


def text_color_from_rgb(rgb, threshold=0.48):
    """
    Calculate relative luminance of a color.

    The calculation adheres to the W3C standards
    (https://www.w3.org/WAI/GL/wiki/Relative_luminance)

    This was borrowed from pandas.df.style.background_gradient
    The equation have been changed to match de 0-255 range of the RGB values

    Parameters
    ----------
    color : rgb or rgba tuple

    Returns
    -------
    float
        The relative luminance as a value from 0 to 1
    """
    r, g, b = (
        x / 3294.6 if x <= 10.0164 else ((x + 14.025) / 289.9653362853444)
        for x in rgb
    )
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return "#f1f1f1" if luminance < threshold else "#000000"


def convert_cols_to_int(df, columns):
    convert_dict = {}
    # For some reasons, if a colum doesn't contain any NaN value, we can't convert it to the nullable type Int64...
    for col in columns:
        if df[col].isnull().any():
            convert_dict[col] = 'Int64'
        else:
            convert_dict[col] = 'int64'

    return df.astype(convert_dict)
