import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


def grouped_scatter(dataframe, group_key, x_axis, y_axis, title=None, colormap=cm.viridis, ax=None):
    group_unique_keys = dataframe[group_key].unique()

    colorlist = {key: colors.rgb2hex(colormap(i)) for key, i in
                 zip(group_unique_keys, np.linspace(0, 0.9, len(group_unique_keys)))}

    if ax is None:
        _, ax = plt.subplots()

    for key, group in dataframe.groupby(group_key):
        group.plot.scatter(ax=ax, x=x_axis, y=y_axis, c=colorlist[key], label=key, title=title)


def sub_cols_with_cond_and_create_new_col(df, new_col_name, col_to_sub, cond1, cond2, output_cond):
    temp = df[cond1][col_to_sub].reset_index(drop=1) - df[cond2][col_to_sub].reset_index(drop=1)

    temp.index = df[output_cond].index
    df.loc[output_cond, new_col_name] = temp

    return df


def color_row_by_attribute(sample, attribute, colors_by_config):
    css = f"background-color: {colors_by_config[sample[attribute]]}"
    return [css] * len(sample.index)
