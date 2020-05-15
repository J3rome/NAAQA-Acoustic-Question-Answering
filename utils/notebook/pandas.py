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


def convert_cols_to_int(df, columns):
    convert_dict = {}
    # For some reasons, if a colum doesn't contain any NaN value, we can't convert it to the nullable type Int64...
    for col in columns:
        if df[col].isnull().any():
            convert_dict[col] = 'Int64'
        else:
            convert_dict[col] = 'int64'

    return df.astype(convert_dict)
