import matplotlib.pyplot as plt
import numpy as np


def plot_discrete_hist(data, key=None, title=None, legend_label=None, sort_key_fct=None, show_fig=False,
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

    if capitalize and type(labels[0]) == 'str':
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

    rects = ax.bar(x, counts, width=bar_width, align='center', label=legend_label)
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


def plot_2d_matrix(matrix, xlabels, ylabels=None, title=None, xaxis_name='predictions', yaxis_name="ground truth",
                   show_fig=True, colormap=plt.cm.Blues, add_annotations=True, normalize=False):

    if ylabels is None:
        ylabels = xlabels

    if normalize:
        if type(normalize) == int:
            norm_ax = normalize
        else:
            norm_ax = 1

        sums = matrix.sum(axis=norm_ax)[:, np.newaxis]
        sums[sums == 0] = 1
        matrix = matrix / sums

    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=colormap)
    # FIXME : Colorbar is too big
    ax.figure.colorbar(im, ax=ax)#, boundaries=[0, max_val])
    ax.set(xticks=np.arange(matrix.shape[1]), yticks=np.arange(matrix.shape[0]), yticklabels=ylabels,
           title=title, ylabel=yaxis_name.capitalize(), xlabel=xaxis_name.capitalize())
    ax.set_xticklabels(xlabels, rotation=90)
    ax.axis('image')

    if add_annotations:
        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'

        min_val = matrix.min()
        thresh = min_val + (matrix.max() - min_val)/2

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value > 0:
                    string = format(value, fmt)
                    if normalize and (value < 0.01 or value > 0.99):
                        string = "~" + string

                    ax.text(j, i, string,
                            ha="center", va="center", fontsize=6,
                            color="white" if value > thresh else "black")

    fig.tight_layout()

    if show_fig:
        plt.show()

    return fig, ax


def autolabel_bar(ax, rects):
    for rect in rects:
        h = rect.get_height()
        h_str = f'{int(h)}' if h >= 1 or h == 0 else f'{float(h):.2f}'
        ax.text(rect.get_x() + rect.get_width()/2., 1.00 * h, h_str, ha='center', va='bottom', fontsize=6)