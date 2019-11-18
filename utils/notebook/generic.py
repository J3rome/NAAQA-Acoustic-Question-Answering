from collections import defaultdict

from IPython.core.display import display, HTML


def full_width_notebook():
    html_str = "<style>.container { width:99% !important; }\n"
    html_str += "div.cell.selected { border-left-width: 1px !important; }\n"
    html_str += "div.output_scroll { resize: vertical !important }</style>"
    display(HTML(html_str))


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


def format_epoch_folder(epoch_folder):
    if type(epoch_folder) == int or epoch_folder.isdigit():
        epoch_folder = f"Epoch_{epoch_folder:02d}"

    assert epoch_folder.startswith('Epoch') or epoch_folder == 'best', "Invalid epoch folder provided"

    return epoch_folder
