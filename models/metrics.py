import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


# Based on Bob Yang answer : https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
def calc_f1_score(preds, ground_truths, labels=None):
    if isinstance(preds, torch.Tensor) and preds.device.type != 'cpu':
        preds = preds.cpu()
        ground_truths = ground_truths.cpu()

    # FIXME : This kinda break the grad.. Rewrite in torch ?
    if labels is None:
        labels = unique_labels(preds, ground_truths)

    conf_matrix = confusion_matrix(ground_truths, preds, labels=labels)
    f1_scores = []

    for row_col_id in range(conf_matrix.shape[0]):
        row = conf_matrix[row_col_id, :]
        col = conf_matrix[:, row_col_id]

        true_positive = conf_matrix[row_col_id, row_col_id]
        false_positive = sum(row) - true_positive
        false_negative = sum(col) - true_positive

        if true_positive > 0:
            precision = true_positive / (true_positive + false_positive)
            recall = true_positive / (true_positive + false_negative)
            f1_scores.append(2 * (recall * precision) / (recall + precision))
        else:
            f1_scores.append(0)

    return np.mean(f1_scores)
