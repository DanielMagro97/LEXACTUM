# Copyright (C) 2020  Daniel Magro
# Full License at: https://github.com/DanielMagro97/LEXACTUM/blob/main/LICENSE

from typing import List                 # for type annotation

import numpy as np                      # for np.linspace
from sklearn.metrics import roc_curve   # for calculating ROC curves
from sklearn.metrics import auc         # for calculating AUC metric
import matplotlib.pyplot as plt         # for plotting ROC curves


# Function which find the number of positive and negative labels in the data set
def get_total_pos_neg(test_labels: List, pos_label):
    total_pos: int = 0
    total_neg: int = 0

    for label in test_labels:
        if label == pos_label:
            total_pos += 1
        else:
            total_neg += 1

    return total_pos, total_neg


# TODO get_tpr_0 and get_tpr_10 might not work correctly when pos_label is 0
# TODO the fix for such a situation would be to reverse the np.linspace(1, 0, 1001)
# Function which calculates the TPR and FPR given the true labels, predicted labels,
# threshold, total negatives and total positives
def get_tpr_fpr(true_labels: List, pred_labels: List, pos_label, threshold: float,
                total_pos: int, total_neg: int):
    true_pos: int = 0
    false_pos: int = 0
    # TODO might need brackets around the tuple
    for y_true, y_pred in zip(true_labels, pred_labels):
        if y_pred <= threshold:  # TODO < or <=
            y_pred = 0
        else:
            y_pred = 1

        # check if the model's prediction is a true positive
        if y_true == pos_label and y_pred == pos_label:
            true_pos += 1
        # if not, check if it is a false positive
        elif y_true != pos_label and y_pred == pos_label:
            false_pos += 1

    tpr: float = true_pos / total_pos
    fpr: float = false_pos / total_neg

    return tpr, fpr


def get_tpr_0(true_labels: List, pred_labels: List, pos_label,
              total_pos: int, total_neg: int):
    tpr_0: float = 0.0
    # for p in np.linspace(0, 1, 1001):
    for p in np.linspace(1, 0, 1001):
        tpr, fpr = get_tpr_fpr(true_labels, pred_labels, pos_label, p,
                               total_pos, total_neg)
        if fpr == 0:
            tpr_0 = tpr
        else:
            break

    return tpr_0


def get_tpr_10(true_labels: List, pred_labels: List, pos_label,
              total_pos: int, total_neg: int):
    tpr_10: float = 0.0
    # for p in np.linspace(0, 1, 1001):
    for p in np.linspace(1, 0, 1001):
        tpr, fpr = get_tpr_fpr(true_labels, pred_labels, pos_label, p,
                               total_pos, total_neg)
        if fpr <= (10 / total_neg):
            tpr_10 = tpr
        else:
            break

    return tpr_10


def show_results(true_labels: List, pred_labels: List, pos_label=1):
    # Adapted from: https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(true_labels, pred_labels, pos_label=pos_label)
    auc_keras = auc(fpr_keras, tpr_keras)
    print('AUC: ' + str(auc_keras))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # plt.show()

    # get the total number of positive and negative labels
    total_pos, total_neg = get_total_pos_neg(true_labels, pos_label)
    # print(total_pos)
    # print(total_neg)

    tpr, fpr = get_tpr_fpr(true_labels, pred_labels, pos_label, 0.5,
                           total_pos, total_neg)
    print('TPR: ' + str(tpr))
    print('FPR: ' + str(fpr))

    # Find TPR0
    tpr_0: float = get_tpr_0(true_labels, pred_labels, pos_label,
                             total_pos, total_neg)
    print('TPR_0: ' + str(tpr_0))

    # Find TPR10
    tpr_10: float = get_tpr_10(true_labels, pred_labels, pos_label,
                               total_pos, total_neg)
    print('TPR_10: ' + str(tpr_10))

    return auc_keras, tpr_0, tpr_10
