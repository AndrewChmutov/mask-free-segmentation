import numpy as np
from scipy.special import softmax


def weighted_crossentropy(y_pred, y_true, class_weights=[1.0, 5.0]):
    # Apply softmax to logits
    y_pred = softmax(y_pred, axis=-1)

    # Gather class weights
    weights = np.take(class_weights, y_true)

    # Compute cross-entropy loss
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.log(y_pred[np.arange(len(y_true)), y_true])

    # Apply weights and compute mean loss
    weighted = loss * weights
    return np.mean(weighted)


def tversky(y_pred, y_true, alpha=5, beta=0.3):
    # Apply softmax to logits
    y_pred = softmax(y_pred, axis=-1)

    # One-hot encode y_true
    num_classes = y_pred.shape[-1]
    y_true_onehot = np.eye(num_classes)[y_true]

    # Compute TP, FP, FN
    TP = np.sum(y_true_onehot * y_pred, axis=(1, 2))
    FP = np.sum((1 - y_true_onehot) * y_pred, axis=(1, 2))
    FN = np.sum(y_true_onehot * (1 - y_pred), axis=(1, 2))

    tversky_index = TP / (TP + alpha * FP + beta * FN + 1e-7)
    return 1 - np.mean(tversky_index)


def iou(y_pred, y_true):
    # Intersection and Union
    intersection = np.sum(y_pred & y_true)
    union = np.sum(y_pred | y_true)

    # IoU loss
    iou = intersection / union if union != 0 else 1
    iou = 1 - iou
    return iou


def dice(y_pred, y_true):
    # Intersection
    intersection = np.sum(y_pred & y_true)

    # Dice coefficient
    dice = (
        (2 * intersection) /
        (np.sum(y_pred) + np.sum(y_true))
        if (np.sum(y_pred) + np.sum(y_true)) != 0 else 1
    )
    dice = 1 - dice
    return dice
