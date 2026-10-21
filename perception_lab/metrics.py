"""Pixel-level segmentation metrics, with explicit absent-class handling."""

import numpy as np
from .data import CLASS_NAMES


def confusion_matrix(truth, predicted, classes=3):
    truth, predicted = np.asarray(truth).ravel(), np.asarray(predicted).ravel()
    if truth.shape != predicted.shape or not truth.size:
        raise ValueError("Labels must have equal, nonempty shape")
    if (not np.issubdtype(truth.dtype, np.integer)
            or not np.issubdtype(predicted.dtype, np.integer)):
        raise ValueError("Labels must be integers")
    if truth.min() < 0 or predicted.min() < 0 or truth.max() >= classes or predicted.max() >= classes:
        raise ValueError("Label outside class range")
    return np.bincount(truth * classes + predicted, minlength=classes**2).reshape(classes, classes)


def summarize(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (3, 3) or np.any(matrix < 0) or matrix.sum() == 0:
        raise ValueError("Expected nonnegative, nonempty 3 x 3 confusion matrix")
    diagonal = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - diagonal
    iou = np.divide(diagonal, union, out=np.full(3, np.nan), where=union != 0)
    # Floor false positives matter: a dangerous object might be labelled free space.
    floor_tp = int(matrix[1, 1])
    floor_predicted, floor_true = int(matrix[:, 1].sum()), int(matrix[1].sum())
    return {"pixel_accuracy": float(diagonal.sum() / matrix.sum()),
            "mean_iou": float(np.nanmean(iou)),
            "iou_by_class": {name: (None if np.isnan(value) else float(value))
                             for name, value in zip(CLASS_NAMES, iou)},
            "floor_precision": floor_tp / floor_predicted if floor_predicted else None,
            "floor_recall": floor_tp / floor_true if floor_true else None,
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist()}
