"""Provides convenient methods to calculate the loss values"""

__author__ = 'khanh.p.d.truong@ntnu.no'
__date__ = '2024-08-23'

import numpy as np


def loss_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    a: float,
    b: float,
    dist: callable
) -> float:
    """Compute the loss function for the given ground truth and prediction.

    Args:
        y_true (np.ndarray): Ground truth set.
        y_pred (np.ndarray): Prediction set.
        a (float): False negative penalty.
        b (float): False positive penalty.
        dist (callable): The distance function.

    Returns:
        float: The loss value.
    """
    # Convert y_true and y_pred to numpy arrays for efficient computation
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute the pairwise distances between each y_true and each y_pred
    # Shape: len(y_true) x len(y_pred)
    distances = np.array([
        [dist(y_true_i, y_pred_j)
         for y_pred_j in y_pred] for y_true_i in y_true])

    # For each y_pred_j, determine the closest y_true_i's index
    # Shape: len(y_pred)
    min_dist_idx = np.argmin(distances, axis=0)

    # For each y_true_i, count the number of y_pred_j's that are assigned to it
    # (size of S_y_true_i). Shape: len(y_true)
    counts = np.bincount(min_dist_idx, minlength=len(y_true))

    # For each y_true_i, calculate the minimum distance to its assigned
    # y_pred_j's. Shape: len(y_true)
    min_dist = np.array([
        distances[i, min_dist_idx == i].min() if counts[i] > 0 else 0
        for i in range(len(y_true))])

    # Calculate the loss for y_true_i with non-zero S_y_true_i
    counts_mask = counts > 0
    L = np.sum(min_dist[counts_mask] + b * (counts[counts_mask] - 1))

    # Add the penalty for y_true_i with zero S_y_true_i
    L += np.sum(counts == 0) * a

    L = L / len(y_true)  # Normalize the loss by the number of y_true

    return L


def timestamp_dist(y_true_i: np.ndarray, y_pred_j: np.ndarray) -> float:
    """Calculate the absolute difference in seconds between two timestamps.

    Args:
        y_true_i (np.ndarray): The first timestamp. Can be multi-dimensional.
        y_pred_j (np.ndarray): The second timestamp. Can be multi-dimensional.

    Returns:
        float: The absolute difference in seconds between the two timestamps.
    """
    # Check if the input is multi-dimensional
    if (
        isinstance(y_true_i, (list, np.ndarray)) and
        isinstance(y_pred_j, (list, np.ndarray))
    ):
        if len(y_true_i) > 1:  # Multi-dimensional case
            # Calculate the absolute difference for each dimension and sum
            dimension_dist = ([
                abs((y_true_i[k] - y_pred_j[k]).total_seconds())
                for k in range(len(y_true_i))])
            return sum(dimension_dist)
        else:  # 1-dimensional case
            return abs((y_true_i[0] - y_pred_j[0]).total_seconds())
    else:  # 1-dimensional case
        return abs((y_true_i - y_pred_j).total_seconds())
