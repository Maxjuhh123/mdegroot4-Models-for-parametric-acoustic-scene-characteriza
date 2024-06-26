"""
Module containing functionality regarding mutual information (estimation).
"""
import numpy as np
from sklearn.feature_selection import mutual_info_regression


def estimate_mutual_information(xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Estimate mutual information between a list of SIM data and corresponding t60 data.

    :param xs: List containing SIM data
    :param ys: List containing T60 data
    :return: Estimated mutual information
    """
    return mutual_info_regression(xs, ys, discrete_features=False, copy=True)
