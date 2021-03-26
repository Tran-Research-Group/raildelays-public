############################
# imports
############################
import pdb

import tensorflow as tf
import datetime
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.metrics import mean_squared_error, mean_absolute_error

############################
# functions
############################


def evaluation(y, y_hat, output_stats):
    '''
    Evaluation function: interface to calculate MAPE, MAE and RMSE between ground truth and prediction.
    Extended version: multi-step prediction can be calculated by self-calling.
    :param y: np.ndarray or int, ground truth, y.shape = (batch_size, 1, n_nodes, n_features_out=1)
    :param y_: np.ndarray or int, prediction, y_hat.shape = (batch_size, 1, n_nodes, n_features_out=1)
    :param output_stats: dict, paras of z-scores (mean & std).
    :return: np.ndarray, averaged metric values.
    '''
    # only works for the case of 1 output feature
    v = z_score_inverse(y, output_stats["mean"], output_stats["std"]).numpy().squeeze()
    v_hat = z_score_inverse(y_hat, output_stats["mean"], output_stats["std"]).numpy().squeeze()
    rmse = mean_squared_error(v, v_hat, squared = False)
    mae = mean_absolute_error(v, v_hat)

    return rmse, mae


def MAPE(y, y_hat):
    """mean absolute percent error

    Args:
        y (np.ndarray): ground truth
        y_hat (np.ndarray): predicted value

    Returns:
        int: RMSE averages over all elements of the input
    """
    return np.mean(np.abs((y - y_hat) / y + 1e-5)) * 100


def z_score(x, mean, std):
    '''
    Z-score normalization function: $z = (X - \mu) / \sigma $,
    where z is the z-score, X is the value of the element,
    $\mu$ is the population mean, and $\sigma$ is the standard deviation.
    :param x: np.ndarray, input array to be normalized.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score normalized array.
    '''
    return (x - mean) / std


def z_score_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return (x * std) + mean

