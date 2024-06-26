"""
Module for functionality relating to estimators
"""
import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tol_colors import tol_cset

from sims.speech_intelligibility import IntrusiveSIM


def calculate_mse(t60s_ground_truth: [float], t60s_estimate: [float]) -> float:
    """
    Calculate mean squared error given ground truth and estimate values.

    :param t60s_ground_truth: Ground truth data
    :param t60s_estimate: Estimate data
    :return: Mean squared error
    """
    return mean_squared_error(y_true=t60s_ground_truth, y_pred=t60s_estimate)


def calculate_mae(t60s_ground_truth: [float], t60s_estimate: [float]) -> float:
    """
    Calculate mean absolute error given ground truth and estimate values.

    :param t60s_ground_truth: Ground truth data
    :param t60s_estimate: Estimate data
    :return: Mean absolute error
    """
    return mean_absolute_error(y_true=t60s_ground_truth, y_pred=t60s_estimate)


class Estimator:
    """
    Class representing an estimator
    """
    estimate_function: Callable[[float], float]
    name: str
    params: Dict[str, float]

    def __init__(self, estimate_function: Callable[[float], float], name: str, params: Dict[str, float]):
        """
        Constructor for an estimator.

        :param estimate_function: The function which given a value, calculates the output of the estimator
        :param name: Name of the estimator
        :param params: Dictionary defining parameters of the estimator and their optimized values
        """
        self.estimate_function = estimate_function
        self.name = name
        self.params = params

    def estimate(self, x: float) -> float:
        """
        Apply the estimator.

        :param x: Value to apply estimator tox
        :return: The estimate
        """
        return self.estimate_function(x)

    def visualize(self, xs_train: [float], ys_train: [float], xs_val: [float], ys_val: [float], x_name: str, y_name: str, sim_type: str, save_path: str,
                  invert: bool = False) -> None:
        """
        Visualize the estimator by plotting expected values on scatter plot of original reverb.

        :param xs_train: List of training values on x-axis
        :param ys_train: List of training values on y-axis
        :param xs_val: List of validation values on x-axis
        :param ys_val: List of validation values on y-axis
        :param x_name: Name for x-axis
        :param y_name: Name for y-axis
        :param sim_type: Name of SIM
        :param save_path: Path to folder to save visualizations to
        :param invert: Whether x and y axes should be inverted
        """
        cmap = tol_cset('vibrant')
        x_plot = np.linspace(min(xs_train), max(xs_train), 400)  # Avoid zero to prevent division by zero
        line_color = cmap[4]
        train_color = cmap[1]
        valid_color = cmap[0]
        label_pad = 20
        # Compute the corresponding y values
        y_plot = [self.estimate_function(x) for x in x_plot]

        plt.figure(figsize=(10, 6))
        label = 'Estimator'
        if invert:
            plt.scatter(ys_val, xs_val, label='Validation', color=valid_color)
            plt.scatter(ys_train, xs_train, label='Training', color=train_color)
            plt.plot(y_plot, x_plot, color=line_color, label=label)
            plt.xlabel(y_name, labelpad=label_pad)
            plt.ylabel(x_name, labelpad=label_pad)
        else:
            plt.scatter(xs_val, ys_val, label='Validation', color=valid_color)
            plt.scatter(xs_train, ys_train, label='Training', color=train_color)
            plt.plot(x_plot, y_plot, color=line_color, label=label)
            plt.xlabel(x_name, labelpad=label_pad)
            plt.ylabel(y_name, labelpad=label_pad)

        plt.title(self.name)
        plt.legend(loc='upper right', labelspacing=1.5, borderpad=7, fontsize='small')
        plt.tick_params(axis='both', which='major', pad=10)
        plt.savefig(os.path.join(save_path, f'{sim_type}.svg'))
        plt.close()

    def evaluate(self, reverb_path: str, sim_path: str, metrics: List[str]) -> Dict[str, float]:
        """
        Evaluate an estimator using different metrics.

        :param reverb_path: Path to csv overview of reverberation reverb.
        :param sim_path: Path to csv overview of sim reverb.
        :param metrics: List of metrics to evaluate
        :return: Map of evaluation scores of the estimator
        """
        sim_data = pd.read_csv(sim_path)
        reverb_data = pd.read_csv(reverb_path)

        # Merge SIM and reverb reverb
        sim_reverb_data = pd.merge(sim_data, reverb_data,
                                   on='reverb_audio',
                                   how='inner')

        # Only keep relevant columns
        filtered_data = sim_reverb_data[["measure", "t60"]]
        t60s = filtered_data['t60']  # True t60s
        sims = np.array(filtered_data['measure'])
        t60s_estimate = [self.estimate(sim) for sim in sims]  # Estimated t60s

        res = {}

        for metric in metrics:
            normalized_metric = metric.lower()
            result = -1.0
            if normalized_metric == 'mse':
                result = calculate_mse(t60s, t60s_estimate)
            elif normalized_metric == 'mae':
                result = calculate_mae(t60s, t60s_estimate)
            res[metric] = result
        return res

    def evaluate_not_processed(self, signal_data: [(np.ndarray, np.ndarray, float)],
                               sim_type: IntrusiveSIM, metrics: List[str], fs=48000) -> Dict[str, float]:
        """
        Evaluate the estimator on non-processed data (SIMs not calculated yet)

        :param signal_data: Dataset of clean speech signals, reverberant speech signals, and the corresponding T60
        :param sim_type: Type of SIM to evaluate for
        :param metrics: List of metrics to evaluate
        :param fs: Sampling frequency of the signals (should be equal)
        :return: Map of evaluation scores of the estimator
        """
        sim_t60_values = [(sim_type.apply(clean, reverb, fs), t60) for clean, reverb, t60 in signal_data]
        sims, ground_truth_t60s = ([sim for sim, _ in sim_t60_values], [t60 for _, t60 in sim_t60_values])
        estimated_t60s = [self.estimate(sim) for sim in sims]

        res = {}

        for metric in metrics:
            normalized_metric = metric.lower()
            result = -1.0
            if normalized_metric == 'mse':
                result = calculate_mse(ground_truth_t60s, estimated_t60s)
            elif normalized_metric == 'mae':
                result = calculate_mae(ground_truth_t60s, estimated_t60s)
            res[metric] = result
        return res


def fraction_model(x: float, a: float):
    """
    Model of hyperbolic rational function a/x.

    :param x: The input to the function
    :param a: The parameter to be optimized
    :return: The result of applying the function to x with parameter a
    """
    return a/x


def fraction_estimator(t60s: [float], sims: [float]) -> Estimator:
    """
    Given t60 and SIM data construct a curve fit estimator by fitting a hyperbolic rational function.
    Minimizes mean squared error using BGFS approach

    :param t60s: T60 data
    :param sims: SIM data
    :return: The estimator
    """
    # Define the function to minimize (sum of squared errors)
    def objective(params, x, y):
        a = params
        y_pred = fraction_model(x, a)
        return mean_squared_error(y, y_pred)

    # Initial guesses for a and b
    initial_guess = 1

    # Perform the optimization
    result = minimize(objective, initial_guess, args=(sims, t60s), method='BFGS')

    # Get the estimated parameters
    a_est = result.x
    print(f"Estimated parameters: a = {a_est}")

    return Estimator(lambda x: fraction_model(x, a_est),
                     f'Hyperbolic Rational MSE Estimator \$(a = {np.round(a_est, decimals=2)[0]})\$', {
        'a': a_est[0]
    })


def exp_model(x: float, ld: float, th: float):
    """
    Model of shifted exponential function

    :param x: The input to the function
    :param ld: The rate parameter lambda to be optimized
    :param theta: The shift parameter theta to be optimized
    :return: The result of applying the function to x with parameters ld and th
    """
    return th + (np.log(1/ld) - np.log(x)) * ld


def exp_estimator(t60s: [float], sims: [float]) -> Estimator:
    """
    Given t60 and SIM data construct a curve fit estimator by fitting a shifted exponential function.
    Minimizes mean squared error using BGFS approach

    :param t60s: T60 data
    :param sims: SIM data
    :return: The estimator
    """
    # Define the function to minimize (sum of squared errors)
    def objective(params, x, y):
        ld, th = params
        y_pred = exp_model(x, ld, th)
        return mean_squared_error(y, y_pred)

    # Initial guesses for a and b
    initial_guess = [1, 0]

    # Perform the optimization
    result = minimize(objective, initial_guess, args=(sims, t60s), method='CG')

    # Get the estimated parameters
    ld_est, th_est = result.x
    print(f"Estimated parameters: lambda = {ld_est}, theta = {th_est}")

    return Estimator(lambda x: exp_model(x, ld_est, th_est), f'Inverse Exponential MSE Estimator \$(\lambda = {np.round(ld_est, decimals=2)}, \\theta = {np.round(th_est, decimals=2)})\$', {
        '\lambda': ld_est,
        '\\theta': th_est
    })
