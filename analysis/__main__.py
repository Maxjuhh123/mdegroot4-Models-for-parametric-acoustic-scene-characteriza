"""
Main module for analysing SIM measure results.
"""
import csv
import os
from argparse import Namespace, ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
import pandas as pd

from analysis.estimator import Estimator, exp_estimator, fraction_estimator
from analysis.mutual_information import estimate_mutual_information

OVERVIEW_FILE_NAME = 'overview.csv'  # Name of the file the analysis results overview will be saved to
EVALUATION_METRICS = ['MSE', 'MAE']  # Metric to evaluate based on (mean squared error and mean absolute error)


def get_args() -> Namespace:
    """
    Get arguments from CLI.

    --sim_type: Type of SIM to analyse
    --sim_folder_train: Path to folder containing a csv file (named the same as the sim_type) containing training SIM data
    --sim_folder_validation: Path to folder containing a csv file (named the same as the sim_type) containing validation SIM data
    --reverb_path_train: Path to a csv file containing data about the training reverb dataset
    --reverb_path_train: Path to a csv file containing data about the validation reverb dataset
    --save_path: Path to folder to which the analysis results (figures and overview csv) will be saved
    """
    parser = ArgumentParser()

    parser.add_argument('--sim_type', type=str, default='SIIB')
    parser.add_argument('--sim_folder_train', type=str, default='resources/sims')
    parser.add_argument('--sim_folder_validation', type=str, default='resources/eval/sims')
    parser.add_argument('--reverb_path_train', type=str, default='resources/rev/overview.csv')
    parser.add_argument('--reverb_path_validation', type=str, default='resources/eval/rev/overview.csv')
    parser.add_argument('--save_path', type=str, default="resources/analysis_results")

    return parser.parse_args()


def process(sim_path_train: str, reverb_path_train: str, sim_path_valid: str, reverb_path_valid: str,
            save_path: str, sim_type: str) -> (Estimator, float):
    """
    Analyse T60 and SIM data, constructs an estimator and evaluates it (MSE and MAE).

    :param sim_path_train: Path where the training sim measure csv file is located
    :param reverb_path_train: Path where the training reverb csv file is located
    :param sim_path_valid: Path where the validation sim measure csv file is located
    :param reverb_path_valid: Path where the validation reverb csv file is located
    :param save_path: Path to the folder the results should be saved
    :param sim_type: Type of SIM
    :return: Estimator found and the mutual information
    """
    sim_data_train = pd.read_csv(sim_path_train)
    reverb_data_train = pd.read_csv(reverb_path_train)
    sim_data_valid = pd.read_csv(sim_path_valid)
    reverb_data_valid = pd.read_csv(reverb_path_valid)

    # Merge SIM and reverb reverb
    sim_reverb_data_train = pd.merge(sim_data_train, reverb_data_train,
                               on='reverb_audio',
                               how='inner')
    sim_reverb_data_valid = pd.merge(sim_data_valid, reverb_data_valid,
                                     on='reverb_audio',
                                     how='inner')

    # Only keep relevant columns
    filtered_data_train = sim_reverb_data_train[["measure", "t60"]]
    filtered_data_valid = sim_reverb_data_valid[["measure", "t60"]]

    t60s_train = filtered_data_train['t60']
    sims_train = np.array(filtered_data_train['measure']).reshape(-1, 1)
    t60s_valid = filtered_data_valid['t60']
    sims_valid = np.array(filtered_data_valid['measure']).reshape(-1, 1)

    estim = exp_estimator(t60s_train, sims_train) if 'stoi' in sim_type.lower() \
        else fraction_estimator(t60s_train, sims_train)
    estim.visualize(sims_train, t60s_train, sims_valid, t60s_valid, sim_type, 't60 (s)',
                    sim_type, save_path=save_path, invert=True)
    mutual_information = estimate_mutual_information(sims_train, t60s_train)

    return estim, mutual_information


def main() -> None:
    """
    Main method, entrypoint.
    """
    args = get_args()
    sim_type = args.sim_type.lower()
    sim_folder_train = args.sim_folder_train
    sim_path_train = os.path.join(sim_folder_train, f'{sim_type}.csv')
    reverb_path_train = args.reverb_path_train
    reverb_path_valid = args.reverb_path_validation
    sim_path_valid = os.path.join(args.sim_folder_validation, f'{sim_type}.csv')
    save_path = args.save_path

    # matplotlib setup
    plt.rcParams['svg.fonttype'] = 'none'
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Get estimator
    estimator, mutual_info = process(sim_path_train, reverb_path_train, sim_path_valid,
                                     reverb_path_valid, save_path, sim_type)
    # Evaluate estimator
    overview_path = os.path.join(save_path, OVERVIEW_FILE_NAME)
    file_exists = os.path.isfile(overview_path)
    evaluation_results = estimator.evaluate(reverb_path_valid, sim_path_valid, EVALUATION_METRICS)

    with open(overview_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file doesn't exist
        evaluation_metrics = evaluation_results.keys()
        if not file_exists:
            header = ['sim', 'mutual_info', 'parameters', 'timestamp'] + [x for x in evaluation_metrics]
            writer.writerow(header)
        # Write the reverb row
        data = [sim_type, str(mutual_info), str(estimator.params), str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))] \
            + [x for x in evaluation_results.values()]
        writer.writerow(data)


if __name__ == '__main__':
    main()
