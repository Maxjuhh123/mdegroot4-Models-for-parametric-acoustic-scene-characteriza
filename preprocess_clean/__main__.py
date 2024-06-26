"""
Module to split EARS dataset into training and validation datasets (50/50)
Also calculates statistics for both training and validation datasets.
"""
import json
import os.path
from argparse import Namespace, ArgumentParser
from typing import List

import numpy as np
import soundfile as sf
from sklearn.utils import shuffle


def get_args() -> Namespace:
    """
    Get arguments from CLI.

    --dataset_path: Path to dataset folder, this script is tailored to the ears dataset.
    --training_output_path: Path to folder where the training dataset will be saved
    --validation_output_path: Path to folder where the training dataset will be saved

    :return: Namespace containing the arguments as properties
    """
    parser = ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default='.')
    parser.add_argument('--training_output_path', type=str, default='')
    parser.add_argument('--validation_output_path', type=str, default='')

    return parser.parse_args()


EXCLUDED_NAMES = ['cough', 'nonverbal', 'laugh', 'cry', 'sneez', 'throat', 'yawn', 'eat']  # TODO check which others to exclude


def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it does not exist.

    :param folder_path: Path to folder
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f'Created folder: {folder_path}')


def process_clean_file(file_path: str, target_folder_path: str) -> None:
    """
    Process a clean speech file by copying to a target folder.

    :param file_path: Path to the file to copy
    :param target_folder_path: Path to the folder to copy to
    """
    create_folder_if_not_exists(target_folder_path)

    # Construct the full path for the destination file
    file_name = os.path.basename(file_path)
    destination_path = os.path.join(target_folder_path, file_name)

    data, fs = sf.read(file_path)
    sf.write(destination_path, data, fs)


def split_data(folder_path: str, excluded_names: [str] = [], nested=True) -> (List[str], List[str]):
    """
    Randomly split data in a folder into two datasets.

    :param folder_path: Path to folder
    :param excluded_names: List of names which when present in a file path, the file will be ignored
    :param nested: Whether folders are nested (clean speech) or not (rirs)
    :return: Split data into two datasets.
    """
    left = []
    right = []

    for folder_name in os.listdir(folder_path) if nested else '.':
        curr_folder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(curr_folder_path):
            for file_name in shuffle(os.listdir(curr_folder_path)):
                file_path = os.path.join(curr_folder_path, file_name)
                excluded = False
                for excluded_name in excluded_names:
                    if excluded or excluded_name in file_path:
                        excluded = True
                        break
                if not excluded:
                    if len(left) > len(right):
                        right.append(file_path)
                    else:
                        left.append(file_path)
    return left, right


def save_json_stats(data: dict, file_path: str) -> None:
    """
    Save dictionary containing statistics to json file.

    :param data: Dictionary containing data to save
    :param file_path: Path to file to save to
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
        file.flush()
        file.close()


def calculate_and_save_statistics(file_paths: [str], save_folder_path: str, overview_file_path: str) -> None:
    """
    Given a list of file_paths, calculate statistics about the data.

    :param file_paths: Paths to files.
    :param save_folder_path: Folder to which the overview of statistics should be saved
    :param overview_file_path: Path to json file containing overview of statistics for all files
    """
    create_folder_if_not_exists(save_folder_path)
    stats_overview = json.load(open(overview_file_path, 'r'))
    stats_count = {}
    normalized_stats_count = {}

    # For each statistic type calculate the count for each label
    for speaker, stats in stats_overview.items():
        for file_path in file_paths:
            if speaker in file_path:
                for stat_type, label in stats.items():
                    if stat_type not in stats_count.keys():
                        stats_count[stat_type] = {}
                    if label not in stats_count[stat_type].keys():
                        stats_count[stat_type][label] = 1
                    else:
                        stats_count[stat_type][label] += 1

    # Calculate normalized counts for each stat type and label
    for stat_type, stats in stats_count.items():
        if stat_type not in normalized_stats_count.keys():
            normalized_stats_count[stat_type] = {}
        stat_count = np.sum([value for value in stats.values()])
        for label, count in stats.items():
            normalized_count = np.round((100.0 * count)/stat_count, decimals=2)
            normalized_stats_count[stat_type][label] = normalized_count

    # Save normal and normalized statistics to json files
    save_json_stats(stats_count, os.path.join(save_folder_path, 'stats.json'))
    save_json_stats(normalized_stats_count, os.path.join(save_folder_path, 'stats_normalized.json'))


def main() -> None:
    """
    Main method, entrypoint of the application.
    """
    args = get_args()

    dataset_path = args.dataset_path
    training_path = args.training_output_path
    validation_path = args.validation_output_path

    overview_path = os.path.join(dataset_path, 'speaker_statistics.json')

    # Create export folders
    create_folder_if_not_exists(training_path)
    create_folder_if_not_exists(validation_path)

    # Split data 50/50 into training and validation directories
    training_paths, validation_paths = split_data(dataset_path, EXCLUDED_NAMES)
    [process_clean_file(file_path, training_path) for file_path in training_paths]
    [process_clean_file(file_path, validation_path) for file_path in validation_paths]

    # Calculate statistics for both training and validation datasets
    calculate_and_save_statistics(training_paths, training_path, overview_path)
    calculate_and_save_statistics(validation_paths, validation_path, overview_path)


if __name__ == '__main__':
    main()
