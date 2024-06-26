"""
Main module, calculate Speech Intelligibility Measures (SIMs) and save to a file.
"""
import csv
import os

import numpy as np

from argparse import Namespace, ArgumentParser
from reverb.utils import load_wavs
from sims.speech_intelligibility import string_to_sim, SIMResult, IntrusiveSIM


def get_args() -> Namespace:
    """
    Get arguments from CLI.

    --save_path: Path to folder where SIM data should be saved
    --reverb_path: Path to folder containing reverb data
    --clean_path: Path to folder containing preprocess_clean speech data
    --sim_type: Type of SIM
    --normalize_clean: Whether the SIM data should be normalized by preprocess_clean intelligibility or not

    :return: Namespace containing arguments
    """
    parser = ArgumentParser()

    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--reverb_path', type=str, default='')
    parser.add_argument('--clean_path', type=str, default='')
    parser.add_argument('--sim_type', type=str, default='')
    parser.add_argument('--normalize_clean', type=bool, default=True)

    return parser.parse_args()


def calculate_measures(reverbs: [(np.ndarray, str, int)], cleans: [(np.ndarray, str, int)],
                       sim: IntrusiveSIM, should_normalize: bool) -> [SIMResult]:
    """
    Calculate SIM measures.

    :param reverbs: List of reverb speech reverb and corresponding file names
    :param cleans: List of preprocess_clean speech reverb and corresponding file names
    :param sim: The SIM to calculate
    :param fs: The sampling frequency of the speech reverb
    :param should_normalize: Whether the calculated intelligibility should be normalized bu the preprocess_clean intelligibility
    :return: The calculation results of the SIM
    """
    results = []
    for reverb, reverb_path, reverb_fs in reverbs:
        # Find corresponding preprocess_clean file
        for clean, clean_path, clean_fs in cleans:
            if clean_path.lower().replace('.wav', '') in reverb_path.lower():
                assert reverb_fs == clean_fs
                assert clean.shape == reverb.shape

                reverb_intelligibility = sim.apply(clean, reverb, clean_fs)
                if should_normalize:
                    clean_intelligibility = sim.apply(clean, clean, clean_fs)
                    reverb_intelligibility /= clean_intelligibility
                results.append(SIMResult(reverb_intelligibility, clean_path, reverb_path))
                break
    return results


def save_results(results: [SIMResult], folder_path: str, sim_name: str) -> None:
    """
    Save SIM calculation results to a csv file.

    :param results: The results to save
    :param folder_path: Which folder to save the results csv file to
    :param sim_name: Name of SIM type used in calculation
    """
    csv_file_path = os.path.join(folder_path, f'{sim_name}.csv')
    csv_file = open(csv_file_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = ['clean_audio', 'reverb_audio', 'measure']
    writer.writerow(header)

    for result in results:
        clean_audio = result.clean_speech_name
        reverb_audio = result.reverb_name
        measure = result.result

        # Write row to rir parameter file
        writer.writerow([clean_audio, reverb_audio, measure])

    csv_file.flush()
    csv_file.close()


def main() -> None:
    """
    Main function, entrypoint.
    """
    args = get_args()
    save_path = args.save_path
    reverb_path = args.reverb_path
    clean_path = args.clean_path
    sim_type = args.sim_type
    should_normalize = args.normalize_clean

    sim = string_to_sim(sim_type)

    if not os.path.exists(save_path):
        print(f"Directory '{save_path}' created")
        os.makedirs(save_path)

    reverbs = load_wavs(reverb_path)
    cleans = load_wavs(clean_path)
    results = calculate_measures(reverbs, cleans, sim, should_normalize=should_normalize)

    save_results(results, folder_path=save_path, sim_name=sim_type)


if __name__ == '__main__':
    main()
