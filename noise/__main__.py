"""
Module for evaluating the estimator in noisy conditions
"""
import csv
import os.path
import random
from argparse import Namespace, ArgumentParser
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from sklearn.utils import shuffle

from analysis.__main__ import EVALUATION_METRICS
from analysis.estimator import Estimator, exp_model, fraction_model
from preprocess_rirs.__main__ import estimate_t60
from reverb.noise import add_white_gaussian_noise
from reverb.utils import load_wavs
from sims.speech_intelligibility import string_to_sim

FS = 48000          # Sampling frequency of the RIRs and speech signals
MAX_SAMPLES = 100   # Maximum amount of RIRs to use for the evaluation


def get_args() -> Namespace:
    """
    Get arguments from CLI.

    --clean_path: Path to folder containing clean speech.
    --rir_path: Path to file containing RIRs to process
    --sim_type: Type of SIM to evaluate on
    --ld: Lambda of the shifted exponential distribution
    --theta: Theta of the shifted exponential distribution
    --save_path: Path to save evaluation results to

    :return: Namespace containing arguments as properties
    """
    parser = ArgumentParser()

    parser.add_argument('--clean_path', type=str, default=None)
    parser.add_argument('--rir_path', type=str, default=None)
    parser.add_argument('--sim_type', type=str, default=None)
    parser.add_argument('--ld', type=float, default=-1)
    parser.add_argument('--theta', type=float, default=-1)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--a', type=float, default=-1)
    parser.add_argument('--snr', type=float, default=-1)

    return parser.parse_args()


def load_and_process_signals(clean_path: str, rir_path: str, snr: float, fs: int = 48000) -> [(np.ndarray, np.ndarray, float)]:
    """
    Load and process (add noise) clean signals and rirs, also calculates t60.

    :param clean_path: Path to folder containing clean speech
    :param rir_path: Path to folder containing room impulse responses
    :return: list of clean speech, reverb speech, and the t60.
    """
    cleans = [clean for clean, _, _ in load_wavs(clean_path)]
    rirs = [rir for rir, _, _ in shuffle(load_wavs(rir_path))[:MAX_SAMPLES]]

    res = []
    for rir in rirs:
        clean = random.choice(cleans)
        reverb = signal.fftconvolve(clean, rir)
        t60 = estimate_t60(rir, fs)
        reverb = reverb[:len(clean)]
        reverb = add_white_gaussian_noise(reverb, snr)
        res.append((clean, reverb, t60))
    return res


def main() -> None:
    """
    Main method, entrypoint of the application.
    """
    args = get_args()
    clean_path = args.clean_path
    rir_path = args.rir_path
    sim_type = args.sim_type
    ld = args.ld
    theta = args.theta
    a = args.a
    save_path = args.save_path
    snr = args.snr

    # Load estimator given parameters
    sim = string_to_sim(sim_type)
    estimator = None
    if 'stoi' in sim_type.lower():
        estimator = Estimator(lambda x: exp_model(x, ld, theta), f'Exponential MSE Estimator \$(\lambda = {np.round(ld, decimals=2)}, \\theta = {np.round(theta, decimals=2)})\$', {
            '\lambda': ld,
            '\\theta': theta
        })
    else:
        estimator = Estimator(lambda x: fraction_model(x, a),
                     f'Hyperbolic Rational MSE Estimator \$(a = {np.round(a, decimals=2)})\$', {
        'a': a
        })

    # Load clean and rir files, process as reverb
    signals = load_and_process_signals(clean_path, rir_path, snr)

    # Save T60 distribution
    t60s = [signal[2] for signal in signals]
    plt.hist(t60s)
    plt.savefig(os.path.join(save_path, f'{sim_type}-t60.jpg'))

    # Evaluate estimator on real rirs
    evaluation_results = estimator.evaluate_not_processed(signals, sim, EVALUATION_METRICS)

    # Save evaluation results
    overview_path = os.path.join(save_path, 'overview.csv')
    with open(overview_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the header if the file doesn't exist
        evaluation_metrics = evaluation_results.keys()
        if not os.path.isfile(overview_path):
            header = ['sim', 'parameters', 'timestamp'] + [x for x in evaluation_metrics]
            writer.writerow(header)
        # Write the reverb row
        data = [sim_type, str(estimator.params), str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))] \
               + [x for x in evaluation_results.values()]
        writer.writerow(data)


if __name__ == '__main__':
    main()