"""
Module for processing process_dechorate dataset RIRs for validation purposes.
bibtex citation:
@article{dicarlo2021dechorate,
  title={process_dechorate: a calibrated room impulse response dataset for echo-aware signal processing},
  author={{Di Carlo}, Diego and Tandeitnik, Pinchas and Foy, Cedri{\'c} and Bertin, Nancy and Deleforge, Antoine and Gannot, Sharon},
  journal={EURASIP Journal on Audio, Speech, and Music Processing},
  volume={2021},
  number={1},
  pages={1--15},
  year={2021},
  publisher={Springer}
}
"""
import csv
import os.path
import random

import matplotlib.pyplot as plt
import scipy.signal as signal
import uuid
from argparse import Namespace, ArgumentParser
from datetime import datetime

import h5py
import numpy as np
from sklearn.utils import shuffle

from analysis.__main__ import EVALUATION_METRICS
from analysis.estimator import Estimator, exp_model, fraction_model
from preprocess_rirs.__main__ import estimate_t60
from preprocess_rirs.rir import RIR
from reverb.utils import load_wavs
from sims.speech_intelligibility import string_to_sim


FS = 48000
RIR_TIME = 3
MAX_SAMPLES = 300


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

    return parser.parse_args()


def load_real_rirs(rir_path: str) -> [RIR]:
    """
    Load real rirs given path to the file.

    :param rir_path: Path to RIRs file
    :return: List of RIRs
    """
    res = []
    with h5py.File(rir_path, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        keys = list(f.keys())
        for key in keys:
            for group in list(f[key]):
                if 'rir' in group:
                    rir = f[key][group]
                    for rir_group in rir:
                        rir_group_folder = f[key][group][rir_group]
                        for rir_sub_group in rir_group_folder:
                            rir_data = np.array(f[key][group][rir_group][rir_sub_group])
                            rir_data = rir_data[np.argmax(rir_data):]
                            t60 = estimate_t60(rir_data, FS)
                            max_length = int(FS * 1.2 * t60)
                            rir_data = rir_data[:max_length]
                            rir_data = rir_data.reshape(rir_data.shape[0])
                            name = f'resources/new/{uuid.uuid4()}.jpg'
                            # plt.plot(rir_data)
                            # plt.savefig(name)
                            # plt.close()
                            rir_to_add = RIR(rir_data, t60, (0, 0, 0), 0, name)
                            res.append(rir_to_add)
    res = shuffle(res)[:MAX_SAMPLES]

    return res


def load_and_process_signals(clean_path: str, rir_path: str) -> [(np.ndarray, np.ndarray, float)]:
    """
    Load and process clean signals and rirs, also calculates t60.

    :return: list of clean speech, reverb speech, and the t60.
    """
    cleans = [clean for clean, _, _ in load_wavs(clean_path)]
    rirs = load_real_rirs(rir_path)

    res = []
    for rir in rirs:
        clean = random.choice(cleans)
        reverb = signal.fftconvolve(clean, rir.rir_data)
        reverb = reverb[:len(clean)]
        res.append((clean, reverb, rir.t60))
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
    signals = load_and_process_signals(clean_path, rir_path)

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