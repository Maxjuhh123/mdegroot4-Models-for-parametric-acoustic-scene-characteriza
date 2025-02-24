"""
Python module to preprocess_rirs RIRs generated by Habets Image Source implementation,
split into training and testing sets, then additional preprocessing steps to ensure they are in the right format (direct peak first etc).
"""
import os.path
import uuid

import numpy as np
from argparse import ArgumentParser
from multiprocessing.dummy import Namespace

from scipy.io import loadmat
from matplotlib import pyplot as plt
from pyroomacoustics.experimental import rt60
from scipy.io.wavfile import write
from sklearn.utils import shuffle

from preprocess_clean.__main__ import split_data, create_folder_if_not_exists
from preprocess_rirs.rir import save_rirs, RIR
from reverb.utils import load_wav


MAX_SAMPLES = 2000


def get_args() -> Namespace:
    """
    Get arguments from CLI.

    --input_rirs_path1: Path to first folder containing RIRs to process
    --input_rirs_path2: Path to second folder containing RIRs to process
    --intermediate_rirs_path: Path to folder where RIRs will be saved intermediately as wav files
    --training_rirs_path: Path to folder where processed training RIRs should be saved
    --validation_rirs_path: Path to folder where processed validation RIRs should be saved
    :return: Namespace containing arguments as attributes
    """
    parser = ArgumentParser()

    parser.add_argument('--input_rirs_path1', type=str, default=None)
    parser.add_argument('--input_rirs_path2', type=str, default=None)
    parser.add_argument('--intermediate_rirs_path', type=str, default='')
    parser.add_argument('--training_rirs_path', type=str, default='')
    parser.add_argument('--validation_rirs_path', type=str, default='')

    return parser.parse_args()


def estimate_t60(rir: np.ndarray, fs: int) -> float:
    """
    Given a RIR, estimate the T60. First tries to extrapolate from t30, then t20, then t15, fails if none succeed.

    :param rir: The room impulse response
    :param fs: The sampling frequency of the room impulse response
    :return: The estimated T60 (or error if T15 fails)
    """
    try:
        return rt60.measure_rt60(rir, fs=fs, decay_db=30)
    except:
        try:
            return rt60.measure_rt60(rir, fs=fs, decay_db=20)
        except:
            return rt60.measure_rt60(rir, fs=fs, decay_db=15)


def pre_process_rir(rir: np.ndarray, file_name: str, save_dir: str, fs: int = 16000, should_visualize=False) -> RIR:
    """
    Preprocess a room impulse response. Makes sure first value is the direct sound peak and measures t60.

    :param rir: Room impulse response as an array
    :param file_name: Name of the RIR file
    :param save_dir: Directory to which preprocessing data will be saved
    :param fs: Sampling frequency (Hz)
    :param should_visualize: Whether the processed RIR should get visualized or not
    :return: Processed RIR
    """
    assert len(rir) > 0
    rir = rir[np.argmax(rir):]
    t60 = estimate_t60(rir, fs)

    # Save visualization (debug purposes)
    if should_visualize:
        save_path = os.path.join(save_dir, file_name.lower().replace('.wav', '.png'))
        plt.plot(rir)
        plt.savefig(save_path)
        plt.close()

    write(os.path.join(save_dir, file_name), fs, rir)

    # Set unknown parameters to 0
    return RIR(rir, t60, (0, 0, 0), 0, file_name)


def normalize_rir(rir: np.ndarray) -> np.ndarray:
    """
    Normalize room impulse response so that the peak corresponds to 1.

    :param rir: The room impulse response
    :return: The normalized room impulse response
    """
    return rir / np.max(rir)


def load_rirs_from_mat(input_rir_path: str) -> ([RIR], int):
    """
    Load room impulse responses and sample rate from .mat file.

    :param input_rir_path: Path to .mat file containing RIRs
    :return: List of rirs and sampling frequency
    """
    res = []
    mat = loadmat(input_rir_path)
    fs = 48000
    h_keys = [str(key) for key in mat.keys() if key.startswith("h")]
    for key in h_keys:
        room_rirs = mat[key]
        for i in range(room_rirs.shape[1]):
            for j in range(room_rirs.shape[2]):
                rir_data = room_rirs[:, i, j]
                rir = RIR(rir_data, 0, (0, 0, 0), 0, f'{key}-{i}-{j}-{uuid.uuid4()}.wav')
                res.append(rir)

    res = shuffle(res)[:MAX_SAMPLES]
    return res, fs


def load_wavs_from_paths(paths: [str]) -> [(np.ndarray, str)]:
    """
    Given a list of paths to .wav files, load them as numpy arrays.

    :param paths: The list of paths
    :return: Array of signals loaded from wav files, along with the file names
    """
    res = []
    for path in paths:
        name = os.path.basename(path)
        if not name.lower().endswith('.wav'):
            continue
        folder = str(os.path.dirname(path)).replace('.', '')
        signal = load_wav(folder, name)[0]
        res.append((signal, name))
    return res


def main() -> None:
    """
    Main method, entrypoint of the application.
    """
    args = get_args()
    input_rirs_path1 = args.input_rirs_path1
    input_rirs_path2 = args.input_rirs_path2
    training_rirs_path = args.training_rirs_path
    validation_rirs_path = args.validation_rirs_path
    intermediate_rirs_path = args.intermediate_rirs_path

    # Load Habets rirs from .mat file and save in intermediate directory
    create_folder_if_not_exists(intermediate_rirs_path)
    rirs = []
    fs = -1
    for path in [input_rirs_path1, input_rirs_path2]:
        print(f'Loading RIRs from {path}')
        path_rirs, path_fs = load_rirs_from_mat(path)
        for rir in path_rirs:
            rirs.append(rir)
        print(f'Loaded {len(path_rirs)} from {path}')
        fs = path_fs
    save_rirs(rirs, intermediate_rirs_path, fs=fs, should_save_plots=False)

    create_folder_if_not_exists(training_rirs_path)
    create_folder_if_not_exists(validation_rirs_path)
    training_paths, validation_paths = split_data(intermediate_rirs_path, nested=False)
    training_rirs = load_wavs_from_paths(training_paths)
    validation_rirs = load_wavs_from_paths(validation_paths)

    # Processing training data
    processed_training_rirs = []
    for rir, file_name in training_rirs:
        try:
            processed_training_rirs.append(pre_process_rir(rir, file_name, training_rirs_path, fs))
        except:
            print(f'Failed to process {file_name}, excluding it')
    save_rirs(processed_training_rirs, training_rirs_path, fs=fs, should_save_plots=False)
    print(f'Saved processed RIRs to {training_rirs_path}')

    # Process validation data
    processed_validation_rirs = []
    for rir, file_name in validation_rirs:
        try:
            processed_validation_rirs.append(pre_process_rir(rir, file_name, validation_rirs_path, fs))
        except:
            print(f'Failed to process {file_name}, excluding it')
    save_rirs(processed_validation_rirs, validation_rirs_path, fs=fs, should_save_plots=False)
    print(f'Saved processed RIRs to {validation_rirs_path}')


if __name__ == '__main__':
    main()
