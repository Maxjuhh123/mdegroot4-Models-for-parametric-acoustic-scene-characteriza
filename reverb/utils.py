"""
Utility module for reverb data.
"""
import os
import numpy as np
import soundfile as sf

from datetime import datetime


def create_folder(save_folder_path: str, suffix: str = "") -> str:
    """
    Generates a new folder for a database based on current datetime.

    :param save_folder_path: Path to create the new db folder in
    :param suffix: The suffix to put after the generated folder name
    :return: The name of the generated folder
    """
    if not os.path.exists(save_folder_path):
        print(f"Directory '{save_folder_path}' created")
        os.makedirs(save_folder_path)

    # Generate new folder for rir db within the given save_path
    new_dir_name = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S_{suffix}")
    new_path = os.path.join(save_folder_path, new_dir_name)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    return new_path


def save_signals(signals: [(np.ndarray, str, int)], save_folder_path: str) -> str:
    """
    Save list of signals to a specified directory.
    Creates new directory if it does not exist.

    :param signals: List of signals (tuples of numpy arrays, file names and sample rates).
    :param save_folder_path: Path the signals should be saved to
    :return: The path the signals were saved to
    """
    if not os.path.exists(save_folder_path):
        print(f"Directory '{save_folder_path}' created")
        os.makedirs(save_folder_path)

    for signal, file_name, fs in signals:
        file_path = os.path.join(save_folder_path, file_name)
        sf.write(file_path, signal, fs)
    return save_folder_path


def load_wav(folder_path: str, file_name: str) -> (np.ndarray, str, int):
    """
    Load wavfile as numpy array, the file name and the sampling frequency (Hz).

    :param folder_path: The folder where the wav file is located
    :param file_name: The file name of the wav file
    :return: Wavfile as numpy array, the file name and the sampling frequency (Hz).
    """
    file_path = os.path.join(folder_path, file_name)
    if not file_name.lower().endswith('wav'):
        raise ValueError(f'{file_path} is not a .wav file')

    data, fs = sf.read(file_path)
    return data, file_name, fs


def load_wavs(folder_path: str) -> [(np.ndarray, str, int)]:
    """
    Load all .wav files in a folder as arrays, also includes the file_path and sampling frequency (Hz).

    :param folder_path: The path to the folder
    :return: List of numpy arrays from the .wav files, along with filenames and sampling frequencies
    """
    return [load_wav(folder_path, path) for path in os.listdir(folder_path) if path.lower().endswith('wav')]
