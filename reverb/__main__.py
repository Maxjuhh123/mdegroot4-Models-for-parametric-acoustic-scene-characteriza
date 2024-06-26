"""
Module containing functionality relating to reverbed speech.
"""
import random
import numpy as np
import pandas as pd
import scipy.signal as signal
import os
import soundfile as sf

from argparse import ArgumentParser, Namespace
from preprocess_clean.__main__ import create_folder_if_not_exists
from reverb.noise import add_white_gaussian_noise
from reverb.utils import load_wavs


def get_args() -> Namespace:
    """
    Get arguments from CLI.

    --clean_filepath: Path to directory containing all preprocess_clean audio samples to process
    --rir_filepath: Path to directory containing all RIRs to process
    --rir_csv_filepath: Path to CSV file containing the properties of the RIRs to process
    --reverberant_filepath: Path to directory to which generated reverb samples will be saved.
    --reverberant_csv_filepath: Path (including filename, excluding extension) where the csv containing an overview of the samples will be saved to.

    :return: Namespace containing arguments as attributes
    """
    parser = ArgumentParser()
    parser.add_argument("--clean_filepath",
                        help="Path to directory containing all clean audio samples to process.",
                        type=str,
                        default=""
                        )
    parser.add_argument("--rir_filepath",
                        help="Path to directory containing all RIRs to process.",
                        type=str,
                        default=""
                        )
    parser.add_argument("--rir_csv_filepath",
                        help="Path to CSV file containing the properties of the RIRs to process",
                        type=str,
                        default=""
                        )
    parser.add_argument("--reverberant_filepath",
                        help="Path to directory to which generated reverb samples will be saved.",
                        type=str,
                        default=""
                        )
    parser.add_argument("--reverberant_csv_filepath",
                        help="Path (including filename, excluding extension) where the csv containing an overview of "
                             "the samples will be saved to.",
                        type=str,
                        default=""
                        )
    parser.add_argument("--snr",
                        help="Signal to noise ratio, no noise is added if not specified",
                        type=float,
                        default=-1.0
                        )
    return parser.parse_args()


def generate_reverb_sample(clean: (np.ndarray, str, int), rir_data: (np.ndarray, str, int),
                           rev_filepath: str, snr: float = -1) -> [str, str, str]:
    """
    Generates a set of reverberant audio samples from one preprocess_clean audio sample and a set of room impulse responses (RIR).
    Saves result to a .wav file in the directory mentioned.

    Sample rate of RIRs should be equal to the sample rate of the preprocess_clean audio. Will throw error otherwise.

    :param clean: (audio, filename, sample_rate) - preprocess_clean audio sample
    :param rir_data: (audio, filename, sample_rate,) - RIR
    :param rev_filepath: path to directory to save resulting reverberant audio samples to as .wav files.
    :param snr: for additive noise added to the reverbed speech data, if smaller than 0, no noise is added
    :return: array with information on samples: `[clean_audio, rir_audio, reverb_audio]`
    """
    clean_audio: np.ndarray
    clean_audio, name_clean, sr_clean = clean
    rir: np.ndarray
    rir, name_rir, sr_rir = rir_data

    assert clean_audio.dtype == np.float64
    assert rir.dtype == np.float64
    assert sr_rir == sr_clean

    reverb = signal.fftconvolve(clean_audio, rir)[:clean_audio.shape[0]]  # FFT convolution because it's faster
    assert reverb.shape == clean_audio.shape

    # Adding additive Gaussian noise if snr > 0
    if snr > 0:
        reverb = add_white_gaussian_noise(reverb, snr)

    # saving:
    _, tc = os.path.split(name_clean)  # audio.wav
    _, tr = os.path.split(name_rir)  # audio.wav
    nc, __ = os.path.splitext(tc)
    nr, __ = os.path.splitext(tr)
    name_reverb = nc + '_r_' + nr + '.wav'
    reverb_path = rev_filepath + '/' + name_reverb

    sf.write(reverb_path, reverb, sr_clean)
    return [tc, tr, name_reverb]  # names of preprocess_clean, RIR and reverb audio files


def process_samples(cleans: [(np.ndarray, str, int)], rirs: [(np.ndarray, str, int)],
                    rev_filepath: str, snr: float = -1) -> [[str, str, str]]:
    """
    Process preprocess_clean and rir samples to obtain reverb data.
    For each RIR randomly (uniform) picks preprocess_clean sample to convolve with.

    :param cleans: List containing preprocess_clean data (raw data, file name, sample rate)
    :param rirs: List containing RIR data (raw data, file name, sample rate)
    :param rev_filepath: Path to directory to which reverb data will be saved
    :param snr: for additive noise added to the reverbed speech data, if smaller than 0, no noise is added
    :return: array with information on samples (as arrays): `[clean_audio, rir_audio, reverb_audio]`
    """
    res = []
    for rir in rirs:
        # Pick random signal from preprocess_clean signals
        clean = random.choice(cleans)
        res.append(generate_reverb_sample(clean, rir, rev_filepath, snr))
    return res


def process(clean_filepath: str, rir_filepath: str, rev_filepath: str,
            rir_csv_filepath: str, res_csv_filepath: str, snr: float = -1) -> None:
    """
    Processes a set of preprocess_clean audio samples and room impulse responses (RIRs) to produce a reverb audio dataset saved at
    the specified location. Also generates a csv file with an overview of the samples and their properties with the columns:
    `[clean_audio, rir_audio, reverb_audio, ...]` where '...' are the RIR properties provided in the csv at 'rir_csv_filepath'

    :param clean_filepath: path to directory containing all preprocess_clean audio files to process. Expects `.wav` files.
    :param rir_filepath: path to directory containing all RIRs to process. Expects `.wav` files.
    :param rev_filepath: path to directory to which generated reverb samples will be saved
    :param rir_csv_filepath: path to csv file containing properties of the RIRs
    :param res_csv_filepath: path to which generated CSV file will be saved
    :param snr: for additive noise added to the reverbed speech data, if smaller than 0, no noise is added
    :return: None
    """
    rirs = load_wavs(rir_filepath)
    cleans = load_wavs(clean_filepath)

    res = process_samples(cleans, rirs, rev_filepath, snr)
    res_df = pd.DataFrame(np.array(res), columns=['clean_audio', 'rir_audio', 'reverb_audio'])

    rir_df = pd.read_csv(rir_csv_filepath)  # should have column [x] indicating file name incl .wav
    # filename column used as foreign key to join the two csv's

    final_info_df = res_df.join(rir_df.set_index('rir_name'), on='rir_audio', validate='m:1')

    final_info_df.to_csv(res_csv_filepath, index=False)
    print('Saved csv to', res_csv_filepath)


def main() -> None:
    """
    Main method, entrypoint of the application.
    """
    args = get_args()
    create_folder_if_not_exists(args.reverberant_filepath)
    process(args.clean_filepath,
            args.rir_filepath,
            args.reverberant_filepath,
            args.rir_csv_filepath,
            args.reverberant_csv_filepath,
            args.snr
            )


if __name__ == "__main__":
    main()
