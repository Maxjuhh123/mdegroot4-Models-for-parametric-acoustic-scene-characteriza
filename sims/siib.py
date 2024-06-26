"""
Module containing functionality related to Speech Intelligibility in Bits (SIIB).
pysiib library: https://github.com/kamo-naoyuki/pySIIB
Original papers:
- S. Van Kuyk, W. B. Kleijn, and R. C. Hendriks, ‘An instrumental intelligibility metric based on information theory’, IEEE Signal Processing Letters, 2018.
- S. Van Kuyk, W. B. Kleijn, and R. C. Hendriks, ‘An evaluation of intrusive instrumental intelligibility metrics’, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2018.
"""
import numpy as np
from pysiib import SIIB


def calculate_siib(clean_speech: np.ndarray, reverb_speech: np.ndarray, fs: int = 8000) -> float:
    """
    Calculates speech intelligibility in bits given preprocess_clean and reverbed speech using pysiib library.

    :param clean_speech: Clean speech signal
    :param reverb_speech: Reverbed speech signal
    :param fs: Sampling frequency of the signals (Hz)
    :return: The calculates SIIB score
    """
    return SIIB(clean_speech, reverb_speech, fs=fs, window='hamming', gauss=False)


def calculate_siib_gauss(clean_speech: np.ndarray, reverb_speech: np.ndarray, fs: int = 8000) -> float:
    """
    Calculates speech intelligibility in bits given preprocess_clean and reverbed speech (gaussian variant) using pysiib library.

    :param clean_speech: Clean speech signal
    :param reverb_speech: Reverbed speech signal
    :param fs: Sampling frequency of the signals (Hz)

    :return: The calculates SIIB score
    """
    return SIIB(clean_speech, reverb_speech, fs=fs, window='hamming', gauss=True)
