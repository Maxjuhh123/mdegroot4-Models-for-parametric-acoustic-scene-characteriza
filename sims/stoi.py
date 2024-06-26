"""
Module containing functionality relating to the (extendend) short time objective intelligibility measure.
pystoi library: https://github.com/mpariente/pystoi
original papers:
- C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time Objective Intelligibility Measure for Time-Frequency Weighted Noisy Speech', ICASSP 2010, Texas, Dallas.
- C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech', IEEE Transactions on Audio, Speech, and Language Processing, 2011.
- J. Jensen and C. H. Taal, 'An Algorithm for Predicting the Intelligibility of Speech Masked by Modulated Noise Maskers', IEEE Transactions on Audio, Speech and Language Processing, 2016.
"""
import numpy as np
from pystoi import stoi


def calculate_stoi(clean_speech: np.ndarray, reverb_speech: np.ndarray, fs=16000) -> float:
    """
    Calculate the STOI intelligibility measure.

    :param clean_speech: Clean speech array
    :param reverb_speech: Reverb speech array
    :param fs: Sampling frequency of the speech fragments
    :return: Calculated intelligibility score
    """
    return stoi(clean_speech, reverb_speech, fs)


def calculate_extended_stoi(clean_speech: np.ndarray, reverb_speech: np.ndarray, fs=16000) -> float:
    """
    Calculate the ESTOI intelligibility measure.

    :param clean_speech: Clean speech array
    :param reverb_speech: Reverb speech array
    :param fs: Sampling frequency of the speech fragments
    :return: Calculated intelligibility score
    """
    return stoi(clean_speech, reverb_speech, fs, extended=True)
