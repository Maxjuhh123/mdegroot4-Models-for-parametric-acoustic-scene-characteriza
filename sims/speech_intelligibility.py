"""
Module containing functionality related to speech intelligibility measures.
"""
import numpy as np

from dataclasses import dataclass
from typing import Callable
from sims.siib import calculate_siib, calculate_siib_gauss
from sims.stoi import calculate_stoi, calculate_extended_stoi


class IntrusiveSIM:
    """
    Class representing an intrusive speech intelligibility measure.
    """
    calculation_function: Callable[
        [np.ndarray, np.ndarray, int], float]  # Maps preprocess_clean speech, reverbed speech and sampling frequency to an intelligibility score

    def __init__(self, calculation_function: Callable[[np.ndarray, np.ndarray, int], float]):
        self.calculation_function = calculation_function

    def apply(self, clean_speech: np.ndarray, reverbed_speech: np.ndarray, fs: int) -> float:
        """
        Applies the calculation function of the SIM to preprocess_clean speech and reverb speech

        :param clean_speech: The preprocess_clean speech signal
        :param reverbed_speech: The reverbed speech signal
        :param fs: The sampling frequency of the speech signals
        :return: An intelligibility score
        """
        return self.calculation_function(clean_speech, reverbed_speech, fs)


@dataclass
class SIMResult:
    """
    Class describing the result of a SIM computation.
    """
    result: float
    clean_speech_name: str
    reverb_name: str


def string_to_sim(sim_name: str) -> IntrusiveSIM:
    """
    Converts a speech intelligibility metric name to a class representing it.

    :param sim_name: Name of the speech intelligibility metric
    :return: Representation of the metric
    """
    normalized_name = sim_name.lower().replace(' ', '')
    if normalized_name == 'siib':
        return IntrusiveSIM(calculation_function=calculate_siib)
    elif normalized_name == 'siib-gauss':
        return IntrusiveSIM(calculation_function=calculate_siib_gauss)
    elif normalized_name == 'stoi':
        return IntrusiveSIM(calculation_function=calculate_stoi)
    elif normalized_name == 'estoi':
        return IntrusiveSIM(calculation_function=calculate_extended_stoi)
    else:
        raise ValueError(f'Invalid intelligibility metric: {sim_name}')
