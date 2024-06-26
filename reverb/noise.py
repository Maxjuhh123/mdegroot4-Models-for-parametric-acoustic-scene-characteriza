"""
Module containing functionality to add additive white gaussian noise to a signal
"""
import numpy as np


def add_white_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add white Gaussian noise to a signal to achieve a desired SNR.

    :param signal: The original signal.
    :param snr_db: Desired signal-to-noise ratio in decibels (dB).

    :return: Signal with added white Gaussian noise.
    """
    # Calculate signal power and convert to dB
    signal_power = np.mean(signal ** 2)
    signal_power_db = 10 * np.log10(signal_power)

    # Calculate noise power in dB
    noise_power_db = signal_power_db - snr_db

    # Convert noise power from dB to linear scale
    noise_power = 10 ** (noise_power_db / 10)

    # Generate white Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)

    # Add noise to the original signal
    noisy_signal = signal + noise

    return noisy_signal
