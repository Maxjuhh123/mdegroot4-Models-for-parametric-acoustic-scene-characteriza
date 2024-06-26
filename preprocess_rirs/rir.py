"""
This is a module containing functionality related to RIRs
"""
import csv
import os
import numpy as np

from dataclasses import dataclass
from typing import List
from matplotlib import pyplot as plt
from soundfile import write
from reverb.utils import create_folder


@dataclass
class RIR:
    """
    Class representing a room impulse response and its parameters
    """
    rir_data: np.ndarray
    t60: float
    room_dimensions: (float, float, float)
    absorption_coefficient: float
    file_name: str


def save_rirs(rirs: List[RIR], save_folder_path: str, should_save_plots: bool = True, fs: int = 8000,
              create_new_dir: bool = False, only_data: bool = False) -> None:
    """
    Save list of room impulse responses to a folder, also generates csv file containing RIR parameters.

    :param rirs: List of room impulse responses (and parameters)
    :param save_folder_path: Path to folder where the files will be saved
    :param should_save_plots: If set to true, a plot for each RIR will be saved
    :param fs: Sampling frequency used for RIR generation
    :param create_new_dir: Whether a new folder should be generated for the RIRs or not.
    :param only_data: When true, only data is saved to an overview file and new RIRs are not saved to wav files
    """
    db_folder_name = create_folder(save_folder_path, suffix='rir_db') if create_new_dir else save_folder_path
    if not create_new_dir and not os.path.exists(save_folder_path):
        os.mkdir(save_folder_path)

    # Create RIR parameter file
    csv_file_path = os.path.join(db_folder_name, 'rir-parameters.csv')
    csv_file = open(csv_file_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = ['rir_name', 't60', 'room_dim_x', 'room_dim_y', 'room_dim_z', 'absorption_coefficient']
    writer.writerow(header)

    for i in range(len(rirs)):
        rir = rirs[i]
        rir_data = rir.rir_data
        t60 = rir.t60
        file_name = rir.file_name
        (x, y, z) = rir.room_dimensions

        # Save RIR to .wav file
        if not only_data:
            file_path = os.path.join(db_folder_name, file_name)
            write(file_path, rir_data, fs)

            # Save RIR plot
            if should_save_plots:
                plt.plot(rir_data)
                plt.savefig(file_path.replace('.wav', '.png'))
                plt.close()

        # Write row to rir parameter file
        writer.writerow([file_name, t60, x, y, z, rir.absorption_coefficient])

    # Also visualize distribution of t60
    t60s = [rir.t60 for rir in rirs]
    plt.hist(t60s, bins=30)
    plt_path = os.path.join(db_folder_name, 't60s.png')
    plt.savefig(plt_path)
    plt.close()

    # Flush csv file
    csv_file.flush()
    csv_file.close()
