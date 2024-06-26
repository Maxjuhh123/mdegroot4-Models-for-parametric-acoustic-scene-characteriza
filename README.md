# Statistically estimating Room Impulse Response (RIR) parameters based on Speech Intelligibility Measures (SIMs)
Author: Maxim de Groot
Contact: m.r.degroot-1@student.tudelft.nl

This codebase provides a full pipeline to construct a statistical estimator for the reverberation time (T60) using Speech Intelligibility Measures (SIMs).
The codebase constructs 1 estimator for each SIM. The SIMs used are SIIB, SIIB-Gauss, STOI and ESTOI. The first two are implemented using the pysiib library and the other two 
using the pystoi library. The codebase is divided into numerous modules, but the entire pipeline to reproduce results presented in my paper is divided into three shell scripts described in the
remainder of this documentation.

## Licensing Information
### Code licensing
The source code in this project is licensed under the GNU General Public License (GPL) v3.0. This license allows you to use, modify, and distribute the code, provided that any derivative works are also licensed under the GPL v3.0. This ensures that the code remains free and open for the community.

### Dataset licensing
The dataset included in this project is licensed under a Creative Commons Attribution-NonCommercial (CC BY-NC) 4.0 License. This license permits the use and modification of the dataset, but strictly for non-commercial purposes only. Proper attribution must be given when using the dataset, and commercial use is not allowed.

### Commercial Use Notice
If you wish to use this project for commercial purposes, you must adhere to the following conditions:
- Codebase: The codebase can be used under the terms of the GPL v3.0, which allows for commercial use as long as the terms of the GPL are followed.
- Dataset: The included dataset cannot be used for commercial purposes due to its non-commercial license. To use this project commercially, you must substitute the included dataset with a different anechoic clean speech dataset that permits commercial use.

### Important Notes
1. Code Usage and Distribution: You may use, modify, and distribute the source code under the terms of the GPL v3.0. This includes the right to use the code commercially, provided that all derivative works are also licensed under the GPL v3.0.
2. Dataset Usage: The dataset included in this project is restricted to non-commercial use only. Any commercial use of this dataset is strictly prohibited. For commercial applications, you must find and use an alternative dataset that allows for commercial use.
3. Combining Code and Dataset: If you are using the code in conjunction with the dataset, ensure that your usage complies with both the GPL v3.0 for the code and the CC BY-NC 4.0 License for the dataset. For commercial use, replace the non-commercial dataset with a commercially permissible one.

## Preliminaries
In order to run the pipeline, make sure you can execute shell scripts on your device and that you are using a Python 3.9 virtual environment.
In order
Using the `rir-scripts/rir_test_t60_uniform.m` script, generate two .mat files containing RIRs, the default parameters should suffice but can be adjusted.
Save the two .mat files relative to the root directory as:
- `resources/rirs/rirs_test_last.mat`
- `resources/rirs/rirs_test_first.mat`
Note that this is not my code, it uses Habets RIR generator implementation. It has an MIT license.

This split into two files is required as the original dataset is too large to be accessed in a single file.

## Speech Signals
Download the EARS speech dataset from https://github.com/facebookresearch/ears_dataset and copy all folders in the dataset to `resources/rirs/speech`

## Obtaining and Evaluating Estimators
In order to obtain the estimators, you can run the `pipeline_python3.sh` script from the root directory, parameters can be changed but default parameters should suffice.
This script splits both clean speech and rir datasets into training and validation datasets randomly. Then these are convolved using the fast Fourier transform to produce the reverb dataset (including ground truth T60).

Then for each SIM, the dataset is created using the clean and reverb datasets.

Lastly, the estimator is constructed and evaluated using the sim dataset and reverb dataset.
After this last step, analysis results will be saved to the `resources/output` folder.

## Evaluating under noise
In order to evaluate the estimators under noise conditions, the `evaluate_noise.sh` script can be used.
In this script a few parameters are required to be defined:
- CLEAN_PATH: Path to clean speech, default is the validation dataset
- RIR_PATH: Path to folder containing RIRs, default is the validation dataset
- SNR: The signal to noise ratio the reverbed samples should have
- LAMBDA: Lambda rate parameter of the shifted exponential function (for STOI and ESTOI)
- THETA: Theta shift parameter of the shifted exponential function (for STOI and ESTOI)
- A: A scale parameter of the hyperbolic rational function (for SIIB and SIIB-Gauss)
- SIM: The SIM to evaluate for (SIIB, SIIB-Gauss, STOI, ESTOI)
- SAVE_PATH: Where the resulting overview should be saved, this directory must exist!

## Evaluate on real rirs
In order to evaluate the estimators on real rirs, the `evaluate_real_ris.sh` script can be used.
In this script the same parameters as for `evaluation_noise.sh` must be defined, apart from SNR.
The script is specifically designed to work for the dEchorate RIR .hdf5 file, so the RIR_PATH in this case
must be the path to this file. The file is downloaded from: https://zenodo.org/records/4626590

## Modularity
This codebase is divided into many modules, below we give an overview of what each module does:
- analysis: Contains code to construct and evaluate the estimators
- noise: Contains code to evaluate the estimators under noise conditions
- preprocess_clean: Contains code to pre process the clean data (training and validation split)
- preprocess_rirs: Contains code to pre process RIRs (training and validation split and calculate T60s)
- process_dechorate: Contains code to evaluate the estimators on real RIRs from the dEchorate dataset
- reverb: Contains code to produce reverbed speech signals from the clean speech and RIRs
- sims: Contains code to calculate SIM value

For more in depth explanations, each file contains documentation and the entire methodology is documented in my paper.

## Common errors
Whenever you ecounter an error like 'python not found'
- Make sure python is installed
- Change `python` in the command to `python3` and `pip` to `pip3` (or the other way around), do this in the `.sh` scripts too.