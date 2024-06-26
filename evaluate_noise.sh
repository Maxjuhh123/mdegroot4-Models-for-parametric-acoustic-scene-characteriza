#!/bin/bash
# Script to evaluate estimator on noise
# Define parameters
CLEAN_PATH="resources/validation/speech/clean"  # Path to clean speech
RIR_PATH="resources/validation/rirs/processed"  # Path to RIRs
SAVE_PATH="resources/output/noise"              # Folder to save results to
SIM="ESTOI"                                     # Which SIM to use (SIIB, SIIB-Gauss, STOI, ESTOI)
SNR=30                                          # Signal to noise ratio

LAMBDA=0.3958424603174925                       # Shifted exponential distribution parameter (for STOI and ESTOI)
THETA=0.4785852124483606                        # Shifted exponential distribution parameter (for STOI and ESTOI)
A=0.04331165                                    # Hyperbolic rational parameter              (for SIIB and SIIB-Gauss)

pip install -r requirements.txt

python -m noise --clean_path $CLEAN_PATH --rir_path $RIR_PATH --ld $LAMBDA --theta $THETA --sim_type $SIM --save_path $SAVE_PATH --a $A --snr $SNR
/bin/bash