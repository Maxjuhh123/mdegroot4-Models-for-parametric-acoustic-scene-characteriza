#!/bin/bash
# Script to evaluate estimator on real RIRs
CLEAN_PATH="resources/validation/speech/clean"  # Path to clean speech
RIR_PATH="resources/rirs/dechorate.hdf5"        # Path to RIRs
SAVE_PATH="resources/rirs/real"                 # Folder to save results to
SIM="ESTOI"                                     # Which SIM to use (SIIB, SIIB-Gauss, STOI, ESTOI)

LAMBDA=2.0119341380952083                       # shifted exponential distribution parameter (for STOI and ESTOI)
THETA=0.6695765827513377                        # shifted exponential distribution parameter (for STOI and ESTOI)
A=0.04331165                                    # hyperbolic rational parameter               (for SIIB and SIIB-Gauss)

pip install -r requirements.txt

python -m process_dechorate --clean_path $CLEAN_PATH --rir_path $RIR_PATH --ld $LAMBDA --theta $THETA --sim_type $SIM --save_path $SAVE_PATH --a $A
/bin/bash