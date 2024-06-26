#!/bin/bash
# Full pipeline to generate and evaluate a T60 estimator using speech intelligibility measures given a RIR database and preprocess_clean speech database
# The training and validation input parameters below are expected to be non-empty
# Be aware that new directories may be created by this script (ensure it has permission to do so)

# Define input parameters
CLEAN_PATH="resources/speech"  # Path to folder containing clean speech
RIRS_PATH1="resources/rirs/rirs_test_last.mat"  # Path to first file containing room impulse responses generated using Habets
RIRS_PATH2="resources/rirs/rirs_test_first.mat"  # Path to second file containing room impulse responses generated using Habets
ANALYSIS_OUTPUT_PATH="resources/output"  # Path to folder where analysis results will be saved
SNR=-1.0  # Signal to noise ratio for additive gaussian noise added to reverberant sound

# Defining constants (not recommended to change)
INTERMEDIATE_RIRS_PATH="resources/rirs/intermediate"
CLEAN_TRAIN_PATH="resources/training/speech/clean"  # Path to folder containing clean speech audio files (.wav) for generating the estimator
CLEAN_VALIDATION_PATH="resources/validation/speech/clean"  # Path to folder containing clean speech audio files (.wav) for evaluating the estimator
PROCESSED_TRAIN_RIRS_PATH="resources/training/rirs/processed"
PROCESSED_TRAIN_RIRS_CSV_PATH="resources/training/rirs/processed/rir-parameters.csv"
PROCESSED_VALIDATION_RIRS_PATH="resources/validation/rirs/processed"
PROCESSED_VALIDATION_RIRS_CSV_PATH="resources/validation/rirs/processed/rir-parameters.csv"
TRAIN_REVERB_PATH="resources/training/rev"
TRAIN_REVERB_CSV_PATH="resources/training/rev/overview.csv"
TRAIN_SIM_PATH="resources/training/sims"
VALIDATION_REVERB_PATH="resources/validation/rev"
VALIDATION_REVERB_CSV_PATH="resources/validation/rev/overview.csv"
VALIDATION_SIM_PATH="resources/validation/sims"
SIMS=("SIIB" "SIIB-gauss" "STOI" "ESTOI")

# Install dependencies from requirements file
echo "Generating and evaluating estimator, this may take a while..."
pip install -r requirements.txt

# Preprocess RIRs (ensure first peak is direct and calculate groundtruth t60s) and split into training/validation sets
echo "Preprocessing RIRs..."
python -m preprocess_rirs --input_rirs_path1 $RIRS_PATH1 --input_rirs_path2 $RIRS_PATH2 --training_rirs_path $PROCESSED_TRAIN_RIRS_PATH --validation_rirs_path $PROCESSED_VALIDATION_RIRS_PATH --intermediate_rirs_path $INTERMEDIATE_RIRS_PATH
echo "Done preprocessing RIRs."

# Split clean speech data into training and validation datasets
echo "Splitting clean speech into training and validation datasets..."
python -m preprocess_clean --dataset_path $CLEAN_PATH --training_output_path $CLEAN_TRAIN_PATH --validation_output_path $CLEAN_VALIDATION_PATH
echo "Done splitting clean speech data."

# Create reverbed training dataset
echo "Creating training reverb speech dataset..."
python -m reverb --clean_filepath $CLEAN_TRAIN_PATH --rir_filepath $PROCESSED_TRAIN_RIRS_PATH --rir_csv_filepath $PROCESSED_TRAIN_RIRS_CSV_PATH --reverberant_filepath $TRAIN_REVERB_PATH --reverberant_csv_filepath $TRAIN_REVERB_CSV_PATH --snr $SNR
echo "Done creating training reverb speech dataset..."

# Create reverbed validation dataset
echo "Creating validation reverb speech dataset..."
python -m reverb --clean_filepath $CLEAN_VALIDATION_PATH --rir_filepath $PROCESSED_VALIDATION_RIRS_PATH --rir_csv_filepath $PROCESSED_VALIDATION_RIRS_CSV_PATH --reverberant_filepath $VALIDATION_REVERB_PATH --reverberant_csv_filepath $VALIDATION_REVERB_CSV_PATH --snr $SNR
echo "Done creating validation reverb speech dataset..."

# Calculate training SIM data
echo "Calculating SIM training data..."
for sim in "${SIMS[@]}"
do
    echo "Calculating intelligibility measures for $sim..."
    # Calculate SIM dataset and save to file
    python -m sims --save_path $TRAIN_SIM_PATH --reverb_path $TRAIN_REVERB_PATH --clean_path $CLEAN_TRAIN_PATH --sim_type $sim
    echo "Done calculating intelligibility measures for $sim."
done
echo "Done calculating SIM training data."

# Calculate validation SIM data
echo "Calculating SIM validation data..."
for sim in "${SIMS[@]}"
do
    echo "Calculating intelligibility measures for $sim..."
    # Calculate SIM dataset and save to file
    python -m sims --save_path $VALIDATION_SIM_PATH --reverb_path $VALIDATION_REVERB_PATH --clean_path $CLEAN_VALIDATION_PATH --sim_type $sim
    echo "Done calculating intelligibility measures for $sim."
done
echo "Done calculating SIM validation data."

# Generate and evaluate estimator
for sim in "${SIMS[@]}"
do
    echo "Analysing data for $sim..."
    # Calculate SIM dataset and save to file
    python -m analysis --sim_type $sim --sim_folder_train $TRAIN_SIM_PATH --reverb_path_train $TRAIN_REVERB_CSV_PATH --sim_folder_validation $VALIDATION_SIM_PATH --reverb_path_validation $VALIDATION_REVERB_CSV_PATH --save_path $ANALYSIS_OUTPUT_PATH
    echo "Done analysis data for $sim."
done
echo "Done generating and evaluating estimator."

/bin/bash
