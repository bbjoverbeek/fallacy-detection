#!/bin/bash
# Create a separate folder that contains all the necessary files to reproduce the experiment.
# Run this file in the root folder of the repository.

# ---------------------
# --- set variables ---
# ---------------------

ENV_FOLDER="./env/"
EXPERIMENTS_FOLDER="./experiments/"
MODEL="google/flan-t5-small"

# provide cli argument 'test' if you want to predict on test set instead of dev
if [ "$1" == "test" ]; then
    TEST_SET=true
    FOLDER_NAME="test/"
    DATASET="./data/test/mafalda_gold_standard_dataset.jsonl"
    else
    TEST_SET=false
    FOLDER_NAME="dev/"
    DATASET="./data/dev/fallacy_corpus.jsonl"
fi

# create experiments folder if it does not exist yet
mkdir -p $EXPERIMENTS_FOLDER $EXPERIMENTS_FOLDER$FOLDER_NAME

# -----------------------------------------------
# --- Create a new folder for the experiment. ---
# -----------------------------------------------

# Get a list of all the existing folders with the desired filename.
# Do not show any error messages if no folders exist yet.
folders=$(ls -d $EXPERIMENTS_FOLDER$FOLDER_NAME"experiment-"* 2> /dev/null)

# Find the highest folder number.
highest_folder_number=-1
for folder in $folders; do
    folder_number=$(basename $folder | cut -d "-" -f 2)
    if [ $folder_number -gt $highest_folder_number ]; then
        highest_folder_number=$folder_number
    fi
done

# Increment the highest folder number by 1.
new_folder_number=$((highest_folder_number + 1))

# Create the new folder.
EXPERIMENT_FOLDER="${EXPERIMENTS_FOLDER}$FOLDER_NAME"experiment-"$new_folder_number/"
mkdir $EXPERIMENT_FOLDER

# --------------------------------------------
# --- Copy the correct files to the folder ---
# --------------------------------------------

# copy the dataset
cp $DATASET $EXPERIMENT_FOLDER$(basename $DATASET)

# copy the python script
cp "./prompt_model.py" $EXPERIMENT_FOLDER"prompt_model.py"

# copy the run script
cp "./run_experiment.sh" $EXPERIMENT_FOLDER"run_experiment.sh"

# # --------------------------
# # --- Run the experiment ---
# # --------------------------

# activate the virtual environment
source $ENV_FOLDER"/bin/activate"

# call the script with the correct arguments
python3 $EXPERIMENT_FOLDER"prompt_model.py" \
        --model $MODEL \
        --prompt-features zero-shot chain-of-thought \
        --dataset $EXPERIMENT_FOLDER$(basename $DATASET) \
        --logpath $EXPERIMENT_FOLDER"log.txt"
