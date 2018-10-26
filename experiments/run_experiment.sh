#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOTDIR="${DIR}/.."
OLDDIR=$PWD
CURRENT_DATE_TIME=$(date +'%d-%m-%Y_%Hh%M')

EXPERIMENT_NAME=$1
if [ "${EXPERIMENT_NAME: -1}" = "/" ]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME:: -1}"
fi
EXPERIMENT_DIR="${DIR}/${EXPERIMENT_NAME}"
ROOT_LOG_DIR="${EXPERIMENT_DIR}/log"

if [ ! -d "${ROOT_LOG_DIR}" ]; then
  mkdir "${ROOT_LOG_DIR}"
fi

LOG_DIR="${ROOT_LOG_DIR}/${CURRENT_DATE_TIME}"
if [ ! -d "${LOG_DIR}" ]; then
  mkdir "${LOG_DIR}"
fi

# TODO : This script must be run in the virtual environment. Figure a way to make sure we are in the environment ?

# Will stop the script on first error
set -e

cd $ROOTDIR
echo "-----------------------------------------------------------------------------------------------------------"
echo "    AQA Film Network  --- Experiment : ${EXPERIMENT_NAME}  ---  `date +"%d/%m/%Y %H:%M"`"
echo "-----------------------------------------------------------------------------------------------------------"
echo "[NOTE] This script should be run inside the virtual environment associated with aqa neural network"
echo "[NOTE] The output of each process can be found in the log folder of the experiment"
echo "[NOTE] Stopping this script will not stop the background process."
echo "[NOTE] Make sure all the process are stopped if CTRL+C on this script"
echo "-----------------------------------------------------------------------------------------------------------"


# Preprocess images
echo "Preprocessing images (Extract Features)"
python src/clevr/preprocess_data/extract_image_features.py @${EXPERIMENT_DIR}/preprocess_images_features.args
echo -e "Preprocessing of images done\n"

# Preprocess questions
echo "Preprocessing questions (Tokenization)"
python src/clevr/preprocess_data/create_dictionary.py @${EXPERIMENT_DIR}/preprocess_questions.args
echo -e "Preprocessing of questions done\n"

# Train network
echo "Starting network training"
python src/clevr/train/train_aqa.py @${EXPERIMENT_DIR}/training.args

# Save trained network in specific place
