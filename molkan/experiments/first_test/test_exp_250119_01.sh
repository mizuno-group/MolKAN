#!/bin/bash

# activate your python environment
source /opt/pip-env/bin/activate

# get the directory of the current script
DIR=$(dirname $0)

# get the base name of the current script without extension
BASE=$(basename $0 .sh)

# run the corresponding .py file
python $DIR/$BASE.py \
    data_dir None \
    --note None \
    --seed 42 \
    --train True \
    --num_epochs 100 \
    --batch_size 256, 512, 1024 \
    --lr 0.01, 0.001, 0.0001 \
    --metrics accuracy, AUROC, sensitivity, precision, F1, MCC \
    --num_workers 16 \