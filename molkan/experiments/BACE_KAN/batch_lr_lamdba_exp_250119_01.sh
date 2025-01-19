#!/bin/bash

# activate your python environment
source /opt/pip-env/bin/activate

# get the directory of the current script
DIR=$(dirname $0)

# get the base name of the current script without extension
BASE=$(basename $0 .sh)

# run the corresponding .py file
python $DIR/$BASE.py \
    --note "grid search of bs-lr-lambda" \
    --seed 42 \
    --train True \
    --num_epochs 150 \
    --batch_size 16,32,64 \
    --lr 0.01,0.001,0.0001 \
    --lambda 0.1,0.01,0.001,0.0001 \
    --metrics accuracy,AUROC,sensitivity,precision,F1,MCC \
    --num_workers 16 \