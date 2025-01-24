#!/bin/bash

# activate your python environment
source /opt/pip-env/bin/activate

# get the directory of the current script
DIR=$(dirname $0)

# get the base name of the current script without extension
BASE=$(basename $0 .sh)

# run the corresponding .py file
python $DIR/$BASE.py \
    --note "BACE-KAN3layer-RAdamSchedulerFree" \
    --seed 222 \
    --train True \
    --num_epochs 300 \
    --num_grids 8,16,32 \
    --batch_size 4,16,64,256,1024 \
    --lr 1.0e-2,1.0e-3,1.0e-4 \
    --lambda 1.0e-2,1.0e-3,1.0e-4 \
    --metrics accuracy,AUROC,sensitivity,precision,F1,MCC \
    --num_workers 16 \