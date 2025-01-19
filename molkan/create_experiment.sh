#!/bin/bash

# activate your python environment
source ~/python_env/chemoinfo/bin/activate

DIR=$(dirname $0)

python $DIR/create_experiment.py \
    --template $DIR/experiments/first_test/test_exp_250119_01.py \
    --subdir BACE \
    --subname "batch_lr_lamdba" \