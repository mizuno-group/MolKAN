#!/bin/bash

# activate your python environment
source /opt/pip-env/bin/activate

DIR=$(dirname $0)

python $DIR/create_experiment.py \
    --template /workspace/ToxPred/MolKAN/molkan/experiments/MLP/TfVAE_enc/BACE/batch_lr_lamdba_scheduler_free_1layer_exp_250124_02.py \
    --subdir MLP/TfVAE_enc/BACE \
    --subname "batch_lr_lamdba_scheduler_free_3layer" \