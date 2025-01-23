#!/bin/bash

# activate your python environment
source /opt/pip-env/bin/activate

DIR=$(dirname $0)

python $DIR/create_experiment.py \
    --template /workspace/ToxPred/MolKAN/molkan/experiments/MLP/TfVAE_enc/bace_batch_lr_lamdba_scheduler_free_2layer_exp_250123_01.py \
    --subdir MLP/TfVAE_enc \
    --subname "bace_batch_lr_lamdba_scheduler_free_3layer" \