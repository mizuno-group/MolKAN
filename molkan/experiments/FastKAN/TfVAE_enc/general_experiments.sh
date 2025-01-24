#!/bin/bash

source /opt/pip-env/bin/activate
DIR=$(dirname $0)
BASE=$(basename $0 .sh)
today=$(date +'%y%m%d')

# [Datasets list] choose from these !
# BACE
# ClinTox
# CYP2C9
# CYP3A4
# hERG
# LD50
# SIDER
# Tox21
# ToxCast

seed=(42 222)
layer=(1 2 3)
scheduler_free=("False" "True")

for s in "${seed[@]}"; do
    for l in "${layer[@]}"; do
        for scl in "${scheduler_free[@]}"; do
            
            if [ "$scl" == "False" ]; then
                optim="RAdamScheduleFree"
            else
                optim="AdamW"
            fi
            
            python $DIR/$BASE.py \
                --note hERG-KAN${l}layer-${optim}-seed${s} \
                --datasets hERG \
                --label_columns Y \
                --results_dirname ${today}_grids_batch_lr_lamdba_${optim}_${l}layer_seed${s} \
                --seed $s \
                --train True \
                --mode classification \
                --num_epochs 300 \
                --layer $l \
                --num_grids 8,16,32 \
                --batch_size 4,16,64,256,1024 \
                --lr 1.0e-2,1.0e-3,1.0e-4 \
                --lambda 1.0e-2,1.0e-3,1.0e-4,0 \
                --scheduler_free $scl \
                --metrics accuracy,AUROC,sensitivity,precision,F1,MCC \
                --num_workers 16
        
        done
    done
done