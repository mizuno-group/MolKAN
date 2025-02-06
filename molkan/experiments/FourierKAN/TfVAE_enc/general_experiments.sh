source /opt/torch250/bin/activate

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

seed=(42)
layer=(2 3)
scheduler_free=("True")

for s in "${seed[@]}"; do
    for l in "${layer[@]}"; do
        for scl in "${scheduler_free[@]}"; do
            
            if [ "$scl" == "True" ]; then
                optim="RAdamScheduleFree"
            else
                optim="AdamW"
            fi
            
            python $DIR/$BASE.py \
                --note hERG-FourierKAN${l}layer_big-${optim}-seed${s} \
                --datasets hERG \
                --label_columns Y \
                --results_dirname ${today}_grids_batch_lr_${optim}_${l}layer_big_seed${s} \
                --seed $s \
                --train True \
                --mode classification \
                --num_epochs 500 \
                --layer $l \
                --num_grids 1,2,4,6,8 \
                --batch_size 4,16,64,256,1024 \
                --lr 5.0e-3,3.0e-3,5.0e-4,3.0e-4 \
                --lambda 0 \
                --scheduler_free $scl \
                --metrics accuracy,AUROC,sensitivity,precision,F1,MCC \
                --num_workers 9 \

        done
    done
done