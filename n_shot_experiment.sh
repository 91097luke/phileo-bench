#!/bin/bash
labels=('roads' 'building' 'lc')

MODEL='baseline_cnn'
LR=0.001
BATCH_SIZE=16
EPOCHS=250
LR_SCHEDULER='reduce_on_plateau'
WARMUP=True
NUM_WORKERS=4
VIS_VAL=True
AUGMENTATIONS=True
PATCH_SIZE=128

s='lc'


for label in "${labels[@]}";
do
  experiment_name=$MODEL"_"$label"_n_split"
  if test "$label" = "$s";
  then
    output_channels=11
  else
    output_channels=1
  fi

  for n_shot in 10 50 100 500 1000 1500 2000 5000 10000 20000 25000 30000;
    do
      python training_script.py --experiment_name=$experiment_name --model=$MODEL --lr=$LR --batch_size=$BATCH_SIZE --epochs=$EPOCHS --lr_scheduler=$LR_SCHEDULER --warmup=$WARMUP --vis_val=$VIS_VAL --augmentations=$AUGMENTATIONS --patch_size=$PATCH_SIZE --n_shot=$n_shot --downstream_task=$label --output_channels=$output_channels
    done
done
