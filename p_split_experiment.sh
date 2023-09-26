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

p_splits=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
for label in "${labels[@]}";
do
  experiment_name=$MODEL"_"$label"_p_split"
  if test "$label" = "$s";
  then
    output_channels=11
  else
    output_channels=1
  fi

  for p_split in "${p_splits[@]}";
    do
      python training_script.py --experiment_name=$experiment_name --model=$MODEL --lr=$LR --batch_size=$BATCH_SIZE --epochs=$EPOCHS --lr_scheduler=$LR_SCHEDULER --warmup=$WARMUP --vis_val=$VIS_VAL --augmentations=$AUGMENTATIONS --patch_size=$PATCH_SIZE --split_ratio=$p_split --downstream_task=$label --output_channels=$output_channels
    done
done
