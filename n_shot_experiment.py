import os
import numpy
import torch
from datetime import date

import training_script

labels=['roads', 'building', 'lc']

MODEL='baseline_cnn'
LR=0.001
BATCH_SIZE=16
EPOCHS=250
LR_SCHEDULER='reduce_on_plateau'
WARMUP= True
NUM_WORKERS=4
VIS_VAL=True
AUGMENTATIONS=False
INPUT_SIZE=128
n_shots = [1, 10, 100, 1000, 2500, 5000, 7500, 10000]


for label in labels:
    experiment_name=f'trained_models/{date.today().strftime("%d%m%Y")}_{MODEL}_{label}_p_split'
    if label == 'lc':
        output_channels = 11
    else:
        output_channels = 1

    for n_shot in n_shots:
        args = {'experiment_name': experiment_name, 'model_name': MODEL, 'lr': LR, 'batch_size': BATCH_SIZE,
                'epochs': EPOCHS, 'early_stop': 25, 'lr_scheduler': LR_SCHEDULER, 'warmup': WARMUP,
                'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                'num_workers': NUM_WORKERS, 'vis_val': VIS_VAL, 'downstream_task': label, 'input_channels': 10,
                'input_size': INPUT_SIZE,
                'output_channels': output_channels, 'regions': None, 'n_shot': n_shot, 'split_ratio': None,
                'augmentations': AUGMENTATIONS}

        training_script.main(**args)

