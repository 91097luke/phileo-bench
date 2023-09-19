import os

import torch
import torchmetrics
from functools import partial

import torch.nn as nn
from datetime import date

import sys; sys.path.append("../")

from models.model_CoreCNN import CoreUnet
from models.model_LinearViT import LinearViT
from models.model_AutoEncoderViT import AutoencoderViT

from utils import data_protocol
from utils import load_data
from utils import training_loops


if __name__ == "__main__":
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 250
    BATCH_SIZE = 16
    num_workers = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y = 'lc'

    if y == 'lc':
        lc = True
    else:
        lc = False

    model = LinearViT(out_chans=11)
    lr_scheduler = 'reduce_on_plateau' # None, 'reduce_on_plateau', 'cosine_annealing'

    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{y}'
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{y}_{lr_scheduler}'
        if lr_scheduler == 'reduce_on_plateau':
            LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start



    x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                                    dst='/phileo_data/downstream/downstream_datasets_nshot/',
                                                                    n=10,
                                                                    y=y)

    x_test, y_test = data_protocol.get_testset(folder='/phileo_data/downstream/downstream_dataset_patches_np/',
                                               y=y)

    dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                    with_augmentations=False,
                                                    num_workers=num_workers,
                                                    batch_size=BATCH_SIZE,
                                                    encoder_only=False,
                                                    land_cover=lc,
                                                    device=device
                                                    )

    obj = training_loops.TrainViTLandCover(epochs=NUM_EPOCHS, lr=LEARNING_RATE, model=model, device=device,
                                       lr_scheduler=lr_scheduler, train_loader=dl_train,
                                       val_loader=dl_val, test_loader=dl_test, name=NAME,
                                       out_folder=OUTPUT_FOLDER,)

    obj.train()
    obj.test()
    obj.save_info()


