import os

import torch
import torchmetrics
from functools import partial


import torch.nn as nn
from datetime import date
import argparse
import sys; sys.path.append("../")

from models.model_CoreCNN import CoreUnet
from models.model_LinearViT import LinearViT
from models.model_AutoEncoderViT import AutoencoderViT

from utils import data_protocol
from utils import load_data
from utils import training_loops

def get_args():
    parser = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser.add_argument('--experiment_name', type=str, default=f'experiment_{date.today().strftime("%d%m%Y")}')
    parser.add_argument('--model', type=str, choices=['CoreUNET', 'LinearViT', 'AutoEncoderViT'], required=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--lr_scheduler', type=str, default=None, choices=[None, 'reduce_on_plateau', 'cosine_annealing'])
    parser.add_argument('--warmup', type=str, default=False)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--vis_val', type=bool, default=True)
    parser.add_argument('--downstream_task', type=str, choices=['roads', 'building', 'lc'], required=True)
    parser.add_argument('--regions', type=list, default=None)
    parser.add_argument('--n_shot', type=int, default=None)
    parser.add_argument('--split_ratio', type=float, default=None)
    parser.add_argument('--augmentations', type=bool, default=False)
    return parser


if __name__ == "__main__":

    parser = get_args()
    args = parser.parse_args()

    assert not(args.n_shot == None) and not(args.split_ratio == None), 'Please define data partition protocol!'
    assert not isinstance(args.n_shot, int) and not isinstance(args.split_ratio, float), 'n_shot cannot be used with split_ratio!'

    if args.downstream_task == 'lc':
        lc = True
        out_chans = 11
    else:
        lc = False
        out_chans = 1

    if args.model == 'CoreUNET':
        model = CoreUnet(output_dim=out_chans)
    elif args.model == 'LinearViT':
        model = LinearViT(out_chans=out_chans)
    elif args.model == 'AutoEncoderViT':
        model = AutoencoderViT(out_chans=out_chans)

    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{args.experiment_name}/{date.today().strftime("%d%m%Y")}_{NAME}_{args.downstream_task}'
    if args.lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{args.downstream_task}_{ args.lr_scheduler}'
        if args.lr_scheduler == 'reduce_on_plateau':
            LEARNING_RATE = args.lr / 100000 # for warmup start

    if isinstance(args.n_shot, int):
        x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                                        dst='/phileo_data/downstream/downstream_datasets_nshot/',
                                                                        n=args.n_shot,
                                                                        y=args.downstream_task)

    elif isinstance(args.split_ratio, float):

        x_test, y_test = data_protocol.get_testset(folder='/phileo_data/downstream/downstream_dataset_patches_np/',
                                                   y=args.downstream_task)

    dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                    with_augmentations=args.augmentations,
                                                    num_workers=args.num_workers,
                                                    batch_size=args.batch_size,
                                                    encoder_only=False,
                                                    land_cover=lc,
                                                    device=args.device
                                                    )

    obj = training_loops.TrainViTLandCover(epochs=args.epochs, lr=args.lr, model=model, device=args.device,
                                           lr_scheduler=args.lr_scheduler, train_loader=dl_train,
                                           val_loader=dl_val, test_loader=dl_test, name=NAME,
                                           out_folder=OUTPUT_FOLDER,)

    obj.train()
    obj.test()
    obj.save_info()


