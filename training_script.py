import os

import torch
from functools import partial
from torchinfo import summary


import torch.nn as nn
from datetime import date
import argparse
import sys; sys.path.append("../")

from models.model_Baseline import BaselineNet
from models.model_CoreCNN_versions import CoreUnet_base, CoreUnet_large, CoreUnet_huge
from models.model_Mixer_versions import Mixer_base, Mixer_large, Mixer_huge
from models.model_LinearViT_versions import LinearViT_base, LinearViT_large, LinearViT_huge
from models.model_AutoEncoderViT_versions import AutoencoderViT_base, AutoencoderViT_large, AutoencoderViT_huge

from utils import data_protocol
from utils import load_data
from utils import training_loops

CNN_LIST = ['baseline_cnn', 'core_unet_base', 'core_unet_large', 'core_unet_huge']
MIXER_LIST = ['mixer_base', 'mixer_large', 'mixer_huge']
VIT_LIST = ['linear_vit_base', 'linear_vit_larger', 'linear_vit_huge',
            'autoencoder_vit_base', 'autoencoder_vit_large', 'autoencoder_vit_huge']

MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST


def get_models(model_name):
    if model_name == 'baseline_cnn':
        return BaselineNet(input_dim=args.input_channels, output_dim=args.output_channels)
    elif model_name == 'core_unet_base':
        return CoreUnet_base(input_dim=args.input_channels, output_dim=args.output_channels)
    elif model_name == 'core_unet_large':
        return CoreUnet_large(input_dim=args.input_channels, output_dim=args.output_channels)
    elif model_name == 'core_unet_huge':
        return CoreUnet_huge(input_dim=args.input_channels, output_dim=args.output_channels)
    elif model_name == 'mixer_base':
        return Mixer_base(chw=(args.input_channels, args.patch_size, args.patch_size),
                          output_dim=args.output_channels)
    elif model_name == 'mixer_large':
        return Mixer_large(chw=(args.input_channels, args.patch_size, args.patch_size),
                           output_dim=args.output_channels)
    elif model_name == 'mixer_huge':
        return Mixer_huge(chw=(args.input_channels, args.patch_size, args.patch_size),
                          output_dim=args.output_channels)
    elif model_name == 'linear_vit_base':
        return LinearViT_base(chw=(args.input_channels, args.patch_size, args.patch_size),
                              output_dim=args.output_channels)
    elif model_name == 'linear_vit_large':
        return LinearViT_large(chw=(args.input_channels, args.patch_size, args.patch_size),
                               output_dim=args.output_channels)
    elif model_name == 'linear_vit_huge':
        return LinearViT_huge(chw=(args.input_channels, args.patch_size, args.patch_size),
                              output_dim=args.output_channels)
    elif model_name == 'autoencoder_vit_base':
        return AutoencoderViT_base(chw=(args.input_channels, args.patch_size, args.patch_size),
                                   output_dim=args.output_channels)
    elif model_name == 'autoencoder_vit_large':
        return AutoencoderViT_large(chw=(args.input_channels, args.patch_size, args.patch_size),
                                    output_dim=args.output_channels)
    elif model_name == 'autoencoder_vit_huge':
        return AutoencoderViT_huge(chw=(args.input_channels, args.patch_size, args.patch_size),
                                   output_dim=args.output_channels)


def get_args():
    parser = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser.add_argument('--experiment_name', type=str, default=f'{date.today().strftime("%d%m%Y")}_experiment',
                        help='Experiment folder name')
    parser.add_argument('--model', type=str, choices=MODEL_LIST, required=True,
                        help='Select appropriate model')
    parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Set training epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='set training loop patience for early stopping')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=[None, 'reduce_on_plateau', 'cosine_annealing'], help='select learning rate scheduler')
    parser.add_argument('--warmup', type=bool, default=False, help='Enables linear 5 epoch warmup scheduler')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='select training device')
    parser.add_argument('--num_workers', type=int, default=0, help='set number of workers')
    parser.add_argument('--vis_val', type=bool, default=True, help='enable saving of intermediate visualization plots')
    parser.add_argument('--downstream_task', type=str, choices=['roads', 'building', 'lc'], required=True,
                        help='select downstream task')
    parser.add_argument('--input_channels', type=int, required=False, default=10, help='Define Number of input channels')
    parser.add_argument('--output_channels', type=int, required=True, default=1, help='Define Number of output channels')
    parser.add_argument('--patch_size', type=int, required=True, default=128, help='Define input patch size')
    parser.add_argument('--regions', type=list, default=None, help='select regions to be included',
                        choices=[None, 'denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                                 'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                                 'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1'])
    parser.add_argument('--n_shot', type=int, default=None,
                        help='Loads n-samples of data from specified geographic regions')
    parser.add_argument('--split_ratio', type=float, default=None,
                        help='Loads a percentage of the data from specified geographic regions.')
    parser.add_argument('--augmentations', type=bool, default=False, help='enables augmentations')
    return parser


if __name__ == "__main__":

    parser = get_args()
    args = parser.parse_args()
    init_lr = args.lr

    assert not(args.n_shot == None) or not(args.split_ratio == None), 'Please define data partition protocol!'
    assert isinstance(args.n_shot, int) ^ isinstance(args.split_ratio, float), 'n_shot cannot be used with split_ratio!'
    if (args.downstream_task == 'lc'):
        assert not(args.output_channels == 1), 'land cover task should have more than 1 output channels'

    if args.downstream_task == 'lc':
        lc = True
    else:
        lc = False

    model = get_models(args.model)

    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{args.experiment_name}/{date.today().strftime("%d%m%Y")}_{NAME}_{args.downstream_task}'
    if args.lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{args.experiment_name}/{date.today().strftime("%d%m%Y")}_{NAME}_{args.downstream_task}_{ args.lr_scheduler}'
        if args.lr_scheduler == 'reduce_on_plateau':
            args.lr = args.lr / 100000 # for warmup start

    if isinstance(args.n_shot, int):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{args.n_shot}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                                        dst='/phileo_data/downstream/downstream_datasets_nshot/',
                                                                        n=args.n_shot,
                                                                        regions=args.regions,
                                                                        y=args.downstream_task,
                                                                        resample=False)

    elif isinstance(args.split_ratio, float):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{args.split_ratio}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_split('/phileo_data/downstream/downstream_dataset_patches_np/',
                                                                      split_percentage=args.split_ratio,
                                                                      regions=args.regions,
                                                                      y=args.downstream_task)

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

    model_summary = summary(model, input_size=(args.batch_size, args.input_channels,  args.patch_size,  args.patch_size),)

    if args.model in (CNN_LIST + MIXER_LIST):
        if args.downstream_task == 'roads' or args.downstream_task == 'roads':
            trainer = training_loops.TrainBase(epochs=args.epochs, lr=args.lr, model=model, device=args.device,
                                               lr_scheduler=args.lr_scheduler, warmup=args.warmup, train_loader=dl_train,
                                               val_loader=dl_val, test_loader=dl_test, name=NAME,
                                               out_folder=OUTPUT_FOLDER,)
        elif args.downstream_task == 'lc':
            trainer = training_loops.TrainLandCover(epochs=args.epochs, lr=args.lr, model=model, device=args.device,
                                                       lr_scheduler=args.lr_scheduler, warmup=args.warmup, train_loader=dl_train,
                                                       val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                       out_folder=OUTPUT_FOLDER,)

    elif args.model in VIT_LIST:
        if args.downstream_task == 'roads' or args.downstream_task == 'roads':
            trainer = training_loops.TrainViT(epochs=args.epochs, lr=args.lr, model=model, device=args.device,
                                              lr_scheduler=args.lr_scheduler, warmup=args.warmup, train_loader=dl_train,
                                              val_loader=dl_val, test_loader=dl_test, name=NAME,
                                              out_folder=OUTPUT_FOLDER,)

        elif args.downstream_task == 'lc':
            trainer = training_loops.TrainViTLandCover(epochs=args.epochs, lr=args.lr, model=model, device=args.device,
                                                       lr_scheduler=args.lr_scheduler, warmup=args.warmup, train_loader=dl_train,
                                                       val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                       out_folder=OUTPUT_FOLDER,)


    trainer.train()
    trainer.test()
    trainer.save_info(model_summary=model_summary, n_shot=args.n_shot, p_split=args.split_ratio, warmup=args.warmup,
                      lr=init_lr)


