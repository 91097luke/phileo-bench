import os

import torch
from functools import partial
from torchinfo import summary


import torch.nn as nn
from datetime import date
import argparse
import sys; sys.path.append("../")

from models.model_Baseline import BaselineNet
from models.model_CoreCNN_versions import CoreUnet_nano, CoreUnet_tiny, CoreUnet_base, CoreUnet_large, CoreUnet_huge, Core_nano
from models.model_Mixer_versions import Mixer_nano, Mixer_tiny, Mixer_base, Mixer_large, Mixer_huge
from models.model_LinearViT_versions import LinearViT_base, LinearViT_large, LinearViT_huge
from models.model_AutoEncoderViT_versions import AutoencoderViT_base, AutoencoderViT_large, AutoencoderViT_huge
from models.model_GeoAwarePretrained import MixerGeoPretrained, get_mixer_kwargs, get_core_encoder_kwargs, CoreEncoderGeoPretrained, CoreEncoderGeoPretrained_combined, CoreEncoderGeoAutoEncoder
from models.model_GeoAwarePretrained_classifier import CoreEncoderGeoPretrained_Classifier
from models.model_AutoEncoderViTPretrained import vit_cnn, vit_cnn_gc, vit_large, get_core_decoder_kwargs
from models.model_AutoEncoderViTPretrained_classifier import vit_cnn_classifier, vit_cnn_gc_classifier
from models.model_CoreVAE import CoreVAE_nano
from models.model_SatMAE import satmae_vit_cnn
from models.models_Prithvi import prithvi
from models.model_Seco import seasonal_contrast
from models.model_Resnet50 import resnet

from utils import data_protocol
from utils import load_data
from utils import training_loops
from utils.training_utils import read_yaml
torch.manual_seed(123456)
CNN_LIST = ['baseline_cnn', 'core_unet_nano','core_unet_tiny','core_unet_base', 'core_unet_large', 'core_unet_huge',
            'core_vae_nano', 'resnet_imagenet', 'resnet', 'core_encoder_nano']

VIT_CNN_LIST = ['vit_cnn_base']

MIXER_LIST = ['mixer_nano', 'mixer_tiny', 'mixer_base', 'mixer_large', 'mixer_huge']

VIT_LIST = ['linear_vit_base', 'linear_vit_larger', 'linear_vit_huge',
            'autoencoder_vit_base', 'autoencoder_vit_large', 'autoencoder_vit_huge']

CNN_PRETRAINED_LIST = ['GeoAware_core_nano', 'GeoAware_core_tiny', 'GeoAware_mixer_nano', 'GeoAware_mixer_tiny',
                       'GeoAware_contrastive_core_nano', 'GeoAware_mh_pred_core_nano', 'GeoAware_combined_core_nano',
                       'GeoAware_core_autoencoder_nano', 'seasonal_contrast',
                       'GeoAware_core_nano_classifier', 'GeoAware_contrastive_core_nano_classifier',
                       'GeoAware_mh_pred_core_nano_classifier', 'seasonal_contrast_classifier'
                       ]

VIT_CNN_PRETRAINED_LIST = ['prithvi', 'vit_cnn', 'vit_cnn_gc', 'SatMAE', 'SatMAE_classifier', 'vit_cnn_gc_classifier',
                           'vit_cnn_classifier', 'prithvi_classifier']

MODELS_224 = ['seasonal_contrast', 'resnet_imagenet', 'resnet']
MODELS_224_r30 = ['prithvi']

MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST + CNN_PRETRAINED_LIST + VIT_CNN_LIST + VIT_CNN_PRETRAINED_LIST


def get_trainer(model_name, downstream_task, epochs, lr, model, device, lr_scheduler, warmup, early_stop, dl_train,
                dl_val, dl_test, NAME, OUTPUT_FOLDER, vis_val):

    if model_name in (CNN_LIST + MIXER_LIST + VIT_CNN_LIST + CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST):
        if downstream_task == 'roads' or downstream_task == 'building':
            trainer = training_loops.TrainBase(epochs=epochs, lr=lr, model=model, device=device,
                                               lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                               train_loader=dl_train,
                                               val_loader=dl_val, test_loader=dl_test, name=NAME,
                                               out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)
        elif downstream_task == 'lc':
            trainer = training_loops.TrainLandCover(epochs=epochs, lr=lr, model=model, device=device,
                                                    lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                    train_loader=dl_train,
                                                    val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                    out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)
        elif downstream_task == 'building_classification':
            trainer = training_loops.TrainClassificationBuildings(epochs=epochs, lr=lr, model=model, device=device,
                                                                  lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                                  train_loader=dl_train,
                                                                  val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                                  out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)

        elif downstream_task == 'lc_classification':
            trainer = training_loops.TrainClassificationLC(epochs=epochs, lr=lr, model=model, device=device,
                                                           lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                           train_loader=dl_train,
                                                           val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                           out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)

        elif downstream_task == 'raods_classification':
            trainer = training_loops.TrainClassificationRoads(epochs=epochs, lr=lr, model=model, device=device,
                                                           lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                           train_loader=dl_train,
                                                           val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                           out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)

    elif model_name in (VIT_LIST):
        if downstream_task == 'roads' or downstream_task == 'building':
            trainer = training_loops.TrainViT(epochs=epochs, lr=lr, model=model, device=device,
                                              lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop, train_loader=dl_train,
                                              val_loader=dl_val, test_loader=dl_test, name=NAME,
                                              out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)

        elif downstream_task == 'lc':
            trainer = training_loops.TrainViTLandCover(epochs=epochs, lr=lr, model=model, device=device,
                                                       lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                                       train_loader=dl_train,
                                                       val_loader=dl_val, test_loader=dl_test, name=NAME,
                                                       out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)

    if model_name == 'core_vae_nano':
        trainer = training_loops.TrainVAE(epochs=epochs, lr=lr, model=model, device=device,
                                          lr_scheduler=lr_scheduler, warmup=warmup, early_stop=early_stop,
                                          train_loader=dl_train,
                                          val_loader=dl_val, test_loader=dl_test, name=NAME,
                                          out_folder=OUTPUT_FOLDER, visualise_validation=vis_val)

    return trainer


def get_models(model_name, input_channels, output_channels, input_size):
    if model_name == 'baseline_cnn':
        return BaselineNet(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_nano':
        return CoreUnet_nano(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_encoder_nano':
        return Core_nano(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_tiny':
        return CoreUnet_tiny(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_base':
        return CoreUnet_base(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_large':
        return CoreUnet_large(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_huge':
        return CoreUnet_huge(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'mixer_nano':
        return Mixer_nano(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_tiny':
        return Mixer_tiny(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_base':
        return Mixer_base(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'mixer_large':
        return Mixer_large(chw=(input_channels, input_size, input_size),
                           output_dim=output_channels)
    elif model_name == 'mixer_huge':
        return Mixer_huge(chw=(input_channels, input_size, input_size),
                          output_dim=output_channels)
    elif model_name == 'linear_vit_base':
        return LinearViT_base(chw=(input_channels, input_size, input_size),
                              output_dim=output_channels)
    elif model_name == 'linear_vit_large':
        return LinearViT_large(chw=(input_channels, input_size, input_size),
                               output_dim=output_channels)
    elif model_name == 'linear_vit_huge':
        return LinearViT_huge(chw=(input_channels, input_size, input_size),
                              output_dim=output_channels)
    elif model_name == 'autoencoder_vit_base':
        return AutoencoderViT_base(chw=(input_channels, input_size, input_size),
                                   output_dim=output_channels)
    elif model_name == 'autoencoder_vit_large':
        return AutoencoderViT_large(chw=(input_channels, input_size, input_size),
                                    output_dim=output_channels)
    elif model_name == 'autoencoder_vit_huge':
        return AutoencoderViT_huge(chw=(input_channels, input_size, input_size),
                                   output_dim=output_channels)
    elif model_name == 'core_vae_nano':
        return CoreVAE_nano(input_dim=input_channels, output_dim=10)

    elif model_name == 'vit_cnn_base':
        return vit_large(chw=(input_channels, input_size, input_size),
                         output_dim=output_channels)
    elif model_name == 'resnet_imagenet':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=True, **resnet_kwargs)
    elif model_name == 'resnet_imagenet_classifier':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=True, classifier=True, **resnet_kwargs)
    elif model_name == 'resnet':
        resnet_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return resnet(imagenet_weights=False, **resnet_kwargs)


def get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=None, freeze=False, device='cuda'):
    
    test_input = torch.rand((2,input_channels,input_size,input_size))

    if (model_name == 'GeoAware_core_nano' or model_name == 'GeoAware_contrastive_core_nano' or
            model_name == 'GeoAware_mh_pred_core_nano'):

        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if (model_name == 'GeoAware_core_nano_classifier' or model_name == 'GeoAware_contrastive_core_nano_classifier' or
            model_name == 'GeoAware_mh_pred_core_nano_classifier'):

        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=False)
        model = CoreEncoderGeoPretrained_Classifier(checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_core_autoencoder_nano':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoAutoEncoder(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model

    if model_name == 'GeoAware_combined_core_nano':
        sd_1 = torch.load(path_model_weights[0])
        sd_2 = torch.load(path_model_weights[1])
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano')
        model = CoreEncoderGeoPretrained_combined(output_channels, checkpoint_1=sd_1, checkpoint_2=sd_2,
                                                  core_encoder_kwargs=core_kwargs)

        model(test_input)
        return model
    
    if model_name == 'GeoAware_core_tiny':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_tiny', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
        model(test_input)
        return model
    
    if model_name == 'GeoAware_mixer_nano':
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels,input_size,input_size),output_dim=output_channels, mixer_size='mixer_nano')
        model =  MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        model(test_input)
        return model 
    
    if model_name == 'GeoAware_mixer_tiny':
        sd = torch.load(path_model_weights)
        mixer_kwargs = get_mixer_kwargs(chw=(input_channels,input_size,input_size),output_dim=output_channels, mixer_size='mixer_tiny')
        model = MixerGeoPretrained(output_dim=output_channels, checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=freeze)
        model(test_input)
        return model 

    elif model_name == 'SatMAE':
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels,
                              checkpoint=sd, freeze_body=freeze, classifier=False, **satmae_kwargs)

    elif model_name == 'SatMAE_classifier':
        sd = torch.load(path_model_weights)
        satmae_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return satmae_vit_cnn(img_size=96, patch_size=8, in_chans=input_channels,
                              checkpoint=sd, freeze_body=freeze, classifier=True, **satmae_kwargs)

    elif model_name == 'prithvi':
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return prithvi(checkpoint=sd, freeze_body=freeze, **prithvi_kwargs)

    elif model_name == 'prithvi_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        prithvi_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return prithvi(checkpoint=sd, freeze_body=freeze, classifier=True, **prithvi_kwargs)

    elif model_name == 'vit_cnn':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        return vit_cnn_classifier(checkpoint=sd, freeze_body=freeze, output_dim=output_channels)

    elif model_name == 'vit_cnn_gc':
        sd = torch.load(path_model_weights, map_location=device)
        vit_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return vit_cnn_gc(checkpoint=sd, freeze_body=freeze, **vit_kwargs)

    elif model_name == 'vit_cnn_gc_classifier':
        sd = torch.load(path_model_weights, map_location=device)
        return vit_cnn_gc_classifier(checkpoint=sd, freeze_body=freeze, output_dim=output_channels)


    elif model_name == 'seasonal_contrast':
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return seasonal_contrast(checkpoint=path_model_weights, output_dim=output_channels, freeze_body=freeze,
                                 **seco_kwargs)

    elif model_name == 'seasonal_contrast_classifier':
        seco_kwargs = get_core_decoder_kwargs(output_dim=output_channels, core_size='core_nano')
        return seasonal_contrast(checkpoint=path_model_weights, output_dim=output_channels, freeze_body=freeze, classifier=True,
                                 **seco_kwargs)


def get_args():
    parser_yaml = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser_yaml.add_argument('--read_yaml', type=str, help='take parameters from yaml path', default=None)


    parser = argparse.ArgumentParser(description='Experiment TestBed for Phi-Leo Foundation Model Project')
    parser.add_argument('--experiment_name', type=str, default=f'{date.today().strftime("%d%m%Y")}_experiment',
                        help='Experiment folder name')
    parser.add_argument('--model_name', type=str, choices=MODEL_LIST, required=True,
                        help='Select appropriate model')
    parser.add_argument('--lr', type=float, default=0.001, help='Set learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Set batch size')
    parser.add_argument('--epochs', type=int, default=250, help='Set training epochs')
    parser.add_argument('--early_stop', type=int, default=50, help='set training loop patience for early stopping')
    parser.add_argument('--lr_scheduler', type=str, default=None,
                        choices=[None, 'reduce_on_plateau', 'cosine_annealing'], help='select learning rate scheduler')
    parser.add_argument('--warmup', action="store_true", help='Enables linear 5 epoch warmup scheduler')
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        help='select training device')
    parser.add_argument('--num_workers', type=int, default=0, help='set number of workers')
    parser.add_argument('--vis_val', action="store_true", help='enable saving of intermediate visualization plots')
    parser.add_argument('--downstream_task', type=str, choices=['roads', 'building', 'lc'], required=True,
                        help='select downstream task')
    parser.add_argument('--input_channels', type=int, required=False, default=10, help='Define Number of input channels')
    parser.add_argument('--input_size', type=int, required=True, default=128, help='Define input size')
    parser.add_argument('--output_channels', type=int, required=True, default=1, help='Define Number of output channels')

    parser.add_argument('--regions', type=list, default=None, help='select regions to be included',
                        choices=[None, 'denmark-1', 'denmark-2', 'east-africa', 'egypt-1', 'eq-guinea', 'europe', 'ghana-1',
                                 'isreal-1', 'isreal-2', 'japan', 'nigeria', 'north-america', 'senegal', 'south-america',
                                 'tanzania-1', 'tanzania-2', 'tanzania-3', 'tanzania-4', 'tanzania-5', 'uganda-1'])
    parser.add_argument('--n_shot', type=int, default=None,
                        help='Loads n-samples of data from specified geographic regions')
    parser.add_argument('--split_ratio', type=float, default=None,
                        help='Loads a percentage of the data from specified geographic regions.')
    parser.add_argument('--augmentations', action="store_true", help='enables augmentations')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained weights')
    parser.add_argument('--freeze_pretrained', action="store_true", help='freeze pretrained model weights')

    return parser,parser_yaml


def main(downstream_task:str, experiment_name:str, model_name:str, augmentations:bool=False, batch_size:int=16, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
         early_stop:int=25, epochs:int=250, input_channels:int=10, input_size:int=128, lr:float=0.001, lr_scheduler:str=None,
         n_shot:int=None, num_workers:int=4, output_channels:int=1, regions:list=None, split_ratio:float=0.1, vis_val=True, warmup=False, pretrained_model_path=None, freeze_pretrained=None):

    init_lr = lr
    # device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_default_device(device)
    print('DEVICE', device)

    assert not (n_shot == None) or not (split_ratio == None), 'Please define data partition protocol!'
    assert isinstance(n_shot, int) ^ isinstance(split_ratio, float), 'n_shot cannot be used with split_ratio!'
    if (downstream_task == 'lc') or (downstream_task == 'lc_classification'):
        assert not (output_channels == 1), 'land cover tasks should have 11 output channels'

    if (downstream_task == 'roads') or (downstream_task == 'building'):
        assert output_channels == 1, 'road and building density estimation tasks should have a single output channel'

    if downstream_task == 'building_classification':
        assert output_channels == 5, 'building classification tasks should have a 5 output channels'

    if downstream_task == 'roads_classification':
        assert output_channels == 2, 'road classification tasks should have a 5 output channels'

    if pretrained_model_path is not None:
        print(model_name)
        assert model_name in (CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST), f"Pretrained weights were given but model {model_name} not found in list of pretrained models: {(CNN_PRETRAINED_LIST + VIT_CNN_PRETRAINED_LIST)}"
        assert freeze_pretrained is not None, f"When supplying a pretrained model 'freeze_pretrained' must be either True or False"
        model = get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=pretrained_model_path, freeze=freeze_pretrained)
        if model_name == 'GeoAware_contrastive_core_nano' or model_name == 'GeoAware_contrastive_core_nano_classifier':
            NAME = model.__class__.__name__ +'_contrastive_frozen' if freeze_pretrained else model.__class__.__name__ +'_contrastive_unfrozen'
        elif model_name == 'GeoAware_mh_pred_core_nano' or model_name == 'GeoAware_mh_pred_core_nano_classifier':
            NAME = model.__class__.__name__ +'_mh_pred_frozen' if freeze_pretrained else model.__class__.__name__ +'_mh_pred_unfrozen'
        else:
            NAME = model.__class__.__name__ + '_frozen' if freeze_pretrained else model.__class__.__name__ + '_unfrozen'

    else:
        if freeze_pretrained:
            print(f"Ignoring freeze_pretrained set to {freeze_pretrained} as no pretrained model was supplied")
        model = get_models(model_name, input_channels, output_channels, input_size)
        NAME = model.__class__.__name__

    OUTPUT_FOLDER = f'/home/phimultigpu/phileo_NFS/phileo_data/experiments/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}'
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'/home/phimultigpu/phileo_NFS/phileo_data/experiments/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}_{lr_scheduler}'

    if warmup:
        lr = lr / 100000  # for warmup start

    dataset_folder = '/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np/'
    dataset_name = '128_10m'
    if model_name in MODELS_224_r30:
        dataset_folder = '/phileo_data/downstream/downstream_dataset_patches_np_HLS/'
    if model_name in MODELS_224:
        dataset_folder = '/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_dataset_patches_np_HLS/'
        dataset_name = '224_30m'
    if model_name in MODELS_224:
        dataset_folder = '/phileo_data/downstream/downstream_dataset_patches_np_224/'
        dataset_name = '224_10m'

    if downstream_task == 'pretraining':
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_minifoundation(
            folder='/home/phimultigpu/phileo_NFS/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/',
            y='geo')

        downstream_task = 'geo'

    elif isinstance(n_shot, int):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{n_shot}'

        x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot_memmapped(
            folder=dataset_folder,
            dst='/home/phimultigpu/phileo_NFS/phileo_data/downstream/downstream_datasets_nshot/',
            n=n_shot,
            regions=regions,
            y=downstream_task,
            data_selection='create',
            name=dataset_name)

    elif isinstance(split_ratio, float):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{split_ratio}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_split(
            dataset_folder,
            split_percentage=split_ratio,
            regions=regions,
            y=downstream_task)

    x_test, y_test = data_protocol.get_testset(folder=dataset_folder,
                                               y=downstream_task)

    dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                    with_augmentations=augmentations,
                                                    num_workers=num_workers,
                                                    batch_size=batch_size,
                                                    downstream_task=downstream_task,
                                                    model_name=model_name.split('_')[0],
                                                    device=device
                                                    )
    

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    if model_name == 'SatMAE' or model_name =='SatMAE_classifier':
        model_summary = summary(model,
                                input_size=(batch_size, input_channels, 96, 96), )

    elif model_name == 'prithvi':
        model_summary = summary(model,
                                input_size=(batch_size, 6, 224, 224), dtypes=[torch.float32])

    elif model_name in ['seasonal_contrast', 'resnet_imagenet', 'resnet']:
        model_summary = summary(model,
                                input_size=(batch_size, input_channels, 224, 224), )

    else:
        model_summary = summary(model, input_size=(batch_size, input_channels, input_size, input_size))

    trainer = get_trainer(model_name, downstream_task, epochs, lr, model, device, lr_scheduler, warmup, early_stop, dl_train,
                dl_val, dl_test, NAME, OUTPUT_FOLDER, vis_val)

    trainer.train()
    trainer.test()
    trainer.save_info(model_summary=model_summary, n_shot=n_shot, p_split=split_ratio, warmup=warmup,
                      lr=init_lr)

if __name__ == "__main__":

    parser, parser_yaml = get_args()
    args_yaml, remainder = parser_yaml.parse_known_args()
    
    if args_yaml.read_yaml is not None:
        print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
        args = read_yaml(args_yaml.read_yaml)
    else:
        args = parser.parse_args()

    main(**vars(args))

    # for downstream_task in ['lc', 'roads', 'building']:
    #     args['downstream_task'] = downstream_task
    #     for n_shot in [5, 10, 50, 500, 5000, 50000, 100000, 200000]:
    #         for model_name in ['core_unet_nano', 'GeoAware_contrasive_core_nano', 'GeoAware_core_nano']: #,'mixer_nano','baseline_cnn','linear_vit_base']:
    #             args['n_shot'] = n_shot
    #             args['model_name'] = model_name
    #
    #             if model_name == 'GeoAware_contrasive_core_nano':
    #                 args['pretrained_model_path'] = '/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt'
    #             elif model_name == 'GeoAware_contrasive_core_nano':
    #                 args['pretrained_model_path'] = '/phileo_data/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_19.pt'
    #             if model_name != 'core_unet_nano':
    #                for freeze in [True, False]:
    #                     args['freeze_pretrained'] = freeze
    #                     main(**vars(args))
    #             else:
    #                 args['pretrained_model_path'] = None
    #                 main(**vars(args))







