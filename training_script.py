# python training_script.py --read_yaml default_args.yml

# python training_script.py --read_yaml default_args.yml

# # python training_script.py --read_yaml default_args.yml      

import os

import torch
from functools import partial

from torchinfo import summary



import torch.nn as nn
from datetime import date

import argparse   



import sys; sys.path.append("../")             

from models.model_Baseline import BaselineNet 
from models.model_CoreCNN_versions import CoreUnet_nano, CoreUnet_tiny, CoreUnet_base, CoreUnet_large, CoreUnet_huge

from models.model_Mixer_versions import Mixer_nano, Mixer_tiny, Mixer_base, Mixer_large, Mixer_huge



from models.model_LinearViT_versions import LinearViT_base, LinearViT_large, LinearViT_huge
from models.model_AutoEncoderViT_versions import AutoencoderViT_base, AutoencoderViT_large, AutoencoderViT_huge

from models.model_GeoAwarePretrained import MixerGeoPretrained, get_mixer_kwargs, get_core_encoder_kwargs, CoreEncoderGeoPretrained, CoreEncoderGeoPretrained_combined

from models.model_AutoEncoderViTPretrained import AutoEncoderViTPretrained  



from utils import data_protocol  
from utils import load_data

from utils import training_loops
from utils.training_utils import read_yaml



torch.manual_seed(123456) 





CNN_LIST = ['baseline_cnn', 'core_unet_nano','core_unet_tiny','core_unet_base', 'core_unet_large', 'core_unet_huge']
MIXER_LIST = ['mixer_nano', 'mixer_tiny', 'mixer_base', 'mixer_large', 'mixer_huge']

VIT_LIST = ['linear_vit_base', 'linear_vit_larger', 'linear_vit_huge',
            'autoencoder_vit_base', 'autoencoder_vit_large', 'autoencoder_vit_huge'] 

CNN_PRETRAINED_LIST = ['GeoAware_core_nano', 'GeoAware_core_tiny', 'GeoAware_mixer_nano', 'GeoAware_mixer_tiny',
                       'GeoAware_contrastive_core_nano', 'GeoAware_basic_core_nano', 'GeoAware_combined_core_nano']

VIT_PRETRAINED_LIST = ['AutoEncoderVitPretrained']



MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST + CNN_PRETRAINED_LIST

def get_trainer(model_name, downstream_task, epochs, lr, model, device, lr_scheduler, warmup, early_stop, dl_train,
                dl_val, dl_test, NAME, OUTPUT_FOLDER, vis_val):

    if model_name in (CNN_LIST + MIXER_LIST + CNN_PRETRAINED_LIST):
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

    elif model_name in (VIT_LIST + VIT_PRETRAINED_LIST): 
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

    return trainer


def get_models(model_name, input_channels, output_channels, input_size): 
    if model_name == 'baseline_cnn':
        return BaselineNet(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_nano':
        return CoreUnet_nano(input_dim=input_channels, output_dim=output_channels)
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


def get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=None, freeze=False, device='cuda'):
    
    test_input = torch.rand((2,input_channels,input_size,input_size))

    
    
    if model_name == 'GeoAware_core_nano' or model_name == 'GeoAware_contrastive_core_nano' or model_name == 'GeoAware_basic_core_nano':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
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

    if model_name == 'AutoEncoderVitPretrained':
        sd = torch.load(path_model_weights, map_location=device)
        model = AutoEncoderViTPretrained(chw=(input_channels, input_size, input_size),
                                         output_dim=output_channels, checkpoint=sd, freeze_body=freeze)
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
    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    #torch.set_default_device(device) 
    print('DEVICE', device)

    
    
    assert not (n_shot == None) or not (split_ratio == None), 'Please define data partition protocol!'
    assert isinstance(n_shot, int) ^ isinstance(split_ratio, float), 'n_shot cannot be used with split_ratio!'
    if (downstream_task == 'lc'):
        assert not (output_channels == 1), 'land cover task should have more than 1 output channels'
        lc = True

    if (downstream_task == 'roads') or (downstream_task == 'building'):    
        assert output_channels == 1, 'regression type tasks should have a single output channel'  
        lc = False

    if pretrained_model_path is not None:        
        assert model_name in (CNN_PRETRAINED_LIST + VIT_PRETRAINED_LIST), f"Pretrained weights were given but model {model_name} not found in list of pretrained models: {(CNN_PRETRAINED_LIST + VIT_PRETRAINED_LIST)}"
        assert freeze_pretrained is not None, f"When supplying a pretrained model 'freeze_pretrained' must be either True or False"
        model = get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=pretrained_model_path, freeze=freeze_pretrained)
        
        
        
        #print(model)      

        #print(model_name)  

        
        
        if model_name == 'GeoAware_contrastive_core_nano':
            NAME = model.__class__.__name__ +'_contrastive_frozen' if freeze_pretrained else model.__class__.__name__ +'_contrastive_unfrozen'
        elif model_name == 'GeoAware_basic_core_nano':
            NAME = model.__class__.__name__ +'_basic_frozen' if freeze_pretrained else model.__class__.__name__ +'_basic_unfrozen'
        else:
            NAME = model.__class__.__name__ + '_frozen' if freeze_pretrained else model.__class__.__name__ + '_unfrozen'
        
        OUTPUT_FOLDER = f'/phileo_data/experiments/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}'
        if lr_scheduler is not None:
            OUTPUT_FOLDER = f'/phileo_data/experiments/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}_{lr_scheduler}'

        if isinstance(n_shot, int):
            OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{n_shot}'
            x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot_memmapped(
                '/phileo_data/downstream/downstream_dataset_patches_np/',
                dst='/phileo_data/downstream/downstream_datasets_nshot/',
                n=n_shot,
                regions=regions,
                y=downstream_task,
                data_selection='create') 

        elif isinstance(split_ratio, float):
            OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{split_ratio}'
            x_train, y_train, x_val, y_val = data_protocol.protocol_split(
                '/phileo_data/downstream/downstream_dataset_patches_np/',
                split_percentage=split_ratio,
                regions=regions,
                y=downstream_task)  

        x_test, y_test = data_protocol.get_testset(folder='/phileo_data/downstream/downstream_dataset_patches_np/',
                                                y=downstream_task) 

        dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                        with_augmentations=augmentations,
                                                        num_workers=num_workers,
                                                        batch_size=batch_size,
                                                        land_cover=lc,
                                                        device=device
                                                        )
    
    
    
        
        
        # add the external model       
        
        import os
        #import torch
        import matplotlib.pyplot as plt
        import numpy as np
        import rasterio
        import yaml
        from prithvi.Prithvi import MaskedAutoencoderViT

        NO_DATA = -9999
        NO_DATA_FLOAT = 0.0001
        PERCENTILES = (0.1, 99.9)


        # ### Define some functions for visualization 

        # In[3]:


        def load_raster(path, crop=None):
            with rasterio.open(path) as src:
                img = src.read()

                # load first 6 bands
                img = img[:6]

                img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
                if crop:
                    img = img[:, -crop[0]:, -crop[1]:]
            return img

        def enhance_raster_for_visualization(raster, ref_img=None):
            if ref_img is None:
                ref_img = raster
            channels = []
            for channel in range(raster.shape[0]):
                valid_mask = np.ones_like(ref_img[channel], dtype=bool)
                valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
                mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
                normalized_raster = (raster[channel] - mins) / (maxs - mins)
                normalized_raster[~valid_mask] = 0
                clipped = np.clip(normalized_raster, 0, 1)
                channels.append(clipped)
            clipped = np.stack(channels)
            channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
            rgb = channels_last[..., ::-1]
            return rgb


        # In[4]:


        def plot_image_mask_reconstruction(normalized, mask_img, pred_img):
            # Mix visible and predicted patches 
            rec_img = normalized.clone()
            rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

            mask_img_np = mask_img.numpy().reshape(6, 224, 224).transpose((1, 2, 0))[..., :3]

            rec_img_np = (rec_img.numpy().reshape(6, 224, 224) * stds) + means
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 6))

            for subplot in ax:
                subplot.axis('off')

            ax[0].imshow(enhance_raster_for_visualization(input_data))
            masked_img_np = enhance_raster_for_visualization(input_data).copy()
            masked_img_np[mask_img_np[..., 0] == 1] = 0
            ax[1].imshow(masked_img_np)
            ax[2].imshow(enhance_raster_for_visualization(rec_img_np, ref_img=input_data))


        # ### Loading the model
        # Assuming you have the relevant files under this directory

        # In[5]:


        # load weights
        weights_path = "./prithvi/Prithvi_100M.pt"
        checkpoint = torch.load(weights_path, map_location="cpu")

        # read model config
        model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
        with open(model_cfg_path) as f:
            model_config = yaml.safe_load(f)

        model_args, train_args = model_config["model_args"], model_config["train_params"]

        # let us use only 1 frame for now (the model was trained on 3 frames)
        model_args["num_frames"] = 1

        # instantiate model
        model = MaskedAutoencoderViT(**model_args)
        model.eval()

        # load weights into model
        # strict=false since we are loading with only 1 frame, but the warning is expected
        del checkpoint['pos_embed']
        del checkpoint['decoder_pos_embed']
        _ = model.load_state_dict(checkpoint, strict=False)


        # ### Let's try it out! 
        # We can access the images directly from the HuggingFace space thanks to rasterio

        # In[6]:


        # raster_path = "https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-demo/resolve/main/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
        # input_data = load_raster(raster_path, crop=(224, 224))

        # #input_data = 



        # #input_data = 





        # print(f"Input data shape is {input_data.shape}")
        # raster_for_visualization = enhance_raster_for_visualization(input_data)
        # plt.imshow(raster_for_visualization)


        
        
        # #### Lets call the model!
        # We pass:
        #  - The normalized input image, cropped to size (224, 224)
        #  - `mask_ratio`: The proportion of pixels that will be masked
        # 
        # The model returns a tuple with:
        #  - loss
        #  - reconstructed image
        #  - mask used

        # In[7]:


        
        # # statistics used to normalize images before passing to the model 
        # means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
        # stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)

        # def preprocess_image(image): 
        #     # normalize image
        #     normalized = image.copy()
        #     normalized = ((image - means) / stds)
        #     normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
        #     return normalized


        # # In[8]:


        # #normalized = preprocess_image(input_data)   
        #normalized = dl_train
        normalized = self.train_loader

        
        
        # with torch.no_grad(): 
        #         #mask_ratio = 0.5    
        #         mask_ratio = 0.
        #         _, pred, mask = model(normalized, mask_ratio=mask_ratio)
        #         mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
        #         pred_img = model.unpatchify(pred).detach().cpu()


        # # #### Lets use these to build a nice output visualization 

        # # In[9]:


        # plot_image_mask_reconstruction(normalized, mask_img, pred_img)


        # ## Inference with finetuned Prithvi   
        # 
        # #### Let's explore a finetuned example - Flood Segmentation
        # 
        # This time, lets use the huggingface hub library to directly download the files for the finetuned model.

        # In[10]:


        # %pip install huggingface_hub         


        # In[11]:


        #!pip install ./mmdetection/addict-2.4.0-py3-none-any.whl  
        #!pip install ./mmdetection/yapf-0.31.0-py2.py3-none-any.whl
        #!pip install ./mmdetection/terminaltables-3.1.0-py3-none-any.whl   
        #!pip install ./mmdetection/einops-0.4.1-py3-none-any.whl  
        #!pip install ./mmsegmentation/mmcv-full/mmcv_full-1.5.3-cp37-cp37m-linux_x86_64.whl 
        #!pip install ./openmmlab-essential-repositories/openmmlab-repos/src/mmcls-0.23.1-py2.py3-none-any.whl

        #!cp -r ./mmsegm/mmsegmentation-master /kaggle/working/ && cd /kaggle/working/mmsegmentation-master && pip install -e . && cd ..

        from mmcv import Config 
        #from mmengine.config import Config  

        from mmseg.models import build_segmentor 
        from mmseg.datasets.pipelines import Compose, LoadImageFromFile   
        #from mmdet.datasets.pipelines import Compose, LoadImageFromFile   
        #from mmseg.apis import init_model 
        from mmseg.apis import init_segmentor 
        from model_inference import inference_segmentor, process_test_pipeline 
        from huggingface_hub import hf_hub_download
        import matplotlib
        from torch import nn


        # In[12]:


        # # Grab the config and model weights from huggingface 
        config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")
        #config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M", filename="Prithvi.py") 
        #ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename='sen1floods11_Prithvi_100M.pth') 
        ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M", filename='Prithvi_100M.pt')
        #finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device="cpu")
        finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device='cuda:0')


        # ### Let's grab an image to do inference on

        # In[13]:


        # !wget https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-sen1floods11-demo/resolve/main/Spain_7370579_S2Hand.tif


        # In[14]:


        input_data_inference = load_raster("Spain_7370579_S2Hand.tif")
        print(f"Image input shape is {input_data_inference.shape}")
        raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
        plt.axis('off')
        plt.imshow(raster_for_visualization)


        # In[15]:


        # # adapt this pipeline for Tif files with > 3 images  
        custom_test_pipeline = process_test_pipeline(finetuned_model.cfg.data.test.pipeline)
        result = inference_segmentor(finetuned_model, "Spain_7370579_S2Hand.tif", custom_test_pipeline=custom_test_pipeline)


        # In[16]:


        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        input_data_inference = load_raster("Spain_7370579_S2Hand.tif")
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
        ax[0].imshow(enhance_raster_for_visualization(input_data_inference))
        ax[1].imshow(result[0], norm=norm, cmap="jet")
        ax[2].imshow(enhance_raster_for_visualization(input_data_inference))
        ax[2].imshow(result[0], cmap="jet", alpha=0.3, norm=norm)
        for subplot in ax:
            subplot.axis('off')


        # ## Finetuning for your use case 
        # To finetune, you can now write a PyTorch loop as usual to train on your dataset. Simply extract the backbone from the model with some surgery and run only the model features forward, with no masking!
        # 
        #  In general some reccomendations are:
        # - At least in the beggining, experiment with freezing the backbone. This will give you much faster iteration through experiments.
        # - Err on the side of a smaller learning rate
        # - With an unfrozen encoder, regularization is your friend! (Weight decay, dropout, batchnorm...)

        # In[17]:


        # if going with pytorch:  
        # - remember to normalize images beforehand (find the normalization statistics in the config file)  
        # - turn off masking by passing mask_ratio = 0
        #normalized = preprocess_image(input_data)
        #normalized = dl_train
        normalized = self.train_loader
        # features, _, _ = model.forward_encoder(normalized, mask_ratio=0)


        # # #### What do these features look like?
        # # These are the standard output of a ViT.
        # # - Dim 1: Batch size
        # # - Dim 2: [`cls_token`] + tokens representing flattened image 
        # # - Dim 3: embedding dimension
        # # 
        # # First reshape features into "image-like" shape:  
        # # - Drop cls_token
        # # - reshape into HxW shape

        # # In[18]:


        # print(f"Encoder features have shape {features.shape}")  

        # # drop cls token 
        # reshaped_features = features[:, 1:, :]

        # # reshape
        # feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        # reshaped_features = reshaped_features.view(-1, feature_img_side_length, feature_img_side_length, model_args["embed_dim"])
        # # channels first
        # reshaped_features = reshaped_features.permute(0, 3, 1, 2)
        # print(f"Encoder features have new shape {reshaped_features.shape}")


        # #### Example of a segmentation head  
        # A simple segmentation head can consist of a few upscaling blocks + a final head for classification

        # In[19]:


        #num_classes = 2           
        num_classes = 11

        upscaling_block = lambda in_channels, out_chnnels: nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1), nn.ReLU())
        embed_dims = [model_args["embed_dim"] // (2**i) for i in range(5)]
        segmentation_head = nn.Sequential(
            *[
            upscaling_block(embed_dims[i], embed_dims[i+1]) for i in range(4)
            ],
            nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes))a


        # ### Running features through the segmentation head    
        # We now get an output of shape [batch_size, num_classes, height, width]  

        # In[20]:


        
        
        
        
        features, _, _ = model.forward_encoder(normalized, mask_ratio=0)       

        # # This throws an error.                       

        # # File "/home/nikolaos/phileotestbed/utils/training_loops.py", line 616, in train                 
        #     features, _, _ = model.forward_encoder(normalized, mask_ratio=0)   
        # File "/home/nikolaos/phileotestbed/prithvi/Prithvi.py", line 250, in forward_encoder
        #     x = self.patch_embed(x)
        # File "/home/nikolaos/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
        #     return forward_call(*input, **kwargs)
        # File "/home/nikolaos/phileotestbed/prithvi/Prithvi.py", line 117, in forward
        #     B, C, T, H, W = x.shape
        # # AttributeError: 'DataLoader' object has no attribute 'shape' 

        
        
        # #### What do these features look like?           
        # These are the standard output of a ViT. 
        # - Dim 1: Batch size 
        # - Dim 2: [`cls_token`] + tokens representing flattened image 
        # - Dim 3: embedding dimension
        # 
        # First reshape features into "image-like" shape:    
        # - Drop cls_token
        # - reshape into HxW shape

        # In[18]:


        print(f"Encoder features have shape {features.shape}") 

        # drop cls token 
        reshaped_features = features[:, 1:, :]

        # reshape
        feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        reshaped_features = reshaped_features.view(-1, feature_img_side_length, feature_img_side_length, model_args["embed_dim"])
        # channels first
        reshaped_features = reshaped_features.permute(0, 3, 1, 2)
        print(f"Encoder features have new shape {reshaped_features.shape}")
        
        
        
        
        
        segmentation_head(reshaped_features).shape

        
        
        # end of add the external model







        # # add the external model    
        
        # import os
        # #import torch
        # import matplotlib.pyplot as plt
        # import numpy as np
        # import rasterio
        # import yaml
        # from prithvi.Prithvi import MaskedAutoencoderViT

        # NO_DATA = -9999
        # NO_DATA_FLOAT = 0.0001
        # PERCENTILES = (0.1, 99.9)


        # # ### Define some functions for visualization 

        # # In[3]:


        # def load_raster(path, crop=None):
        #     with rasterio.open(path) as src:
        #         img = src.read()

        #         # load first 6 bands
        #         img = img[:6]

        #         img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        #         if crop:
        #             img = img[:, -crop[0]:, -crop[1]:]
        #     return img

        # def enhance_raster_for_visualization(raster, ref_img=None):
        #     if ref_img is None:
        #         ref_img = raster
        #     channels = []
        #     for channel in range(raster.shape[0]):
        #         valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        #         valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        #         mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        #         normalized_raster = (raster[channel] - mins) / (maxs - mins)
        #         normalized_raster[~valid_mask] = 0
        #         clipped = np.clip(normalized_raster, 0, 1)
        #         channels.append(clipped)
        #     clipped = np.stack(channels)
        #     channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
        #     rgb = channels_last[..., ::-1]
        #     return rgb


        # # In[4]:


        # def plot_image_mask_reconstruction(normalized, mask_img, pred_img):
        #     # Mix visible and predicted patches 
        #     rec_img = normalized.clone()
        #     rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

        #     mask_img_np = mask_img.numpy().reshape(6, 224, 224).transpose((1, 2, 0))[..., :3]

        #     rec_img_np = (rec_img.numpy().reshape(6, 224, 224) * stds) + means
            
        #     fig, ax = plt.subplots(1, 3, figsize=(15, 6))

        #     for subplot in ax:
        #         subplot.axis('off')

        #     ax[0].imshow(enhance_raster_for_visualization(input_data))
        #     masked_img_np = enhance_raster_for_visualization(input_data).copy()
        #     masked_img_np[mask_img_np[..., 0] == 1] = 0
        #     ax[1].imshow(masked_img_np)
        #     ax[2].imshow(enhance_raster_for_visualization(rec_img_np, ref_img=input_data))


        # # ### Loading the model
        # # Assuming you have the relevant files under this directory

        # # In[5]:


        # # load weights
        # weights_path = "./prithvi/Prithvi_100M.pt"
        # checkpoint = torch.load(weights_path, map_location="cpu")

        # # read model config
        # model_cfg_path = "./prithvi/Prithvi_100M_config.yaml"
        # with open(model_cfg_path) as f:
        #     model_config = yaml.safe_load(f)

        # model_args, train_args = model_config["model_args"], model_config["train_params"]

        # # let us use only 1 frame for now (the model was trained on 3 frames)
        # model_args["num_frames"] = 1

        # # instantiate model
        # model = MaskedAutoencoderViT(**model_args)
        # model.eval()

        # # load weights into model
        # # strict=false since we are loading with only 1 frame, but the warning is expected
        # del checkpoint['pos_embed']
        # del checkpoint['decoder_pos_embed']
        # _ = model.load_state_dict(checkpoint, strict=False)


        # # ### Let's try it out! 
        # # We can access the images directly from the HuggingFace space thanks to rasterio

        # # In[6]:


        # # raster_path = "https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-demo/resolve/main/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
        # # input_data = load_raster(raster_path, crop=(224, 224))

        # # #input_data = 



        # # #input_data = 





        # # print(f"Input data shape is {input_data.shape}")
        # # raster_for_visualization = enhance_raster_for_visualization(input_data)
        # # plt.imshow(raster_for_visualization)


        
        
        # # #### Lets call the model!
        # # We pass:
        # #  - The normalized input image, cropped to size (224, 224)
        # #  - `mask_ratio`: The proportion of pixels that will be masked
        # # 
        # # The model returns a tuple with:
        # #  - loss
        # #  - reconstructed image
        # #  - mask used

        # # In[7]:


        
        # # # statistics used to normalize images before passing to the model 
        # # means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
        # # stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)

        # # def preprocess_image(image):
        # #     # normalize image
        # #     normalized = image.copy()
        # #     normalized = ((image - means) / stds)
        # #     normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
        # #     return normalized


        # # # In[8]:


        # # #normalized = preprocess_image(input_data)  
        # normalized = dl_train

        
        
        # # with torch.no_grad(): 
        # #         #mask_ratio = 0.5    
        # #         mask_ratio = 0.
        # #         _, pred, mask = model(normalized, mask_ratio=mask_ratio)
        # #         mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
        # #         pred_img = model.unpatchify(pred).detach().cpu()


        # # # #### Lets use these to build a nice output visualization 

        # # # In[9]:


        # # plot_image_mask_reconstruction(normalized, mask_img, pred_img)


        # # ## Inference with finetuned Prithvi   
        # # 
        # # #### Let's explore a finetuned example - Flood Segmentation
        # # 
        # # This time, lets use the huggingface hub library to directly download the files for the finetuned model.

        # # In[10]:


        # # %pip install huggingface_hub         


        # # In[11]:


        # #!pip install ./mmdetection/addict-2.4.0-py3-none-any.whl  
        # #!pip install ./mmdetection/yapf-0.31.0-py2.py3-none-any.whl
        # #!pip install ./mmdetection/terminaltables-3.1.0-py3-none-any.whl   
        # #!pip install ./mmdetection/einops-0.4.1-py3-none-any.whl  
        # #!pip install ./mmsegmentation/mmcv-full/mmcv_full-1.5.3-cp37-cp37m-linux_x86_64.whl 
        # #!pip install ./openmmlab-essential-repositories/openmmlab-repos/src/mmcls-0.23.1-py2.py3-none-any.whl

        # #!cp -r ./mmsegm/mmsegmentation-master /kaggle/working/ && cd /kaggle/working/mmsegmentation-master && pip install -e . && cd ..

        # from mmcv import Config 
        # #from mmengine.config import Config  

        # from mmseg.models import build_segmentor 
        # from mmseg.datasets.pipelines import Compose, LoadImageFromFile   
        # #from mmdet.datasets.pipelines import Compose, LoadImageFromFile   
        # #from mmseg.apis import init_model 
        # from mmseg.apis import init_segmentor 
        # from model_inference import inference_segmentor, process_test_pipeline 
        # from huggingface_hub import hf_hub_download
        # import matplotlib
        # from torch import nn


        # # In[12]:


        # # # Grab the config and model weights from huggingface 
        # config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")
        # #config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M", filename="Prithvi.py") 
        # #ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename='sen1floods11_Prithvi_100M.pth') 
        # ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M", filename='Prithvi_100M.pt')
        # #finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device="cpu")
        # finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device='cuda:0')


        # # ### Let's grab an image to do inference on

        # # In[13]:


        # # !wget https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-sen1floods11-demo/resolve/main/Spain_7370579_S2Hand.tif


        # # In[14]:


        # input_data_inference = load_raster("Spain_7370579_S2Hand.tif")
        # print(f"Image input shape is {input_data_inference.shape}")
        # raster_for_visualization = enhance_raster_for_visualization(input_data_inference)
        # plt.axis('off')
        # plt.imshow(raster_for_visualization)


        # # In[15]:


        # # # adapt this pipeline for Tif files with > 3 images 
        # custom_test_pipeline = process_test_pipeline(finetuned_model.cfg.data.test.pipeline)
        # result = inference_segmentor(finetuned_model, "Spain_7370579_S2Hand.tif", custom_test_pipeline=custom_test_pipeline)


        # # In[16]:


        # fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        # input_data_inference = load_raster("Spain_7370579_S2Hand.tif")
        # norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
        # ax[0].imshow(enhance_raster_for_visualization(input_data_inference))
        # ax[1].imshow(result[0], norm=norm, cmap="jet")
        # ax[2].imshow(enhance_raster_for_visualization(input_data_inference))
        # ax[2].imshow(result[0], cmap="jet", alpha=0.3, norm=norm)
        # for subplot in ax:
        #     subplot.axis('off')


        # # ## Finetuning for your use case
        # # To finetune, you can now write a PyTorch loop as usual to train on your dataset. Simply extract the backbone from the model with some surgery and run only the model features forward, with no masking!
        # # 
        # #  In general some reccomendations are:
        # # - At least in the beggining, experiment with freezing the backbone. This will give you much faster iteration through experiments.
        # # - Err on the side of a smaller learning rate
        # # - With an unfrozen encoder, regularization is your friend! (Weight decay, dropout, batchnorm...)

        # # In[17]:


        # # if going with pytorch:  
        # # - remember to normalize images beforehand (find the normalization statistics in the config file) 
        # # - turn off masking by passing mask_ratio = 0
        # #normalized = preprocess_image(input_data)
        # normalized = dl_train
        # # features, _, _ = model.forward_encoder(normalized, mask_ratio=0)


        # # # #### What do these features look like?
        # # # These are the standard output of a ViT.
        # # # - Dim 1: Batch size
        # # # - Dim 2: [`cls_token`] + tokens representing flattened image 
        # # # - Dim 3: embedding dimension
        # # # 
        # # # First reshape features into "image-like" shape:  
        # # # - Drop cls_token
        # # # - reshape into HxW shape

        # # # In[18]:


        # # print(f"Encoder features have shape {features.shape}") 

        # # # drop cls token 
        # # reshaped_features = features[:, 1:, :]

        # # # reshape
        # # feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        # # reshaped_features = reshaped_features.view(-1, feature_img_side_length, feature_img_side_length, model_args["embed_dim"])
        # # # channels first
        # # reshaped_features = reshaped_features.permute(0, 3, 1, 2)
        # # print(f"Encoder features have new shape {reshaped_features.shape}")


        # # #### Example of a segmentation head 
        # # A simple segmentation head can consist of a few upscaling blocks + a final head for classification

        # # In[19]:


        # #num_classes = 2      
        # num_classes = 11

        # upscaling_block = lambda in_channels, out_channels: nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1), nn.ReLU())
        # embed_dims = [model_args["embed_dim"] // (2**i) for i in range(5)]
        # segmentation_head = nn.Sequential(
        #     *[
        #     upscaling_block(embed_dims[i], embed_dims[i+1]) for i in range(4)
        #     ],
        #     nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes))


        # # ### Running features through the segmentation head  
        # # We now get an output of shape [batch_size, num_classes, height, width] 

        # # In[20]:


        
        
        
        
        # features, _, _ = model.forward_encoder(normalized, mask_ratio=0)


        # # #### What do these features look like? 
        # # These are the standard output of a ViT.
        # # - Dim 1: Batch size
        # # - Dim 2: [`cls_token`] + tokens representing flattened image 
        # # - Dim 3: embedding dimension
        # # 
        # # First reshape features into "image-like" shape:   
        # # - Drop cls_token
        # # - reshape into HxW shape

        # # In[18]:


        # print(f"Encoder features have shape {features.shape}") 

        # # drop cls token 
        # reshaped_features = features[:, 1:, :]

        # # reshape
        # feature_img_side_length = int(np.sqrt(reshaped_features.shape[1]))
        # reshaped_features = reshaped_features.view(-1, feature_img_side_length, feature_img_side_length, model_args["embed_dim"])
        # # channels first
        # reshaped_features = reshaped_features.permute(0, 3, 1, 2)
        # print(f"Encoder features have new shape {reshaped_features.shape}")
        
        
        
        
        
        # segmentation_head(reshaped_features).shape

        
        
        # # end of add the external model    



        
        
        # if model_name == 'GeoAware_contrastive_core_nano': 
        #     NAME = model.__class__.__name__ +'_contrastive_frozen' if freeze_pretrained else model.__class__.__name__ +'_contrastive_unfrozen'
        # elif model_name == 'GeoAware_basic_core_nano':
        #     NAME = model.__class__.__name__ +'_basic_frozen' if freeze_pretrained else model.__class__.__name__ +'_basic_unfrozen'
        # else:
        #     NAME = model.__class__.__name__ + '_frozen' if freeze_pretrained else model.__class__.__name__ + '_unfrozen'



    else:
        if freeze_pretrained: 
            print(f"Ignoring freeze_pretrained set to {freeze_pretrained} as no pretrained model was supplied") 
        model = get_models(model_name, input_channels, output_channels, input_size)
        NAME = model.__class__.__name__


    
    if warmup:
        lr = lr / 100000  # # for warmup start 

    
    
    
    
    # if isinstance(n_shot, int):
    #     OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{n_shot}'
    #     x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot_memmapped(
    #         '/phileo_data/downstream/downstream_dataset_patches_np/',
    #         dst='/phileo_data/downstream/downstream_datasets_nshot/',
    #         n=n_shot,
    #         regions=regions,
    #         y=downstream_task,
    #         data_selection='create')

    # elif isinstance(split_ratio, float):
    #     OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{split_ratio}'
    #     x_train, y_train, x_val, y_val = data_protocol.protocol_split(
    #         '/phileo_data/downstream/downstream_dataset_patches_np/',
    #         split_percentage=split_ratio,
    #         regions=regions,
    #         y=downstream_task)

    # x_test, y_test = data_protocol.get_testset(folder='/phileo_data/downstream/downstream_dataset_patches_np/',
    #                                            y=downstream_task)

    # dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
    #                                                 with_augmentations=augmentations,
    #                                                 num_workers=num_workers,
    #                                                 batch_size=batch_size,
    #                                                 land_cover=lc,
    #                                                 device=device
    #                                                 )

    
    
    model_summary = summary(model,
                            input_size=(batch_size, input_channels, input_size, input_size), )

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





'''

import os

import torch
from functools import partial
from torchinfo import summary


import torch.nn as nn
from datetime import date
import argparse
import sys; sys.path.append("../") 

from models.model_Baseline import BaselineNet
from models.model_CoreCNN_versions import CoreUnet_nano, CoreUnet_tiny, CoreUnet_base, CoreUnet_large, CoreUnet_huge
from models.model_Mixer_versions import Mixer_nano, Mixer_tiny, Mixer_base, Mixer_large, Mixer_huge
from models.model_LinearViT_versions import LinearViT_base, LinearViT_large, LinearViT_huge
from models.model_AutoEncoderViT_versions import AutoencoderViT_base, AutoencoderViT_large, AutoencoderViT_huge
from models.model_GeoAwarePretrained import MixerGeoPretrained, get_mixer_kwargs, get_core_encoder_kwargs, CoreEncoderGeoPretrained, CoreEncoderGeoPretrained_combined
from models.model_AutoEncoderViTPretrained import AutoEncoderViTPretrained

from utils import data_protocol    
from utils import load_data
from utils import training_loops
from utils.training_utils import read_yaml 
torch.manual_seed(123456)
CNN_LIST = ['baseline_cnn', 'core_unet_nano','core_unet_tiny','core_unet_base', 'core_unet_large', 'core_unet_huge']
MIXER_LIST = ['mixer_nano', 'mixer_tiny', 'mixer_base', 'mixer_large', 'mixer_huge']
VIT_LIST = ['linear_vit_base', 'linear_vit_larger', 'linear_vit_huge',
            'autoencoder_vit_base', 'autoencoder_vit_large', 'autoencoder_vit_huge']
CNN_PRETRAINED_LIST = ['GeoAware_core_nano', 'GeoAware_core_tiny', 'GeoAware_mixer_nano', 'GeoAware_mixer_tiny',
                       'GeoAware_contrastive_core_nano', 'GeoAware_basic_core_nano', 'GeoAware_combined_core_nano']

VIT_PRETRAINED_LIST = ['AutoEncoderVitPretrained']     

MODEL_LIST = CNN_LIST + MIXER_LIST + VIT_LIST + CNN_PRETRAINED_LIST 

def get_trainer(model_name, downstream_task, epochs, lr, model, device, lr_scheduler, warmup, early_stop, dl_train,
                dl_val, dl_test, NAME, OUTPUT_FOLDER, vis_val):

    if model_name in (CNN_LIST + MIXER_LIST + CNN_PRETRAINED_LIST):
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

    elif model_name in (VIT_LIST + VIT_PRETRAINED_LIST): 
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

    return trainer


def get_models(model_name, input_channels, output_channels, input_size):
    if model_name == 'baseline_cnn':
        return BaselineNet(input_dim=input_channels, output_dim=output_channels)
    elif model_name == 'core_unet_nano':
        return CoreUnet_nano(input_dim=input_channels, output_dim=output_channels)
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


def get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=None, freeze=False, device='cuda'):
    
    test_input = torch.rand((2,input_channels,input_size,input_size))

    if model_name == 'GeoAware_core_nano' or model_name == 'GeoAware_contrastive_core_nano' or model_name == 'GeoAware_basic_core_nano':
        sd = torch.load(path_model_weights)
        core_kwargs = get_core_encoder_kwargs(output_dim=output_channels, input_dim=input_channels, core_size='core_nano', full_unet=True)
        model = CoreEncoderGeoPretrained(output_channels, checkpoint=sd, core_encoder_kwargs=core_kwargs, freeze_body=freeze)
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

    if model_name == 'AutoEncoderVitPretrained':
        sd = torch.load(path_model_weights, map_location=device)
        model = AutoEncoderViTPretrained(chw=(input_channels, input_size, input_size),
                                         output_dim=output_channels, checkpoint=sd, freeze_body=freeze)
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
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    #torch.set_default_device(device) 
    print('DEVICE', device)

    assert not (n_shot == None) or not (split_ratio == None), 'Please define data partition protocol!'
    assert isinstance(n_shot, int) ^ isinstance(split_ratio, float), 'n_shot cannot be used with split_ratio!'
    if (downstream_task == 'lc'):
        assert not (output_channels == 1), 'land cover task should have more than 1 output channels'
        lc = True

    if (downstream_task == 'roads') or (downstream_task == 'building'):
        assert output_channels == 1, 'regression type tasks should have a single output channel'
        lc = False

    if pretrained_model_path is not None:
        assert model_name in (CNN_PRETRAINED_LIST + VIT_PRETRAINED_LIST), f"Pretrained weights were given but model {model_name} not found in list of pretrained models: {(CNN_PRETRAINED_LIST + VIT_PRETRAINED_LIST)}"
        assert freeze_pretrained is not None, f"When supplying a pretrained model 'freeze_pretrained' must be either True or False"
        model = get_models_pretrained(model_name, input_channels, output_channels, input_size, path_model_weights=pretrained_model_path, freeze=freeze_pretrained)
        if model_name == 'GeoAware_contrastive_core_nano':
            NAME = model.__class__.__name__ +'_contrastive_frozen' if freeze_pretrained else model.__class__.__name__ +'_contrastive_unfrozen'
        elif model_name == 'GeoAware_basic_core_nano':
            NAME = model.__class__.__name__ +'_basic_frozen' if freeze_pretrained else model.__class__.__name__ +'_basic_unfrozen'
        else:
            NAME = model.__class__.__name__ + '_frozen' if freeze_pretrained else model.__class__.__name__ + '_unfrozen'

    else:
        if freeze_pretrained:
            print(f"Ignoring freeze_pretrained set to {freeze_pretrained} as no pretrained model was supplied")
        model = get_models(model_name, input_channels, output_channels, input_size)
        NAME = model.__class__.__name__


    OUTPUT_FOLDER = f'/phileo_data/experiments/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}'
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'/phileo_data/experiments/{experiment_name}/{downstream_task}/{date.today().strftime("%d%m%Y")}_{NAME}_{downstream_task}_{lr_scheduler}'

    if warmup:
        lr = lr / 100000  # for warmup start

    if isinstance(n_shot, int):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{n_shot}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_fewshot_memmapped(
            '/phileo_data/downstream/downstream_dataset_patches_np/',
            dst='/phileo_data/downstream/downstream_datasets_nshot/',
            n=n_shot,
            regions=regions,
            y=downstream_task,
            data_selection='create')

    elif isinstance(split_ratio, float):
        OUTPUT_FOLDER = f'{OUTPUT_FOLDER}_{split_ratio}'
        x_train, y_train, x_val, y_val = data_protocol.protocol_split(
            '/phileo_data/downstream/downstream_dataset_patches_np/',
            split_percentage=split_ratio,
            regions=regions,
            y=downstream_task)

    x_test, y_test = data_protocol.get_testset(folder='/phileo_data/downstream/downstream_dataset_patches_np/',
                                               y=downstream_task)

    dl_train, dl_test, dl_val = load_data.load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                                    with_augmentations=augmentations,
                                                    num_workers=num_workers,
                                                    batch_size=batch_size,
                                                    land_cover=lc,
                                                    device=device
                                                    )

    
    
    # add new 
    
    #torch.save(model.state_dict(), './modeltouse_tocompare')  

    #model.load_state_dict(torch.load('./modeltouse_tocompare')) 
    
    # end the add new
    
    
    
    model_summary = summary(model,
                            input_size=(batch_size, input_channels, input_size, input_size), )

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





'''

