#!/usr/bin/env python
# coding: utf-8

# # Prithvi 100M model
# This notebook will demonstrate basic usage of the Prithvi ViT model.

# ## Getting started with Prithvi - Reconstruction    
# ### Get model files
# 
# To get started, clone the HuggingFace repository for Prithvi 100M, running the command below
# 
# ```bash
# # Make sure you have git-lfs installed (https://git-lfs.com)
# git lfs install
# git clone https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M
# # rename to a valid python module name
# mv Prithvi-100M prithvi
# ```
# 
# Alternatively, you can directly download the [weights](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/tree/main#:~:text=Prithvi_100M.pt,pickle) and [model class](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi.py) and [configuration file](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M_config.yaml) from the repository and place them inside a directory named`prithvi`.
# 
# A third alternative is to leverage the `huggingface_hub` library to download these files directly through code.
# `%pip install huggingface_hub`

# ### Treat it as a module    
# Next, lets add an `__init__.py` file to the downloaded directory, so we can treat it as a module and import the `MaskedAutoencoderViT` class from it.
# Simply create an empty file inside the `prithvi` directory named `__init__.py` by running the code below

# In[1]:


with open("prithvi/__init__.py", "w") as f:   
    f.write("") 


# ### Relevant imports     
# To run this notebook, besides following the installation steps in the [README](./README.md), make sure to install [jupyter](https://jupyter.org/install)

# In[2]:


import os
import torch
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


raster_path = "https://huggingface.co/spaces/ibm-nasa-geospatial/Prithvi-100M-demo/resolve/main/HLS.L30.T13REN.2018013T172747.v2.0.B02.B03.B04.B05.B06.B07_cropped.tif"
input_data = load_raster(raster_path, crop=(224, 224))

#input_data = 



#input_data = 





print(f"Input data shape is {input_data.shape}")
raster_for_visualization = enhance_raster_for_visualization(input_data)
plt.imshow(raster_for_visualization)


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


# statistics used to normalize images before passing to the model
means = np.array(train_args["data_mean"]).reshape(-1, 1, 1)
stds = np.array(train_args["data_std"]).reshape(-1, 1, 1)

def preprocess_image(image):
    # normalize image
    normalized = image.copy()
    normalized = ((image - means) / stds)
    normalized = torch.from_numpy(normalized.reshape(1, normalized.shape[0], 1, *normalized.shape[-2:])).to(torch.float32)
    return normalized


# In[8]:


normalized = preprocess_image(input_data)
with torch.no_grad():
        mask_ratio = 0.5
        _, pred, mask = model(normalized, mask_ratio=mask_ratio)
        mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
        pred_img = model.unpatchify(pred).detach().cpu()


# #### Lets use these to build a nice output visualization

# In[9]:


plot_image_mask_reconstruction(normalized, mask_img, pred_img)


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


# Grab the config and model weights from huggingface
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


# adapt this pipeline for Tif files with > 3 images
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
normalized = preprocess_image(input_data)
features, _, _ = model.forward_encoder(normalized, mask_ratio=0)


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


# #### Example of a segmentation head
# A simple segmentation head can consist of a few upscaling blocks + a final head for classification

# In[19]:


#num_classes = 2   
num_classes = 11

upscaling_block = lambda in_channels, out_channels: nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(kernel_size=3, in_channels=in_channels, out_channels=out_channels, padding=1), nn.ReLU())
embed_dims = [model_args["embed_dim"] // (2**i) for i in range(5)]
segmentation_head = nn.Sequential(
    *[
    upscaling_block(embed_dims[i], embed_dims[i+1]) for i in range(4)
    ],
    nn.Conv2d(kernel_size=1, in_channels=embed_dims[-1], out_channels=num_classes))


# ### Running features through the segmentation head  
# We now get an output of shape [batch_size, num_classes, height, width] 

# In[20]:


segmentation_head(reshaped_features).shape 


# ### Finetuning - MMSeg 
# Alternatively, finetune using the MMSegmentation extension we have opensourced.
# - No model surgery required
# - No need to write boilerplate training code
# - Integrations with Tensorboard, MLFlow, ...
# - Segmentation evaluation metrics / losses built in
# 
# 1. Build your config file. Look [here](./configs/) for examples, the [ReadME](./README.md) for some docs and [MMSeg](https://mmsegmentation.readthedocs.io/en/0.x/tutorials/config.html) for more general tutorials.
# 2. Collect your dataset in the format determined by MMSeg
# 3. `mim train mmsegmentation <path to my config>`

# This is what the model looks like in the MMSeg configuration code.
# 
# All this composition we did above is done for you!
# ```python
# model = dict(
#     type="TemporalEncoderDecoder",
#     frozen_backbone=False,
#     backbone=dict(
#         type="TemporalViTEncoder",
#         pretrained=pretrained_weights_path,
#         img_size=img_size,
#         patch_size=patch_size,
#         num_frames=num_frames,
#         tubelet_size=1,
#         in_chans=len(bands),
#         embed_dim=embed_dim,
#         depth=num_layers,
#         num_heads=num_heads,
#         mlp_ratio=4.0,
#         norm_pix_loss=False,
#     ),
#     neck=dict(
#         type="ConvTransformerTokensToEmbeddingNeck",
#         embed_dim=num_frames*embed_dim,
#         output_embed_dim=embed_dim,
#         drop_cls_token=True,
#         Hp=img_size // patch_size,
#         Wp=img_size // patch_size,
#     ),
#     decode_head=dict(
#         num_classes=num_classes,
#         in_channels=embed_dim,
#         type="FCNHead",
#         in_index=-1,
#         ignore_index=ignore_index,
#         channels=256,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         norm_cfg=norm_cfg,
#         align_corners=False,
#         loss_decode=dict(
#             type="CrossEntropyLoss",
#             use_sigmoid=False,
#             loss_weight=1,
#             class_weight=ce_weights,
#             avg_non_ignore=True
#         ),
#     ),
#     (...)
# ```

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




