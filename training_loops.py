# Standard Library
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json

# utils
from utils import visualize
from utils import config_lc


class TrainBase():

    def __init__(self, model: nn.Module, device: torch.device, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: DataLoader, epochs:int = 50, early_stop:int=25, lr: float = 0.001, lr_scheduler: str = None, warmup:bool=True,
                 metrics: list = None, name: str="model", out_folder :str ="trained_models/", visualise_validation:bool=True, ):

        self.test_loss = None
        self.last_epoch = None
        self.best_sd = None
        self.epochs = epochs
        self.early_stop = early_stop
        self.learning_rate = lr
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.warmup = warmup
        self.name = name
        self.out_folder = out_folder
        self.visualise_validation = visualise_validation
        if visualise_validation:
            os.makedirs(f'{self.out_folder}/val_images', exist_ok=True)

        self.scaler, self.optimizer = self.set_optimizer()
        self.criterion = self.set_criterion()
        self.scheduler = self.set_scheduler()

        if self.warmup:
            self.scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[1, 2, 3, 4, 5], gamma=(10))

        # # initialize torch device 
        #torch.set_default_device(self.device)
        device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            print("No CUDA device available.")

        # # init useful variables 
        self.best_epoch = 0
        self.best_loss = None
        self.best_model_state = model.state_dict().copy()
        self.epochs_no_improve = 0

        # used for plots
        self.tl = []
        self.vl = []
        self.e = []
        self.lr = []



    def set_optimizer(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),
                                      lr=self.learning_rate, eps=1e-06)

        scaler = GradScaler()

        # Save the initial learning rate in optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = self.learning_rate

        return scaler, optimizer

    def set_criterion(self):
        return nn.MSELoss()

    def set_scheduler(self):
        if self.lr_scheduler == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                20,
                2,
                eta_min=0.000001,
                last_epoch=self.epochs - 1,
            )
        elif self.lr_scheduler == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.1, patience=6, min_lr=1e-6)
        else:
            scheduler = None
        return scheduler

    def get_loss(self, images, labels):
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss
    
    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['mse','mae','mave','acc','precision','recall','baseline_mse']
            # intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']

            final_metrics = {'mse':running_metric[0] / (k + 1), 'mae':running_metric[1] / (k + 1), 'mave':running_metric[2] / (k + 1), 'acc':running_metric[3]/ (k + 1), 'precision':running_metric[4]/(running_metric[4]+running_metric[5]), 'recall':running_metric[4]/(running_metric[4]+running_metric[6]), 'baseline_mse':running_metric[7] / (k + 1)}
            final_metrics['f1'] = 2 * final_metrics['precision'] * final_metrics['recall'] / (final_metrics['precision'] + final_metrics['recall'])

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['mse','mae','mave','acc','tp','fp','fn','baseline_mse']
            metric_init = np.zeros(len(intermediary_values)) # 
            return  metric_init
        
        
        else:
            
            outputs = self.model(images)
            # regression metrics
            error = outputs - labels
            squared_error = error**2
            test_mse = squared_error.mean().item()
            test_mae = error.abs().mean().item()
            test_mave = torch.mean(torch.abs(outputs.mean(dim=(1,2)) - labels.mean(dim=(1,2)) ) ).item()

            # regression metrics disguised as classification
            threshold = 0.5
            label_classification = (labels > threshold).type(torch.int8)
            output_classification = (outputs > threshold).type(torch.int8)

            diff = output_classification - label_classification
            fp = torch.count_nonzero(diff==1).item()
            fn = torch.count_nonzero(diff==-1).item()
            tp = label_classification.sum().item() - fn

            test_accuracy = (label_classification==output_classification).type(torch.float).mean().item()
            test_zero_model_mse = (labels**2).mean().item()

            return np.array([test_mse,test_mae,test_mave,test_accuracy,tp,fp,fn,test_zero_model_mse])



    def t_loop(self, epoch, s):
        # Initialize the running loss
        train_loss = 0.0
        # Initialize the progress bar for training
        train_pbar = tqdm(self.train_loader, total=len(self.train_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()
            # get loss
            with autocast(dtype=torch.float16):
                loss = self.get_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += loss.item()

            # display progress on console
            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                f"lr": self.optimizer.param_groups[0]['lr']})

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

        return i, train_loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize(x=images, y=labels, y_pred=outputs, images=5,
                            channel_first=True, vmin=0, vmax=1, save_path=f"{self.out_folder}/{name}.png")

    def v_loop(self, epoch):

        # Initialize the progress bar for training
        val_pbar = tqdm(self.val_loader, total=len(self.val_loader),
                          desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            self.model.eval()
            val_loss = 0
            for j, (images, labels) in enumerate(val_pbar):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)
                val_loss += loss.item()

                # display progress on console
                val_pbar.set_postfix({
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    f"lr": self.optimizer.param_groups[0]['lr']})

            if self.visualise_validation:
                outputs = self.model(images)
                self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(), outputs.detach().cpu().numpy(), name=f'/val_images/val_{epoch}')

            return j, val_loss

    def save_ckpt(self, epoch, val_loss):
        model_sd = self.model.state_dict().copy()
        if self.best_loss is None:
            self.best_epoch = epoch
            self.best_loss = val_loss
            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        elif self.best_loss > val_loss:
            self.best_epoch = epoch
            self.best_loss = val_loss
            self.epochs_no_improve = 0

            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        else:
            self.epochs_no_improve += 1

        torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_last.pt"))

    def plot_curves(self, epoch):
        # visualize loss & lr curves
        self.e.append(epoch)

        fig = plt.figure()
        plt.plot(self.e, self.tl, label='Training Loss', )
        plt.plot(self.e, self.vl, label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"loss.png"))
        plt.close('all')
        fig = plt.figure()
        plt.plot(self.e, self.lr, label='Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"lr.png"))
        plt.close('all')

    def train(self):
        print("Starting training...")
        print("")

        #print(self.device)

        
        
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
        # #normalized = dl_train
        # normalized = self.train_loader

        
        
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
        # #normalized = dl_train
        # normalized = self.train_loader
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

        # # # This throws an error.                       

        # # # File "/home/nikolaos/phileotestbed/utils/training_loops.py", line 616, in train                 
        # #     features, _, _ = model.forward_encoder(normalized, mask_ratio=0)   
        # # File "/home/nikolaos/phileotestbed/prithvi/Prithvi.py", line 250, in forward_encoder
        # #     x = self.patch_embed(x)
        # # File "/home/nikolaos/env/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
        # #     return forward_call(*input, **kwargs)
        # # File "/home/nikolaos/phileotestbed/prithvi/Prithvi.py", line 117, in forward
        # #     B, C, T, H, W = x.shape
        # # # AttributeError: 'DataLoader' object has no attribute 'shape' 

        
        
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
        
        
        
        
        
        # init model
        self.model.to(self.device)
        self.model.train()

        # create dst folder for generated files/artifacts
        os.makedirs(self.out_folder, exist_ok=True)
        s = self.scheduler

        # # Training loop 
        for epoch in range(self.epochs):
            if epoch == 0 and self.warmup == True:
                s = self.scheduler_warmup
                print('Starting linear warmup phase')
            elif epoch == 5 and self.warmup == True:
                s = self.scheduler
                self.warmup = False
                print('Warmup finished')

            i, train_loss = self.t_loop(epoch, s)
            j, val_loss = self.v_loop(epoch)

            self.tl.append(train_loss / (i + 1))
            self.vl.append(val_loss / (j + 1))
            self.lr.append(self.optimizer.param_groups[0]['lr'])

            # Update the scheduler 
            if self.warmup:
                s.step()
            elif self.lr_scheduler == 'reduce_on_plateau':
                s.step(self.vl[-1])

            #save check point
            self.save_ckpt(epoch, val_loss / (j + 1))

            # visualize loss & lr curves
            self.plot_curves(epoch)
            self.model.train()

            # Early stopping 
            if self.epochs_no_improve == self.early_stop:
                print(f'Early stopping triggered after {epoch + 1} epochs.')
                self.last_epoch = epoch + 1
                break

    def test(self):
        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        print("Finished Training. Best epoch: ", self.best_epoch + 1)
        print("")
        print("Starting Testing...")
        self.model.eval()
        test_pbar = tqdm(self.test_loader, total=len(self.test_loader),
                          desc=f"Test Set")
        with torch.no_grad():

            running_metric = self.get_metrics()

            for k, (images, labels) in enumerate(test_pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                running_metric += self.get_metrics(images,labels)


            self.test_metrics = self.get_metrics(running_metric=running_metric, k=k)
        

            print(f"Test Loss: {self.test_metrics}")
            outputs = self.model(images)
            self.val_visualize(images.detach().cpu().numpy(), labels.detach().cpu().numpy(),
                               outputs.detach().cpu().numpy(), name='test')

    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup=None, lr=None):
        artifacts = {'training_parameters': {'model': self.name,
                                             'lr': lr,
                                             'scheduler': self.lr_scheduler,
                                             'warm_up': warmup,
                                             'optimizer': str(self.optimizer).split(' (')[0],
                                             'device': str(self.device),
                                             'training_epochs': self.epochs,
                                             'early_stop': self.early_stop,
                                             'train_samples': len(self.train_loader) * model_summary.input_size[0][0],
                                             'val_samples': len(self.val_loader) * model_summary.input_size[0][0],
                                             'test_samples': len(self.test_loader) * model_summary.input_size[0][0],
                                             'n_shot': n_shot,
                                             'p_split': p_split
                                             },

                     'training_info': {'best_val_loss': self.best_loss,
                                       'best_epoch': self.best_epoch,
                                       'last_epoch': self.last_epoch},

                     'test_metrics': self.test_metrics,

                     'plot_info': {'epochs': self.e,
                                   'val_losses': self.vl,
                                   'train_losses': self.tl,
                                   'lr': self.lr},

                     'model_summary': {'batch_size': model_summary.input_size[0],
                                       'input_size': model_summary.total_input,
                                       'total_mult_adds': model_summary.total_mult_adds,
                                       'back_forward_pass_size': model_summary.total_output_bytes,
                                       'param_bytes': model_summary.total_param_bytes,
                                       'trainable_params': model_summary.trainable_params,
                                       'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                                       'total_params': model_summary.total_params}
                     }

        with open(f"{self.out_folder}/artifacts.json", "w") as outfile:
            json.dump(artifacts, outfile)


class TrainLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model(images)
        outputs = outputs.flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class
            

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model(images)
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()
            
            # stolen from pytorch confusion matrix
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()

class TrainViT(TrainBase):
    def get_loss(self, images, labels):
        outputs = self.model(images)
        labels = self.model.patchify(labels)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=labels.shape[1])
        visualize.visualize(x=images, y=labels, y_pred=outputs.detach().cpu().numpy(), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")


class TrainViTLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model.unpatchify(self.model(images), c=11).flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        outputs = self.model.unpatchify(torch.from_numpy(outputs), c=11)
        visualize.visualize_lc(x=images, y=labels, y_pred=outputs.detach().cpu().numpy().argmax(axis=1), images=5,
                               channel_first=True, vmin=0, save_path=f"{self.out_folder}/{name}.png")

    def get_metrics(self, images=None, labels=None, running_metric=None, k=None):
        
        if (running_metric is not None) and (k is not None):
            metric_names = ['acc','precision','recall','baseline_mse']
            # intermediary_values = ['confusion_matrix']

            confmat = running_metric

            total_pixels = np.sum(confmat)
            
            tp_per_class = np.diagonal(confmat)
            total_tp = tp_per_class.sum()

            fp_per_class = confmat.sum(axis=0) - tp_per_class
            fn_per_class = confmat.sum(axis=1) - tp_per_class
            

            precision_per_class = tp_per_class/(fp_per_class+tp_per_class)
            recall_per_class = tp_per_class/(fn_per_class+tp_per_class)

            precision_micro = total_tp/(fp_per_class.sum() + total_tp)
            recall_micro = total_tp/(fn_per_class.sum() + total_tp)
            precision_macro = np.mean(precision_per_class)
            recall_macro = np.mean(recall_per_class)

            acc_total = total_tp/total_pixels

            final_metrics = {'acc':acc_total, 'precision_per_class':precision_per_class.tolist(),'recall_per_class':recall_per_class.tolist() ,'precision_micro':precision_micro, 'precision_macro':precision_macro, 'recall_micro':recall_micro, 'recall_macro':recall_macro, 'conf_mat':confmat.tolist()}

            return final_metrics


        elif (images == None) and (labels == None):
            intermediary_values = ['confusion_matrix']
            num_classes = len(config_lc.lc_raw_classes.keys())
            metric_init = np.zeros((num_classes,num_classes)) # 
            return  metric_init
        
        
        else:
            outputs = self.model.unpatchify(self.model(images), c=11)
            outputs = outputs.argmax(axis=1).flatten()
            labels = labels.squeeze().flatten()
            
            # stolen from pytorch confusion matrix
            num_classes = len(config_lc.lc_raw_classes.keys())
            unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
            bins = torch.bincount(unique_mapping, minlength=num_classes**2) 
            cfm = bins.reshape(num_classes, num_classes)

            return cfm.cpu().numpy()