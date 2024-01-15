import os.path

import numpy
from datetime import date
import torch
import gc
import glob


import training_script
import experiment_plots

from utils.training_utils import read_yaml

parser, parser_yaml = training_script.get_args()
args_yaml, remainder = parser_yaml.parse_known_args()
torch.cuda.set_device(2)

if args_yaml.read_yaml is not None:
    print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
    args = read_yaml(args_yaml.read_yaml)
else:
    args = parser.parse_args()


for downstream_task in ['lc_classification', ]: 

    args['downstream_task'] = downstream_task
    if downstream_task == 'lc_classification':
        args['output_channels'] = 11
        metric = 'acc'
    else:
        args['output_channels'] = 5
        metric = 'msc'

    experiment_name = args['experiment_name']
    folder = f'/home/phimultigpu/phileo_NFS/phileo_data/experiments/{experiment_name}/{downstream_task}/'

    for n_shot in [50, 100, 500, 1000, 5000]:  
        model_list = ['resnet_imagenet_classifier', 'core_encoder_nano', 'GeoAware_mh_pred_core_nano_classifier', 'vit_cnn_gc_classifier',  'SatMAE_classifier', 'vit_cnn_classifier', 'prithvi_classifier', 'seasonal_contrast_classifier']         

        for model_name in model_list:
            args['n_shot'] = n_shot
            args['model_name'] = model_name
            args['pretrained_model_path'] = None
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            gc.collect()


            if model_name == 'GeoAware_contrastive_core_nano_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt'
            elif model_name == 'GeoAware_core_nano_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_8.pt'
            elif model_name == 'GeoAware_mh_pred_core_nano_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/01112023_CoreEncoderMultiHead_geo_pred_geo_reduce_on_plateau/CoreEncoderMultiHead_geo_pred_best.pt'
            elif model_name == 'prithvi_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/Prithvi_100M.pt'
            elif model_name == 'SatMAE_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/SatMAE_pretrain-vit-large-e199.pth'
                
            elif model_name == 'vit_cnn_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/31102023_MaskedAutoencoderViT/MaskedAutoencoderViT_ckpt.pt'
            elif model_name == 'vit_cnn_gc_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/03112023_MaskedAutoencoderGroupChannelViT/MaskedAutoencoderGroupChannelViT_ckpt.pt'
            elif model_name == 'seasonal_contrast_classifier':
                args['pretrained_model_path'] = '/home/phimultigpu/phileo_NFS/phileo_data/pretrained_models/seco_resnet50_1m.ckpt'
                

            if model_name == 'core_encoder_nano' or model_name == 'vit_cnn_base' or model_name == 'resnet_imagenet_classifier':
                args['pretrained_model_path'] = None
                training_script.main(**vars(args))
          
            else:
                for freeze in [False, True]:
                    args['freeze_pretrained'] = freeze
                    training_script.main(**vars(args))


        # experiment_plots.main(folder=f'/phileo_data/experiments/nshot/{downstream_task}', plot_title=f'nshot experiment on {downstream_task}',
        #                       filter_on=['CoreUnet','Pretrained_frozen','Pretrained_unfrozen', 'Pretrained_contrastive_frozen', 'Pretrained_contrastive_unfrozen'],
        #                       downstream_task=downstream_task, metric=metric, y_logscale=False, x_logscale=False)

