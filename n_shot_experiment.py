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

if args_yaml.read_yaml is not None:
    print(f"WARNING: overwriting all parameters with defaults stored in {args_yaml.read_yaml}")
    args = read_yaml(args_yaml.read_yaml)
else:
    args = parser.parse_args()


for downstream_task in ['building', 'lc', 'roads']:

    args['downstream_task'] = downstream_task
    if downstream_task == 'lc':
        args['output_channels'] = 11
        metric = 'acc'
    else:
        args['output_channels'] = 1
        metric = 'msc'

    experiment_name = args['experiment_name']
    folder = f'/phileo_data/experiments/{experiment_name}/{downstream_task}/'

    for n_shot in [5, 10, 50, 500, 1000, 5000, 10000]: #  50000, 100000, 200000
        for model_name in ['vit_cnn', 'vit_cnn_gc', 'vit_cnn_base', 'SatMAE', 'prithvi', 'core_unet_nano', 'GeoAware_core_nano',
                           'GeoAware_contrastive_core_nano', 'GeoAware_mh_pred_core_nano', 'GeoAware_core_autoencoder_nano', ]:
            args['n_shot'] = n_shot
            args['model_name'] = model_name
            args['batch_size'] = 32
            args['pretrained_model_path'] = None
            torch.clear_autocast_cache()
            torch.cuda.empty_cache()
            gc.collect()

            if model_name == 'GeoAware_contrastive_core_nano':
                args['pretrained_model_path'] = '/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt'
            elif model_name == 'GeoAware_core_nano':
                args['pretrained_model_path'] = '/phileo_data/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_8.pt'
            elif model_name == 'GeoAware_mh_pred_core_nano':
                args['pretrained_model_path'] = '/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/01112023_CoreEncoderMultiHead_geo_pred_geo_reduce_on_plateau/CoreEncoderMultiHead_geo_pred_best.pt'
            elif model_name == 'GeoAware_combined_core_nano':
                args['pretrained_model_path'] = ['/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt', '/phileo_data/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_19.pt']
            elif model_name == 'GeoAware_core_autoencoder_nano':
                args['pretrained_model_path'] = '/phileo_data/pretrained_models/23112023_CoreVAE_pretraining_reduce_on_plateau/CoreVAE_best.pt'
            elif model_name == 'prithvi':
                args['pretrained_model_path'] = '/phileo_data/pretrained_models/Prithvi_100M.pt'
                args['batch_size'] = 4
            elif model_name == 'SatMAE':
                args['pretrained_model_path'] = '/phileo_data/pretrained_models/SatMAE_pretrain-vit-large-e199.pth'
                args['batch_size'] = 8
            elif model_name == 'vit_cnn':
                args['pretrained_model_path'] = '/phileo_data/pretrained_models/31102023_MaskedAutoencoderViT/MaskedAutoencoderViT_ckpt.pt'
                args['batch_size'] = 8
            elif model_name == 'vit_cnn_gc':
                args['pretrained_model_path'] = '/phileo_data/pretrained_models/03112023_MaskedAutoencoderGroupChannelViT/MaskedAutoencoderGroupChannelViT_ckpt.pt'
                args['batch_size'] = 4

            if model_name == 'vit_cnn_base':
                args['batch_size'] = 8
            if model_name != 'core_unet_nano' or model_name != 'vit_cnn_base':
                for freeze in [False]:
                    args['freeze_pretrained'] = freeze
                    training_script.main(**vars(args))
            else:
                args['pretrained_model_path'] = None
                training_script.main(**vars(args))

        # experiment_plots.main(folder=f'/phileo_data/experiments/nshot/{downstream_task}', plot_title=f'nshot experiment on {downstream_task}',
        #                       filter_on=['CoreUnet','Pretrained_frozen','Pretrained_unfrozen', 'Pretrained_contrastive_frozen', 'Pretrained_contrastive_unfrozen'],
        #                       downstream_task=downstream_task, metric=metric, y_logscale=False, x_logscale=False)

