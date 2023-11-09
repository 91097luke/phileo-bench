import numpy
from datetime import date


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


for downstream_task in ['lc', 'building', 'roads']:

    args['downstream_task'] = downstream_task
    if downstream_task == 'lc':
        args['output_channels'] = 11
        metric = 'acc'
    else:
        args['output_channels'] = 1
        metric = 'msc'

    for n_shot in [50, 500, 5000,]: #  50000, 100000, 200000
        for model_name in ['GeoAware_basic_core_nano']: #['core_unet_nano', 'GeoAware_contrastive_core_nano',
                           #'GeoAware_core_nano']:  # ,'mixer_nano','baseline_cnn','linear_vit_base']:
            args['n_shot'] = n_shot
            args['model_name'] = model_name

            if model_name == 'GeoAware_contrastive_core_nano':
                args['pretrained_model_path'] = '/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/27102023_CoreEncoderMultiHead_geo_reduce_on_plateau/CoreEncoderMultiHead_best.pt'
            elif model_name == 'GeoAware_contrasive_core_nano':
                args['pretrained_model_path'] = '/phileo_data/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_19.pt'
            elif model_name == 'GeoAware_basic_core_nano':
                args['pretrained_model_path'] = '/home/lcamilleri/git_repos/Phileo-contrastive-geographical-expert/trained_models/contrastive/01112023_CoreEncoderMultiHead_geo_pred_geo_reduce_on_plateau/CoreEncoderMultiHead_geo_pred_best.pt'
            elif model_name == 'AutoEncoderVitPretrained':
                args['pretrained_model_path'] = '/phileo_data/pretrained_models/31102023_MaskedAutoencoderViT/MaskedAutoencoderViT_ckpt.pt'

            if model_name != 'core_unet_nano':
                for freeze in [False, True]:
                    args['freeze_pretrained'] = freeze
                    training_script.main(**vars(args))
            else:
                args['pretrained_model_path'] = None
                training_script.main(**vars(args))

        # experiment_plots.main(folder=f'/phileo_data/experiments/nshot/{downstream_task}', plot_title=f'nshot experiment on {downstream_task}',
        #                       filter_on=['CoreUnet','Pretrained_frozen','Pretrained_unfrozen', 'Pretrained_contrastive_frozen', 'Pretrained_contrastive_unfrozen'],
        #                       downstream_task=downstream_task, metric=metric, y_logscale=False, x_logscale=False)

