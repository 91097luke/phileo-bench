from matplotlib import pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')
import glob
import json
import os
import argparse

model_labels = {'CoreUnet': {'name':'UNET_fully_sup', 'colour':'green'},
                'Resnet50': {'name':'RESNET_fully_sup', 'colour':'red'},
                'ViTCNN_building': {'name':'VITCNN_fully_sup', 'colour':'blue'},
                'ViTCNN_lc': {'name':'VITCNN_fully_sup', 'colour':'blue'},
                'ViTCNN_roads': {'name':'VITCNN_fully_sup', 'colour':'blue'},
                'CoreEncoderGeoPretrained_mh_pred_unfrozen': {'name':'GeoAware_UNET_ft', 'colour':'darkolivegreen'},
                'CoreEncoderGeoPretrained_mh_pred_frozen': {'name':'GeoAware_UNET_lp', 'colour':'limegreen'},
                'ViTCNN_unfrozen': {'name':'Pretrained_VITCNN_ft', 'colour':'lightblue'},
                'ViTCNN_frozen': {'name':'Pretrained_VITCNN_lp', 'colour':'deepskyblue'},
                'ViTCNN_gc_unfrozen': {'name':'Pretrained_VITCNN_GC_ft', 'colour':'aquamarine'},
                'ViTCNN_gc_wSkip_unfrozen': {'name':'Pretrained_VITCNN_GC_wSkip_ft', 'colour':'royalblue'},
                'ViTCNN_gc_frozen': {'name':'Pretrained_VITCNN_GC_lp', 'colour':'lightseagreen'},
                'SatMAE_unfrozen': {'name': 'SatMAE_ft', 'colour':'steelblue'},
                'SatMAE_frozen': {'name':'SatMAE_lp', 'colour':'lightslategrey'},
                'Prithvi_unfrozen': {'name':'Prithvi_ft', 'colour':'darkorchid'},
                'Prithvi_frozen': {'name':'Prithvi_lp', 'colour':'violet'},
                'Seco_unfrozen': {'name':'Seco_ft', 'colour':'sienna'},
                'Seco_frozen': {'name':'Seco_lp', 'colour':'lightsalmon'},
                }

model_labels_cl = { 'CoreEncoder_building_classification': {'name':'Encoder_benchmark', 'colour':'green'},
                    'CoreEncoder_lc_classification': {'name':'Encoder_benchmark', 'colour':'green'},
                    'CoreEncoder_roads_classification': {'name':'Encoder_benchmark', 'colour':'green'},
                    'CoreEncoderGeoPretrained_Classifier_mh_pred_unfrozen': {'name':'GeoAware_UNET_ft', 'colour':'darkolivegreen'},
                    'CoreEncoderGeoPretrained_Classifier_mh_pred_frozen': {'name':'GeoAware_UNET_lp', 'colour':'limegreen'},
                    'ViTCNN_building_classification': {'name':'VITCNN_benchmark', 'colour':'blue'},
                    'ViTCNN_lc_classification': {'name':'VITCNN_benchmark', 'colour':'blue'},
                    'ViTCNN_roads_classification': {'name':'VITCNN_benchmark', 'colour':'blue'},
                    'ViTCNN_Classifier_unfrozen': {'name':'Pretrained_VITCNN_ft', 'colour':'lightblue'},
                    'ViTCNN_Classifier_frozen': {'name':'Pretrained_VITCNN_lp', 'colour':'deepskyblue'},
                    'ViTCNN_gc_Classifier_unfrozen': {'name':'Pretrained_VITCNN_GC_ft', 'colour':'aquamarine'},
                    'ViTCNN_gc_Classifier_frozen': {'name':'Pretrained_VITCNN_GC_lp', 'colour':'lightseagreen'},
                    'SatMAE_Classifier_unfrozen': {'name': 'SatMAE_ft', 'colour':'steelblue'},
                    'SatMAE_Classifier_frozen': {'name':'SatMAE_lp', 'colour':'lightslategrey'},
                    'PrithviClassifier_unfrozen': {'name':'Prithvi_ft', 'colour':'darkorchid'},
                    'PrithviClassifier_frozen': {'name':'Prithvi_lp', 'colour':'violet'},
                    'Resnet50': {'name':'RESNET_benchmark', 'colour':'red'},
                    'Seco_Classifier_unfrozen': {'name':'Seco_ft', 'colour':'sienna'},
                    'Seco_Classifier_frozen': {'name':'Seco_lp', 'colour':'lightsalmon'}}


def get_args():
    parser = argparse.ArgumentParser(description='Plot experiments test loss')
    parser.add_argument('--folder', type=str, required=True,
                        help='Experiment folder name')
    parser.add_argument('--plot_title', type=str, required=True,)
    parser.add_argument('--y_logscale', type=bool, required=False, default=True)
    parser.add_argument('--x_logscale', type=bool, required=False, default=False)
    parser.add_argument('--metric', type=str, required=False, default='acc')
    parser.add_argument('--filter_on', type=str, nargs='*', required=False, default=['CoreUnet','Pretrained_frozen','Pretrained_unfrozen', 'Pretrained_contrastive_frozen'])
    parser.add_argument('--downstream_task', type=str, required=False, default=None)

    return parser

def main(folder, plot_title, metric, filter_on, downstream_task, y_logscale=False, x_logscale=False, legend=True): # plot_title, y_logscale, x_logscale
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)

    mode = filter_on
    task = f"_{downstream_task}"
    metric = metric #'mse' or 'acc'

    n_shots = [50, 100, 500, 1000, 5000]


    for m in mode:
        files = []
        for n_shot in n_shots:
            files.extend(glob.glob(f"{folder}/*{m}*_{n_shot}/*.json"))

        y = []
        x = []
        if downstream_task.split('_')[-1] == 'classification':
            label = model_labels_cl[m]['name']
            c = model_labels_cl[m]['colour']
        else:
            label = model_labels[m]['name']
            c = model_labels[m]['colour']

        for file in files:

            if task in file:
                f = open(file)
                data = json.load(f)
                x.append(data['training_parameters']['n_shot'])

                if metric == 'best_epoch':
                    best_val_loss = min(data['plot_info']['val_losses'])
                    _val_loss = best_val_loss + (best_val_loss*0.02)
                    epoch, val_loss = min(enumerate(data['plot_info']['val_losses']), key=lambda x: abs(x[1] - _val_loss))
                    y.append(epoch)
                else:
                    y.append(data['test_metrics'][metric])

        ax.plot(x, y, label=label, color=c, alpha=0.6, linestyle='--', marker='o')


    # plt.legend()
    ax.set_title(plot_title)
    if y_logscale:
        ax.set_yscale("log")
    if x_logscale:
        ax.set_xscale("log")
    plt.grid('on')
    if metric == 'best_epoch':
        ax.set_ylabel('best epoch')
    else:
        ax.set_ylabel(metric)
    ax.set_xlabel('n training samples per region')
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.15), ncol=4,)
        
        # plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0))
        plt.savefig(os.path.join(folder, f"test_{metric}{task}.png"), bbox_extra_artists=(lgd, ), bbox_inches='tight')
    else:
        plt.savefig(os.path.join(folder, f"test_{metric}{task}.png"))


    plt.close('all')

if __name__ == '__main__':
    # parser = get_args()
    # args = parser.parse_args()
    # main(**vars(args))
    task = 'lc_classification'

    filter_on_seg = ['CoreUnet', 'CoreEncoderGeoPretrained_mh_pred_unfrozen', 'CoreEncoderGeoPretrained_mh_pred_frozen', 
                    'Resnet50', 'Seco_unfrozen', 'Seco_frozen',
                    f'ViTCNN_{task}', 'ViTCNN_unfrozen', 'ViTCNN_frozen', 'ViTCNN_gc_unfrozen', 'ViTCNN_gc_frozen', 'ViTCNN_gc_wSkip_unfrozen',
                    'SatMAE_unfrozen', 'SatMAE_frozen',
                    'Prithvi_unfrozen', 'Prithvi_frozen']
    
    filter_on_cl = [f'CoreEncoder_{task}', 'CoreEncoderGeoPretrained_Classifier_mh_pred_unfrozen', 
                    'CoreEncoderGeoPretrained_Classifier_mh_pred_frozen', 
                    'Resnet50', 'Seco_Classifier_unfrozen', 'Seco_Classifier_frozen', 
                    f'ViTCNN_{task}', 'ViTCNN_Classifier_unfrozen', 'ViTCNN_Classifier_frozen', 
                    'ViTCNN_gc_Classifier_unfrozen', 'ViTCNN_gc_Classifier_frozen', 
                    'SatMAE_Classifier_unfrozen', 'SatMAE_Classifier_frozen',
                    'PrithviClassifier_unfrozen', 'PrithviClassifier_frozen']

    if task.split('_')[-1] == 'classification':
        filter_on = filter_on_cl
    else:
        filter_on = filter_on_seg

    main(folder=f'/home/phimultigpu/phileo_NFS/phileo_data/experiments/nshot_experiments_eo-hpc/{task}/', plot_title=f'nshot experiment on {task} downstream task',
                          filter_on=filter_on,
                          downstream_task=task, metric='acc', y_logscale=True, x_logscale=True, legend=True)


