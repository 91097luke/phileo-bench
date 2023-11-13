from matplotlib import pyplot as plt
import matplotlib
# import PyQt5
# matplotlib.use('QtAgg')
import glob
import json
import os
import argparse


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

def main(folder, plot_title, metric, filter_on, downstream_task, y_logscale=False, x_logscale=False): # plot_title, y_logscale, x_logscale
    fig = plt.figure(figsize=(8,5))

    mode = filter_on
    task = f"_{downstream_task}"
    metric = metric #'mse' or 'acc'

    n_shots = [ 500, 5000, 50000, 100000, 200000]


    for m in mode:
        files = []
        for n_shot in n_shots:
            files.extend(glob.glob(f"{folder}/*{m}*_{n_shot}/*.json"))

        y = []
        x = []

        for file in files:

            if task in file:
                f = open(file)
                data = json.load(f)
                x.append(data['training_parameters']['n_shot'])
                # y.append(data['training_info']['best_epoch'])
                y.append(data['test_metrics'][metric])

        plt.plot(x, y, label=m, alpha=0.8, linestyle='--', marker='o')


    plt.legend()
    plt.title(plot_title)
    if y_logscale:
        plt.yscale("log")
    if x_logscale:
        plt.xscale("log")
    plt.grid()
    plt.ylabel(metric)
    plt.xlabel('n training samples')
    plt.savefig(os.path.join(folder, f"test_{metric}{task}.png"))

    plt.close('all')

if __name__ == '__main__':
    # parser = get_args()
    # args = parser.parse_args()
    # main(**vars(args))
    task = 'roads'

    main(folder=f'/phileo_data/experiments/n_shots/{task}/', plot_title=f'nshot experiment on {task} downstream task',
                          filter_on=['CoreUnet', 'CoreEncoderGeoPretrained_frozen', 'CoreEncoderGeoPretrained_unfrozen',
                                     'Pretrained_contrastive_frozen', 'Pretrained_contrastive_unfrozen',
                                     'Pretrained_basic_frozen', 'Pretrained_basic_unfrozen',
                                     'Pretrained_combined_frozen', 'Pretrained_combined_unfrozen',
                                     'AutoEncoderViTPretrained_unfrozen', 'AutoEncoderViTPretrained_frozen'],
                          downstream_task=task, metric='mse', y_logscale=True, x_logscale=True)


