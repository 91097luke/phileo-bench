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
    parser.add_argument('--filter_on', type=str, nargs='*', required=False, default=['CoreUnet','Pretrained_frozen','Pretrained_unfrozen'])
    parser.add_argument('--downstream_task', type=str, required=False, default=None)

    return parser

def main(folder, plot_title, y_logscale, x_logscale, metric, filter_on, downstream_task):
    fig = plt.figure(figsize=(8,5))

    mode = args.filter_on
    task = f"_{args.downstream_task}"  if args.downstream_task is not None else '' #'_building_' or '_lc_'
    metric = args.metric #'mse' or 'acc'

    for m in mode:
        files = glob.glob(f"{folder}/*{m}*/*.json")

        y = []
        x = []
        print()

        for file in files:

            if task in file:
                f = open(file)
                data = json.load(f)
                x.append(data['training_parameters']['train_samples'])
                y.append(data['test_metrics'][metric])

        plt.scatter(x, y, label=m)


    plt.legend()
    plt.title(args.plot_title)
    if args.y_logscale:
        plt.yscale("log")
    if args.x_logscale:
        plt.xscale("log")
    plt.grid()
    plt.ylabel(metric)
    plt.xlabel('n training samples')
    plt.savefig(os.path.join(args.folder, f"test11_{metric}{task}.png"))

    plt.close('all')

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    main(**vars(args))


