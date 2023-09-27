from matplotlib import pyplot as plt
import matplotlib
import PyQt5
matplotlib.use('QtAgg')
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


def main(folder, plot_title, y_logscale, x_logscale):
    y = []
    x = []
    artifcat_files = glob.glob(f"{folder}/*/*.json")

    for file in artifcat_files:
        f = open(file)
        data = json.load(f)
        x.append(data['training_parameters']['train_samples'])
        y.append(data['training_info']['test_loss'])

    fig = plt.figure()
    plt.scatter(x, y, label='Test Loss', )
    plt.legend()
    plt.title(args.plot_title)
    if args.y_logscale:
        plt.yscale("log")
    if args.x_logscale:
        plt.xscale("log")
    plt.grid()
    plt.ylabel('test loss')
    plt.xlabel('n training samples')
    plt.savefig(os.path.join(args.folder, f"test_losses.png"))

    plt.close('all')

if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    main(**vars(args))


