import random
import time
random.seed(time.time())

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import buteo as beo

from utils.data_protocol import protocol_fewshot
from utils import config_lc


def render_s2_as_rgb(arr, channel_first=False):
    # If there are nodata values, lets cast them to zero.
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr.filled(0))

    if channel_first:
        arr = beo.channel_first_to_last(arr)
    # Select only Blue, green, and red. Then invert the order to have R-G-B
    rgb_slice = arr[:, :, 0:3][:, :, ::-1]

    # Clip the data to the quantiles, so the RGB render is not stretched to outliers,
    # Which produces dark images.
    rgb_slice = np.clip(
        rgb_slice,
        np.quantile(rgb_slice, 0.02),
        np.quantile(rgb_slice, 0.98),
    )

    # The current slice is uint16, but we want an uint8 RGB render.
    # We normalise the layer by dividing with the maximum value in the image.
    # Then we multiply it by 255 (the max of uint8) to be in the normal RGB range.
    rgb_slice = (rgb_slice / rgb_slice.max()) * 255.0

    # We then round to the nearest integer and cast it to uint8.
    rgb_slice = np.rint(rgb_slice).astype(np.uint8)

    return rgb_slice


def visualize(x, y, y_pred=None, images=5, channel_first=False, vmin=0, vmax=1, save_path=None):
    if images > x.shape[0]:
        images = x.shape[0]

    rows = images
    if y_pred is None:
        columns = 2
    else:
        columns = 3
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for idx in indexes:
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(y[idx]), vmin=vmin, vmax=vmax, cmap='magma')
        plt.axis('on')
        plt.grid()

        if y_pred is not None:
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.squeeze(y_pred[idx]), vmin=vmin, vmax=vmax, cmap='magma')
            plt.axis('on')
            plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def visualize_lc(x, y, y_pred=None, images=5, channel_first=False, vmin=0,save_path=None):
    lc_map_names = config_lc.lc_raw_classes
    lc_map = config_lc.lc_model_map
    lc_map_inverted = {v: k for k, v in zip(lc_map.keys(), lc_map.values())}
    vmax = len(lc_map)

    if images > x.shape[0]:
        images = x.shape[0]
        
    # d = 1 if channel_first else -1
    # # y= y.argmax(axis=d)
    # if y_pred is not None:
    #     y_pred = y_pred.argmax(axis=d)
    cmap = (matplotlib.colors.ListedColormap(config_lc.lc_color_map.values()))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    rows = images
    if y_pred is None:
        columns = 2
    else:
        columns = 3
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, x.shape[0]), images)
    for idx in indexes:
        arr = x[idx]
        rgb_image = render_s2_as_rgb(arr, channel_first)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(y[idx]), vmin=vmin, vmax=vmax, cmap=cmap)
        patches = [mpatches.Patch(color=cmap(norm(u)), label=lc_map_names[lc_map_inverted[u]]) for u in np.unique(y[idx])]
        plt.legend(handles=patches)
        plt.axis('on')
        plt.grid()

        if y_pred is not None:
            i = i + 1
            fig.add_subplot(rows, columns, i)
            plt.imshow(np.squeeze(y_pred[idx]), vmin=vmin, vmax=vmax, cmap=cmap)
            patches = [mpatches.Patch(color=cmap(norm(u)), label=lc_map_names[lc_map_inverted[u]]) for u in np.unique(y_pred[idx])]
            plt.legend(handles=patches)
            plt.axis('on')
            plt.grid()

    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def visualize_reconstruct(x, y, images=5, channel_first=False,save_path=None ):
    rows = images
    columns = 2
    i = 0
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10 * columns, 10 * rows))

    x = np.einsum('nchw->nhwc', x)
    y = np.einsum('nchw->nhwc', y)

    for idx in range(0, images):
        arr_x = x[idx]
        arr_y = y[idx]

        rgb_x = render_s2_as_rgb(arr_x, False)
        rgb_y = render_s2_as_rgb(arr_y, False)


        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_x)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_y)


    fontsize = 96
    axes[0][0].set_title('image', fontdict={'fontsize': fontsize})
    axes[0][1].set_title('recon.', fontdict={'fontsize': fontsize})

    fig.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    plt.clf()
    plt.close()