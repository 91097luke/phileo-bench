import random
import time
random.seed(time.time())

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


from utils.data_protocol import protocol_fewshot
from utils import config_lc
from utils import config_kg
import buteo as beo


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
    plt.close()


def visualize_lc(x, y, y_pred=None, images=5, channel_first=False, vmin=0,save_path=None):
    lc_map_names = config_lc.lc_raw_classes
    lc_map = config_lc.lc_model_map
    lc_map_inverted = {v: k for k, v in zip(lc_map.keys(), lc_map.values())}
    vmax = len(lc_map)

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


def decode_date(encoded_date):
    doy_sin, doy_cos = encoded_date

    doy = np.arctan2((2 * doy_sin - 1), (2 * doy_cos - 1)) * 365 / (2 * np.pi)

    if doy < 1:
        doy += 365

    return np.array([np.round(doy)])


def decode_coordinates(encoded_coords):
    lat_enc, long_sin, long_cos = encoded_coords

    lat = -lat_enc * 180 + 90

    long = np.arctan2((2 * long_sin - 1), (2 * long_cos - 1)) * 360 / (2 * np.pi)

    return np.array([lat, long])


def encode_coordinates(coords):
    lat, long = coords

    lat = (-lat + 90) / 180

    long_sin = (np.sin(long * 2 * np.pi / 360) + 1) / 2

    long_cos = (np.cos(long * 2 * np.pi / 360) + 1) / 2

    return np.array([lat, long_sin, long_cos], dtype=np.float32)


def visualise_contrastive(images, pred_sim, est_sim, y, channel_first=True, save_path=None):


    rows = columns = 5

    c = 0

    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    for i in range(0, columns):
        for j in range(0, rows):
            c = c + 1
            arr = images[j]
            rgb_image = render_s2_as_rgb(arr, channel_first)
            fig.add_subplot(rows, columns, c)

            kg_label = y[j, :31]
            co_ordinate_labels = y[j, 31:34]
            time_lables = y[j, 34:]


            lat, long = decode_coordinates(co_ordinate_labels)

            doy = decode_date(time_lables)
            climate = config_kg.kg_map[int(np.argmax([kg_label]))]['climate_class_str']

            s1 = f"Similarity  : est = {np.round(est_sim[i, j], 2)} \n pred = {np.round(pred_sim[i, j], 2)} \n"

            s2 = f"Label: lat-long = {np.round(lat, 2), np.round(long, 2)} \n climate = {climate} \n DoY = {doy}"

            plt.text(25, 25, s1, fontsize=18, bbox=dict(fill=True))

            plt.text(25, 45, s2, fontsize=18, bbox=dict(fill=True))

            plt.imshow(rgb_image)

        plt.axis('on')

        plt.grid()
        plt.savefig(save_path)


def visualize_arcface(x, y, save_path=None):
    X_embedded = TSNE(n_components=2).fit_transform(x)
    kg_map = config_kg.kg_map

    fig, ax = plt.subplots(figsize=(16, 16))
    for g in np.unique(y):
        ix = np.where(y == g)
        ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c=np.array(kg_map[g]['colour_code'])/255, label=kg_map[g]['description'])
    ax.legend(loc ="upper left")

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()

# def main():
#     s2 = np.load('/phileo_data/mini_foundation/mini_foundation_patches_np/patches_labeled/10_points_filtered_22_10/S2A_MSIL2A_20221028T184501_N0400_R070_T11SLA_20221028T222756_val_s2.npy')
#     t = beo.raster_to_metadata('/phileo_data/mini_foundation/mini_foundation_tifs/10_points_filtered_22_10/S2A_MSIL2A_20221006T075811_N0400_R035_T36MUV_20221006T120858_label_terrain.tif')
#     print()
#
# if __name__ == '__main__':
#     main()