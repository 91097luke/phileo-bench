import random
import time

import torch

# random.seed(time.time())

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import buteo as beo

from utils.data_protocol import protocol_fewshot
from utils import config_lc
from utils import config_kg

from utils import Prithvi_100M_config

# statistics used to normalize images before passing to the model
MEANS_PRITHVI = np.array(Prithvi_100M_config.data_mean).reshape(1, 1, -1)
STDS_PRITHVI = np.array(Prithvi_100M_config.data_std).reshape(1, 1, -1)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def render_s2_as_rgb(arr, channel_first=False):
    # If there are nodata values, lets cast them to zero.
    if np.ma.isMaskedArray(arr):
        arr = np.ma.getdata(arr.filled(0))

    if channel_first:
        arr = beo.channel_first_to_last(arr)

    if arr.shape[-1] == 6:
        arr = (arr * STDS_PRITHVI) + MEANS_PRITHVI
        np.divide(arr, 10000.0, out=arr)

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

def visualize_lc_classification(x, y, y_pred=None, images=5, channel_first=False, num_classes=11, labels=None, save_path=None):

    if images > x.shape[0]:
        images = x.shape[0]

    rows = images
    columns = 1
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

        label = y[idx]
        pred = softmax(y_pred[idx])

        max_class = np.argmax(label)
        max_class_pred = np.argmax(pred)

        s1 = (f"Label: Class = {labels[max_class]} "
               f"\n Percentage = {label[max_class]} ")

        s2 = (f"Prediction: Class = {labels[max_class_pred]} "
               f"\n Percentage = {pred[max_class_pred]} ")

        plt.text(25, 25, s1, fontsize=18, bbox=dict(fill=True))

        plt.text(25, 65, s2, fontsize=18, bbox=dict(fill=True))


    fig.tight_layout()

    del x
    del y
    del y_pred

    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


def visualize_vae(images, labels, outputs, num_images=5, channel_first=False,save_path=None ):
    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()

    rows = num_images
    columns = 2
    i = 0
    fig, axes = plt.subplots(nrows=rows, ncols=columns, figsize=(10 * columns, 10 * rows))
    reconstruction, meta_data, embeddings_ssl = outputs

    images = np.einsum('nchw->nhwc', images)
    reconstruction = np.einsum('nchw->nhwc', reconstruction.detach().cpu().numpy())

    for idx in range(0, num_images):
        arr_x = images[idx]
        arr_y = reconstruction[idx]

        rgb_x = render_s2_as_rgb(arr_x, False)
        rgb_y = render_s2_as_rgb(arr_y, False)

        kg_label = labels[idx, :31]
        co_ordinate_labels = labels[idx, 31:34]
        time_labels = labels[idx, 34:36]

        coord_out  = meta_data[0][idx]
        time_out = meta_data[1][idx]
        kg_out = meta_data[2][idx]

        lat, long = decode_coordinates(co_ordinate_labels)
        lat_pred, long_pred = decode_coordinates(coord_out.detach().cpu().numpy())

        doy = decode_date(time_labels)
        doy_pred = decode_date(time_out.detach().cpu().numpy())

        climate = config_kg.kg_map[int(np.argmax([kg_label]))]['climate_class_str']
        climate_pred = config_kg.kg_map[int(np.argmax([kg_out.detach().cpu().numpy()]))]['climate_class_str']

        s1 = (f"Prediction: lat-long = {np.round(lat_pred, 2), np.round(long_pred, 2)} "
              f"\n climate = {climate_pred} "
              f"\n DoY = {doy_pred}")

        s2 = (f"Label: lat-long = {np.format_float_positional(lat, 2), np.format_float_positional(long, 2)} "
              f"\n climate = {climate} "
              f"\n DoY = {doy}")


        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_x)

        plt.text(25, 25, s1, fontsize=18, bbox=dict(fill=True))

        plt.text(25, 65, s2, fontsize=18, bbox=dict(fill=True))

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

def visualize_paper():
    vmin = 0


    lc_map_names = config_lc.lc_raw_classes
    lc_map = config_lc.lc_model_map
    lc_map_inverted = {v: k for k, v in zip(lc_map.keys(), lc_map.values())}
    vmax = len(lc_map)

    cmap = (matplotlib.colors.ListedColormap(config_lc.lc_color_map.values()))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)


    images = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_s2.npy')
    lc_labels = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_label_lc.npy')
    road_labels = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_label_roads.npy')
    building_labels = np.load('/phileo_data/downstream/downstream_dataset_patches_np/europe_1_train_label_building.npy')

    rows = 25

    columns = 4
    i = 0
    fig = plt.figure(figsize=(10 * columns, 10 * rows))

    indexes = random.sample(range(0, images.shape[0]), rows)
    for idx in indexes:
        rgb_image = render_s2_as_rgb(images[idx], channel_first=False)

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(rgb_image)
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)

        u, inv = np.unique(lc_labels[idx], return_inverse=True)
        y = np.array([lc_map[x] for x in u])[inv].reshape(lc_labels[idx].shape)
        plt.imshow(np.squeeze(y), vmin=vmin, vmax=vmax, cmap=cmap)
        patches = [mpatches.Patch(color=cmap(norm(lc_map[u])), label=lc_map_names[u]) for u in
                   np.unique(lc_labels[idx])]
        plt.legend(handles=patches)
        plt.axis('on')

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(building_labels[idx]), vmin=vmin, vmax=1, cmap='magma')
        plt.axis('on')
        plt.grid()

        i = i + 1
        fig.add_subplot(rows, columns, i)
        plt.imshow(np.squeeze(road_labels[idx]), vmin=vmin, vmax=1, cmap='magma')
        plt.axis('on')
        plt.grid()

    fig.tight_layout()
    plt.savefig('visualization_of_labels.png')
    plt.close()

if __name__ == '__main__':
    visualize_paper()