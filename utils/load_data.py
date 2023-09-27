# Standard Library
import os
from glob import glob

# External Libraries
import buteo as beo
import numpy as np

# PyTorch
import torch
from torch.utils.data import DataLoader
from utils import config_lc

LC_MAP = config_lc.lc_model_map


def callback_preprocess(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    y = y.astype(np.float32, copy=False)

    return x_norm, y


def callback_preprocess_landcover(x, y):
    x_norm = np.empty_like(x, dtype=np.float32)
    np.divide(x, 10000.0, out=x_norm)

    u,inv = np.unique(y,return_inverse = True)
    y = np.array([LC_MAP[x] for x in u])[inv].reshape(y.shape)

    return x_norm, y


def callback_postprocess_decoder(x, y):
    x = beo.channel_last_to_first(x)
    y = beo.channel_last_to_first(y)

    return torch.from_numpy(x), torch.from_numpy(y)


def callback_decoder(x, y):
    x, y = callback_preprocess(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def callback_decoder_landcover(x, y):
    x, y = callback_preprocess_landcover(x, y)
    x, y = callback_postprocess_decoder(x, y)

    return x, y


def load_data(x_train, y_train, x_val, y_val, x_test, y_test, device, with_augmentations=False, num_workers=0,
              batch_size=16, land_cover=False):

    """
    Loads the data from the data folder.
    """
    if land_cover:
        cb_decoder = callback_decoder_landcover
    else:
        cb_decoder = callback_decoder

    if with_augmentations:
        if land_cover:
            cb_preprocess = callback_preprocess_landcover
        else:
            cb_preprocess = callback_preprocess

        ds_train = beo.DatasetAugmentation(
            x_train, y_train,
            callback_pre_augmentation=cb_preprocess,
            callback_post_augmentation=callback_postprocess_decoder,
            augmentations=[
                beo.AugmentationRotationXY(p=0.2, inplace=True),
                beo.AugmentationMirrorXY(p=0.2, inplace=True),
                beo.AugmentationCutmix(p=0.2, inplace=True),
                beo.AugmentationNoiseNormal(p=0.2, inplace=True),
            ]
        )
    else:
        ds_train = beo.Dataset(x_train, y_train, callback=cb_decoder)

    ds_test = beo.Dataset(x_test, y_test, callback=cb_decoder)
    ds_val = beo.Dataset(x_val, y_val, callback=cb_decoder)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                          drop_last=False, generator=torch.Generator(device=device))
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                         drop_last=False, generator=torch.Generator(device=device))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers,
                        drop_last=False, generator=torch.Generator(device=device))

    return dl_train, dl_test, dl_val

def main():
    print()

if __name__ == '__main__':
    main()
