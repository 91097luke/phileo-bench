import argparse
import functools
import os
from typing import List

import numpy as np
import rasterio
import torch
import yaml
from einops import rearrange

from Prithvi import MaskedAutoencoderViT

NO_DATA = -9999
NO_DATA_FLOAT = 0.0001
PERCENTILES = (0.1, 99.9)


def process_channel_group(orig_img, new_img, channels, data_mean, data_std):
    """ Process *orig_img* and *new_img* for RGB visualization. Each band is rescaled back to the
        original range using *data_mean* and *data_std* and then lowest and highest percentiles are
        removed to enhance contrast. Data is rescaled to (0, 1) range and stacked channels_first.

    Args:
        orig_img: torch.Tensor representing original image (reference) with shape = (bands, H, W).
        new_img: torch.Tensor representing image with shape = (bands, H, W).
        channels: list of indices representing RGB channels.
        data_mean: list of mean values for each band.
        data_std: list of std values for each band.

    Returns:
        torch.Tensor with shape (num_channels, height, width) for original image
        torch.Tensor with shape (num_channels, height, width) for the other image
    """

    stack_c = [], []

    for c in channels:
        orig_ch = orig_img[c, ...]
        valid_mask = torch.ones_like(orig_ch, dtype=torch.bool)
        valid_mask[orig_ch == NO_DATA_FLOAT] = False

        # Back to original data range
        orig_ch = (orig_ch * data_std[c]) + data_mean[c]
        new_ch = (new_img[c, ...] * data_std[c]) + data_mean[c]

        # Rescale (enhancing contrast)
        min_value, max_value = np.percentile(orig_ch[valid_mask], PERCENTILES)

        orig_ch = torch.clamp((orig_ch - min_value) / (max_value - min_value), 0, 1)
        new_ch = torch.clamp((new_ch - min_value) / (max_value - min_value), 0, 1)

        # No data as zeros
        orig_ch[~valid_mask] = 0
        new_ch[~valid_mask] = 0

        stack_c[0].append(orig_ch)
        stack_c[1].append(new_ch)

    # Channels first
    stack_orig = torch.stack(stack_c[0], dim=0)
    stack_rec = torch.stack(stack_c[1], dim=0)

    return stack_orig, stack_rec


def read_geotiff(file_path: str):
    """ Read all bands from *file_path* and return image + meta info.

    Args:
        file_path: path to image file.

    Returns:
        np.ndarray with shape (bands, height, width)
        meta info dict
    """

    with rasterio.open(file_path) as src:
        img = src.read()
        meta = src.meta

    return img, meta


def save_geotiff(image, output_path: str, meta: dict):
    """ Save multi-band image in Geotiff file.

    Args:
        image: np.ndarray with shape (bands, height, width)
        output_path: path where to save the image
        meta: dict with meta info.
    """

    with rasterio.open(output_path, "w", **meta) as dest:
        for i in range(image.shape[0]):
            dest.write(image[i, :, :], i + 1)

    return


def _convert_np_uint8(float_image: torch.Tensor):

    image = float_image.numpy() * 255.0
    image = image.astype(dtype=np.uint8)

    return image


def load_example(file_paths: List[str], mean: List[float], std: List[float]):
    """ Build an input example by loading images in *file_paths*.

    Args:
        file_paths: list of file paths .
        mean: list containing mean values for each band in the images in *file_paths*.
        std: list containing std values for each band in the images in *file_paths*.

    Returns:
        np.array containing created example
        list of meta info for each image in *file_paths*
    """

    imgs = []
    metas = []

    for file in file_paths:
        img, meta = read_geotiff(file)

        # Rescaling (don't normalize on nodata)
        img = np.moveaxis(img, 0, -1)   # channels last for rescaling
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, (img - mean) / std)

        imgs.append(img)
        metas.append(meta)

    imgs = np.stack(imgs, axis=0)    # num_frames, H, W, C
    imgs = np.moveaxis(imgs, -1, 0).astype('float32')  # C, num_frames, H, W
    imgs = np.expand_dims(imgs, axis=0)  # add batch dim

    return imgs, metas


def run_model(model: torch.nn.Module, input_data: torch.Tensor, mask_ratio: float, device: torch.device):
    """ Run *model* with *input_data* and create images from output tokens (mask, reconstructed + visible).

    Args:
        model: MAE model to run.
        input_data: torch.Tensor with shape (B, C, T, H, W).
        mask_ratio: mask ratio to use.
        device: device where model should run.

    Returns:
        3 torch.Tensor with shape (B, C, T, H, W).
    """

    with torch.no_grad():
        x = input_data.to(device)

        _, pred, mask = model(x, mask_ratio)

    # Create mask and prediction images (un-patchify)
    mask_img = model.unpatchify(mask.unsqueeze(-1).repeat(1, 1, pred.shape[-1])).detach().cpu()
    pred_img = model.unpatchify(pred).detach().cpu()

    # Mix visible and predicted patches
    rec_img = input_data.clone()
    rec_img[mask_img == 1] = pred_img[mask_img == 1]  # binary mask: 0 is keep, 1 is remove

    # Switch zeros/ones in mask images so masked patches appear darker in plots (better visualization)
    mask_img = (~(mask_img.to(torch.bool))).to(torch.float)

    return rec_img, mask_img


def save_rgb_imgs(input_img, rec_img, mask_img, channels, mean, std, output_dir, meta_data):
    """ Wrapper function to save Geotiff images (original, reconstructed, masked) per timestamp.

    Args:
        input_img: input torch.Tensor with shape (C, T, H, W).
        rec_img: reconstructed torch.Tensor with shape (C, T, H, W).
        mask_img: mask torch.Tensor with shape (C, T, H, W).
        channels: list of indices representing RGB channels.
        mean: list of mean values for each band.
        std: list of std values for each band.
        output_dir: directory where to save outputs.
        meta_data: list of dicts with geotiff meta info.
    """

    for t in range(input_img.shape[1]):
        rgb_orig, rgb_pred = process_channel_group(orig_img=input_img[:, t, :, :],
                                                   new_img=rec_img[:, t, :, :],
                                                   channels=channels, data_mean=mean,
                                                   data_std=std)

        rgb_mask = mask_img[channels, t, :, :] * rgb_orig

        # Saving images

        save_geotiff(image=_convert_np_uint8(rgb_orig),
                     output_path=os.path.join(output_dir, f"original_rgb_t{t}.tiff"),
                     meta=meta_data[t])

        save_geotiff(image=_convert_np_uint8(rgb_pred),
                     output_path=os.path.join(output_dir, f"predicted_rgb_t{t}.tiff"),
                     meta=meta_data[t])

        save_geotiff(image=_convert_np_uint8(rgb_mask),
                     output_path=os.path.join(output_dir, f"masked_rgb_t{t}.tiff"),
                     meta=meta_data[t])


def save_imgs(rec_img, mask_img, mean, std, output_dir, meta_data):
    """ Wrapper function to save Geotiff images (reconstructed, mask) per timestamp.

    Args:
        rec_img: reconstructed torch.Tensor with shape (C, T, H, W).
        mask_img: mask torch.Tensor with shape (C, T, H, W).
        mean: list of mean values for each band.
        std: list of std values for each band.
        output_dir: directory where to save outputs.
        meta_data: list of dicts with geotiff meta info.
    """

    mean = torch.tensor(np.asarray(mean)[:, None, None])  # C H W
    std = torch.tensor(np.asarray(std)[:, None, None])

    for t in range(rec_img.shape[1]):

        # Back to original data range
        rec_img_t = ((rec_img[:, t, :, :] * std) + mean).to(torch.int16)

        mask_img_t = mask_img[:, t, :, :].to(torch.int16)

        # Saving images

        save_geotiff(image=rec_img_t,
                     output_path=os.path.join(output_dir, f"predicted_t{t}.tiff"),
                     meta=meta_data[t])

        save_geotiff(image=mask_img_t,
                     output_path=os.path.join(output_dir, f"mask_t{t}.tiff"),
                     meta=meta_data[t])


def main(data_files: List[str], yaml_file_path: str, checkpoint: str, output_dir: str,
         mask_ratio: float, rgb_outputs: bool):

    os.makedirs(output_dir, exist_ok=True)

    # Get parameters --------

    with open(yaml_file_path, 'r') as f:
        params = yaml.safe_load(f)

    # data related
    num_frames = len(data_files)
    img_size = params['img_size']
    bands = params['bands']
    mean = params['data_mean']
    std = params['data_std']

    # model related
    depth = params['depth']
    patch_size = params['patch_size']
    embed_dim = params['embed_dim']
    num_heads = params['num_heads']
    tubelet_size = params['tubelet_size']
    decoder_embed_dim = params['decoder_embed_dim']
    decoder_num_heads = params['decoder_num_heads']
    decoder_depth = params['decoder_depth']

    batch_size = params['batch_size']

    mask_ratio = params['mask_ratio'] if mask_ratio is None else mask_ratio

    print(f"\nTreating {len(data_files)} files as {len(data_files)} time steps from the same location\n")
    if len(data_files) != 3:
        print("The original model was trained for 3 time steps (expecting 3 files). \nResults with different numbers of timesteps may vary")

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Using {device} device.\n")

    # Loading data ---------------------------------------------------------------------------------

    input_data, meta_data = load_example(file_paths=data_files, mean=mean, std=std)

    # Create model and load checkpoint -------------------------------------------------------------

    model = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=len(bands),
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=4.,
            norm_layer=functools.partial(torch.nn.LayerNorm, eps=1e-6),
            norm_pix_loss=False)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--> Model has {total_params:,} parameters.\n")

    model.to(device)

    state_dict = torch.load(checkpoint, map_location=device)
    # discard fixed pos_embedding weight
    del state_dict['pos_embed']
    del state_dict['decoder_pos_embed']
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint}")

    # Running model --------------------------------------------------------------------------------

    model.eval()
    channels = [bands.index(b) for b in ['B04', 'B03', 'B02']]  # BGR -> RGB

    # Reflect pad if not divisible by img_size
    original_h, original_w = input_data.shape[-2:]
    pad_h = img_size - (original_h % img_size)
    pad_w = img_size - (original_w % img_size)
    input_data = np.pad(input_data, ((0, 0), (0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    # Build sliding window
    batch = torch.tensor(input_data, device='cpu')
    windows = batch.unfold(3, img_size, img_size).unfold(4, img_size, img_size)
    h1, w1 = windows.shape[3:5]
    windows = rearrange(windows, 'b c t h1 w1 h w -> (b h1 w1) c t h w', h=img_size, w=img_size)

    # Split into batches if number of windows > batch_size
    num_batches = windows.shape[0] // batch_size if windows.shape[0] > batch_size else 1
    windows = torch.tensor_split(windows, num_batches, dim=0)

    # Run model
    rec_imgs = []
    mask_imgs = []
    for x in windows:
        rec_img, mask_img = run_model(model, x, mask_ratio, device)
        rec_imgs.append(rec_img)
        mask_imgs.append(mask_img)

    rec_imgs = torch.concat(rec_imgs, dim=0)
    mask_imgs = torch.concat(mask_imgs, dim=0)

    # Build images from patches
    rec_imgs = rearrange(rec_imgs, '(b h1 w1) c t h w -> b c t (h1 h) (w1 w)',
                         h=img_size, w=img_size, b=1, c=len(bands), t=num_frames, h1=h1, w1=w1)
    mask_imgs = rearrange(mask_imgs, '(b h1 w1) c t h w -> b c t (h1 h) (w1 w)',
                          h=img_size, w=img_size, b=1, c=len(bands), t=num_frames, h1=h1, w1=w1)

    # Cut padded images back to original size
    rec_imgs_full = rec_imgs[..., :original_h, :original_w]
    mask_imgs_full = mask_imgs[..., :original_h, :original_w]
    batch_full = batch[..., :original_h, :original_w]

    # Build output images
    if rgb_outputs:
        for d in meta_data:
            d.update(count=3, dtype='uint8', compress='lzw', nodata=0)

        save_rgb_imgs(batch_full[0, ...], rec_imgs_full[0, ...], mask_imgs_full[0, ...],
                      channels, mean, std, output_dir, meta_data)
    else:
        for d in meta_data:
            d.update(compress='lzw', nodata=0)

        save_imgs(rec_imgs_full[0, ...], mask_imgs_full[0, ...], mean, std, output_dir, meta_data)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser('MAE run inference', add_help=False)

    parser.add_argument('--data_files', required=True, type=str, nargs='+',
                        help='Path to the data files. Assumes multi-band files.')
    parser.add_argument('--yaml_file_path', type=str, required=True,
                        help='Path to yaml file containing model training parameters.')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to a checkpoint file to load from.')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Path to the directory where to save outputs.')
    parser.add_argument('--mask_ratio', default=None, type=float,
                        help='Masking ratio (percentage of removed patches). '
                             'If None (default) use same value used for pretraining.')
    parser.add_argument('--rgb_outputs', action='store_true',
                        help='If present, output files will only contain RGB channels. '
                             'Otherwise, all bands will be saved.')
    args = parser.parse_args()

    main(**vars(args))

