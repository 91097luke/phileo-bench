from models.model_AutoEncoderViT import AutoencoderViT
import torch.nn as nn
from functools import partial


def AutoencoderViT_base(**kwargs):
    model = AutoencoderViT(patch_size=4,
                           embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                           decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def AutoencoderViT_large(**kwargs):
    model = AutoencoderViT(patch_size=4,
                           embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                           decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def AutoencoderViT_huge(**kwargs):
    model = AutoencoderViT(patch_size=4,
                           embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                           decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                           norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model