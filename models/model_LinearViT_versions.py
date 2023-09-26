from models.model_LinearViT import LinearViT
import torch.nn as nn
from functools import partial
def LinearViT_base(**kwargs):
    model = LinearViT(patch_size=4,
                      embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def LinearViT_large(**kwargs):
    model = LinearViT(patch_size=4,
                      embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def LinearViT_huge(**kwargs):
    model = LinearViT(patch_size=4,
                      embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model