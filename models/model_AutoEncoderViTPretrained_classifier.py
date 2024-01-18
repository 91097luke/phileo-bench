from models.model_SatMAE import SatMAE_Classifier
from models.model_CoreCNN import CoreCNNBlock
import torch.nn as nn
import torch
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed, Block
from utils.transformer_utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from models.model_AutoEncoderViTPretrained import ViTEncoder


class ViTCNN_Classifier(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, chw:tuple=(10, 64, 64), patch_size:int=4, output_dim:int=10,
                 embed_dim=768, depth=12, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 ):
        super().__init__()

        # Attributes
        self.chw = chw  # (C, H, W)
        self.in_c = chw[0]
        self.img_size = chw[1]
        self.patch_size = patch_size
        self.output_dim = output_dim

        # --------------------------------------------------------------------------
        # encoder specifics
        self.vit_encoder = ViTEncoder(chw=chw, 
                                      patch_size=patch_size, output_dim=output_dim,
                                      embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.classification_head = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=int(embed_dim / 2)),
                                                 nn.LayerNorm(int(embed_dim / 2)),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=int(embed_dim / 2), out_features=output_dim)
                                                 )

    def forward(self, x):
        x = self.vit_encoder(x)
        # select cls token
        x = x[:, 0, :]
        x = self.classification_head(x)
        return x


class ViTCNN_gc_Classifier(SatMAE_Classifier):

    def __init__(self, **kwargs):
        super(ViTCNN_gc_Classifier, self).__init__(**kwargs)


def vit_base_gc_classifier(**kwargs):
    model = ViTCNN_gc_Classifier(
        channel_embed=256, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_gc_classifier(**kwargs):
    model = ViTCNN_gc_Classifier(
        channel_embed=256, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_gc_classifier(**kwargs):
    model = ViTCNN_gc_Classifier(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_classifier(**kwargs):
    model = ViTCNN_Classifier(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_classifier(**kwargs):
    model = ViTCNN_Classifier(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_classifier(**kwargs):
    model = ViTCNN_Classifier(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_cnn_gc_classifier(checkpoint, img_size=128, patch_size=4, in_chans=10, output_dim=1, freeze_body=True, **kwargs):

    model = vit_base_gc_classifier(img_size=img_size, patch_size=patch_size, in_chans=in_chans, output_dim=output_dim, **kwargs)
    state_dict = model.vit_encoder.state_dict()

    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # load pre-trained model
    msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
    print(msg)

    if freeze_body:
        for _, param in model.vit_encoder.named_parameters():
            param.requires_grad = False

    return model


def vit_cnn_classifier(checkpoint, img_size=128, patch_size=4, in_chans=10, output_dim=1, freeze_body=True, **kwargs):

    model = vit_large_classifier(chw=(in_chans, img_size, img_size), patch_size=patch_size, output_dim=output_dim,  **kwargs)
    state_dict = model.vit_encoder.state_dict()

    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # load pre-trained model
    msg = model.vit_encoder.load_state_dict(checkpoint, strict=False)
    print(msg)

    if freeze_body:
        for _, param in model.vit_encoder.named_parameters():
            param.requires_grad = False

    return model

    return model
