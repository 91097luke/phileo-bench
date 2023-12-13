from models.model_SatMAE import SatMAE_Classifier, CoreEncoderBlock
from models.model_CoreCNN import CoreCNNBlock
import torch.nn as nn
import torch
from functools import partial
from collections import OrderedDict
from timm.models.vision_transformer import PatchEmbed, Block
from utils.transformer_utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid


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
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(self.img_size, patch_size, self.in_c, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim),  requires_grad=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # learnable with sin-cos embedding init

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # CNN Decoder Blocks:
        self.classification_head = nn.Sequential(nn.Linear(in_features=embed_dim, out_features=int(embed_dim / 2)),
                                                 nn.LayerNorm(int(embed_dim / 2)),
                                                 nn.ReLU(),
                                                 nn.Linear(in_features=int(embed_dim / 2), out_features=output_dim)
                                                 )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        # # global pooling
        # x = x.mean(dim=1)

        # return cls token
        x = x[:, 0, :]

        return x

    def forward(self, x):
        x = self.forward_encoder(x)
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
    state_dict = model.state_dict()

    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    if freeze_body:
        for name, param in model.named_parameters():
            if not name.startswith('classification'):
                param.requires_grad = False

    # load pre-trained model
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model


def vit_cnn_classifier(checkpoint, img_size=128, patch_size=4, in_chans=10, output_dim=1, freeze_body=True, **kwargs):

    model = vit_large_classifier(chw=(in_chans, img_size, img_size), patch_size=patch_size, output_dim=output_dim,  **kwargs)
    state_dict = model.state_dict()

    for k in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint and checkpoint[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

    # load pre-trained model
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)

    if freeze_body:
        for name, param in model.named_parameters():
            if not name.startswith('classification'):
                param.requires_grad = False

    return model
