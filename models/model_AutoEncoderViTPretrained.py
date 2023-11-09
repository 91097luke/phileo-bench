from models.model_AutoEncoderViT_versions import AutoencoderViT_large, AutoencoderViT_huge
import torch.nn as nn
from functools import partial
from collections import OrderedDict


class AutoEncoderViTPretrained(nn.Module):
    def __init__(self,
                 chw,
                 output_dim,
                 checkpoint,
                 freeze_body=True,
                 ):
        self.freeze_body = freeze_body
        super(AutoEncoderViTPretrained, self).__init__()

        self.ae_vit = AutoencoderViT_large(chw=chw, out_chans=10)

        self.ae_vit.load_state_dict(state_dict=checkpoint)
        self.ae_vit.decoder_pred = nn.Linear(self.ae_vit.decoder_pred.in_features, self.ae_vit.patch_size ** 2 * output_dim,
                                             bias=True)

        if self.freeze_body:
            for name, param in self.ae_vit.named_parameters():
                if 'decoder' not in name:
                    param.requires_grad = False

        self.patchify = self.ae_vit.patchify
        self.unpatchify = self.ae_vit.unpatchify

    def forward(self, identity):
        x = self.ae_vit.forward(identity)
        # x = self.head(x)
        return x