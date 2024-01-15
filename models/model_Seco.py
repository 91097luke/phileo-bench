import torch
import torch.nn as nn
from copy import deepcopy
from models.seco_utils import moco2_module, segmentation
from models.model_SatMAE import CoreDecoderBlock, CoreCNNBlock, CoreEncoderBlock


class Seco(nn.Module):
    def __init__(self, ckpt_path, output_dim=1, decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280]):
        super(Seco, self).__init__()
        model = moco2_module.MocoV2.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.encoder = deepcopy(model.encoder_q)

        # CNN Decoder Blocks:
        self.depths = decoder_depths
        self.dims = decoder_dims
        self.output_dim = output_dim
        self.decoder_blocks = []

        for i in reversed(range(len(self.depths))):
            decoder_block = CoreDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=decoder_norm,
                activation=decoder_activation,
                padding=decoder_padding,
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.decoder_bridge = nn.Sequential(
            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=decoder_norm,
                         activation=decoder_activation,
                         padding=decoder_padding,),
        )

        self.decoder_upsample_block = nn.Sequential(CoreDecoderBlock(depth=1, in_channels=2048,
                                                                     out_channels=self.dims[-1],                 norm=decoder_norm,
                                                                     activation=decoder_activation,
                                                                     padding=decoder_padding,))

        self.decoder_head = nn.Sequential(
            CoreCNNBlock(self.dims[0], self.dims[0], norm=decoder_norm,
                         activation=decoder_activation,
                         padding=decoder_padding,),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

    def forward_decoder(self, x):
        for block in self.decoder_blocks:
            x = block(x)
        return x

    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        x = self.decoder_upsample_block(x)
        x = self.decoder_bridge(x)
        x = self.forward_decoder(x)
        x = self.decoder_head(x)
        return x


class Seco_Classifier(nn.Module):
    def __init__(self, ckpt_path, output_dim=1):
        super(Seco_Classifier, self).__init__()
        model = moco2_module.MocoV2.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.encoder = deepcopy(model.encoder_q)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1,1)),
                                  nn.Flatten(start_dim=1, end_dim=-1),
                                  nn.Linear(2048, 2048),
                                  nn.ReLU(),
                                  nn.Linear(2048, output_dim))

    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        x = self.head(x)

        return x


def seasonal_contrast(checkpoint, output_dim=1, freeze_body=True, classifier=False, **kwargs):

    if classifier:
        model = Seco_Classifier(ckpt_path=checkpoint, output_dim=output_dim)
    else:
        model = Seco(ckpt_path=checkpoint, output_dim=output_dim, **kwargs)

    if freeze_body:
        if classifier:
            for name, param in model.named_parameters():
                if not not name.startswith('classification'):
                    param.requires_grad = False

        else:
            for name, param in model.named_parameters():
                if not name.startswith('decoder'):
                    param.requires_grad = False

    return model


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 4
    CHANNELS = 3
    HEIGHT = 128
    WIDTH = 128

    model = Seco_Classifier(ckpt_path='/phileo_data/pretrained_models/seco_resnet50_1m.ckpt')
    model.cpu()

    x = model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )

    sd = model.state_dict()
    torch.save(sd, 'test.pt')