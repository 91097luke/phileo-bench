import torch
import torch.nn as nn
from copy import deepcopy

from torchvision.models import resnet50, ResNet50_Weights
from models.model_SatMAE import CoreDecoderBlock, CoreCNNBlock


class Resnet50(nn.Module):
    def __init__(self, output_dim=1, imagenet_weights=True, decoder_norm='batch', decoder_padding='same',
                 decoder_activation='relu', decoder_depths=[2, 2, 8, 2], decoder_dims=[160, 320, 640, 1280]):
        super(Resnet50, self).__init__()
        if imagenet_weights:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.encoder = resnet50(weights=weights)
        else:
            self.encoder = resnet50()

        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

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
            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=decoder_norm, activation=decoder_activation,
                         padding=decoder_padding),
        )

        self.decoder_upsample_block = nn.Sequential(CoreDecoderBlock(depth=1, in_channels=2048,
                                                                     out_channels=self.dims[-1], norm=decoder_norm,
                                                                     activation=decoder_activation, padding=decoder_padding))

        self.decoder_head = nn.Sequential(
            CoreCNNBlock(self.dims[0], self.dims[0], norm=decoder_norm, activation=decoder_activation, padding=decoder_padding),
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

class Resnet50_Classifier(nn.Module):
    def __init__(self, output_dim=1, imagenet_weights=True):
        super(Resnet50_Classifier, self).__init__()
        if imagenet_weights:
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = resnet50(weights=weights)
        else:
            model = resnet50()

        self.encoder = nn.Sequential(*list(model.children())[:-1])
        self.classification_head = nn.Sequential(nn.Flatten(start_dim=1, end_dim=-1),
                                  nn.Linear(2048, output_dim))

    def forward(self, x):
        # order S2 bands: 0-B02, 1-B03, 2-B04, 3-B08, 4-B05, 5-B06, 6-B07, 7-B8A, 8-B11, 9-B12
        x = x[:, (2, 1, 0), :, :] # select RGB bands
        x = self.encoder(x)
        x = self.classification_head(x)
        return x

def resnet(imagenet_weights, output_dim=1, freeze_body=True, classifier=False, **kwargs):

    if classifier:
        model = Resnet50_Classifier(output_dim=output_dim, imagenet_weights=imagenet_weights)
        


    else:
        model = Resnet50(output_dim=output_dim, imagenet_weights=imagenet_weights, **kwargs)

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
    CHANNELS = 10
    HEIGHT = 224
    WIDTH = 224

    model = Resnet50_Classifier()
    model.cpu()

    x = model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )

    sd = model.state_dict()
    torch.save(sd, 'test.pt')