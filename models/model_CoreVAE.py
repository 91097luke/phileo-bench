import torch
import torch.nn as nn
from utils.training_utils import get_activation, get_normalization, SE_Block
from models.model_CoreCNN import CoreCNNBlock, CoreEncoderBlock, CoreAttentionBlock, CoreDecoderBlock
from torchvision import transforms


# class CoreDecoderBlock(nn.Module):
#     def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu", padding="same"):
#         super(CoreDecoderBlock, self).__init__()
#
#         self.depth = depth
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.activation_blocks = activation
#         self.activation = get_activation(activation)
#         self.norm = norm
#         self.padding = padding
#
#         self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
#         self.match_channels = CoreCNNBlock(self.in_channels, self.out_channels, norm=self.norm,
#                                            activation=self.activation_blocks, padding=self.padding)
#         # self.attention = CoreAttentionBlock(self.in_channels, self.in_channels, norm=self.norm,
#         #                                     activation=self.activation_blocks, padding=self.padding)
#
#         self.blocks = []
#         for _ in range(self.depth):
#             block = CoreCNNBlock(self.out_channels, self.out_channels, norm=self.norm,
#                                  activation=self.activation_blocks, padding=self.padding)
#             self.blocks.append(block)
#
#         self.blocks = nn.Sequential(*self.blocks)
#
#     def forward(self, x):
#         x = self.upsample(x)
#         # attn_s, attn_c = self.attention(x, skip)
#         # x = torch.cat([x, (skip * attn_s) + (skip + attn_c)], dim=1)
#         x = self.match_channels(x)
#
#         for i in range(self.depth):
#             x = self.blocks[i](x)
#
#         return x

class ScaleSkip2D(nn.Module):
    def __init__(self, c):
        super(ScaleSkip2D, self).__init__()
        self.c = c

        self.y_skipscale = nn.Parameter(torch.ones(1, self.c, 1, 1)) # use as loss-punishment
        self.y_skipbias = nn.Parameter(torch.zeros(1, self.c, 1, 1))

    def forward(self, y): # y is the skip-connect
        # clip negative values to zero
        y = self.y_skipscale * y + self.y_skipbias
        return y


class CoreVAE(nn.Module):
    def __init__(self, *,
        input_dim=10,
        output_dim=10,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
    ):
        super(CoreVAE, self).__init__()

        self.depths = depths
        self.dims = dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.dims = [v // 2 for v in self.dims]

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation,
                         padding=self.padding),
        )

        self.encoder_blocks = []
        for i in range(len(self.depths)):
            encoder_block = CoreEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.encoder_blocks.append(encoder_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

        self.decoder_blocks = []

        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = CoreDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.decoder_blocks.append(decoder_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)

        self.scale_skips = []
        self.scale_skip_weights = [(i+1)/len(self.dims) for i in range(len(self.dims))]
        for i in reversed(range(len(self.encoder_blocks))):
            scale_skip = ScaleSkip2D(c=self.dims[i])
            self.scale_skips.append(scale_skip)

        self.scale_skips = nn.ModuleList(self.scale_skips)

        self.bridge = nn.Sequential(
            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation,
                         padding=self.padding),
        )

        self.head = nn.Sequential(
            CoreCNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

        self.head_climate = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], 31),
        )

        self.head_coord = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], 3),
        )

        self.head_time = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], 2),
        )

    def forward_body(self, x):
        skip_connections = []

        x = self.stem(x)
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.bridge(x)

        out_coords = self.head_coord(x)
        out_time = self.head_time(x)
        out_kg = self.head_climate(x)

        scale_skip_loss = []
        for i, block in enumerate(self.decoder_blocks):
            skip = skip_connections.pop()
            skip = self.scale_skips[i](skip)
            x = block(x, skip)

            scale_skip_loss.append(self.scale_skips[i].y_skipscale.abs().mean() * self.scale_skip_weights[i])

        return x, (out_coords, out_time, out_kg), sum(scale_skip_loss)/len(scale_skip_loss)

    def forward(self, x):

        x, (out_coords, out_time, out_kg), latent = self.forward_body(x)
        reconstruction = self.head(x)

        return reconstruction, (out_coords, out_time, out_kg), latent


def CoreVAE_nano(**kwargs):
    """
    Total params: 16,400,685
    Trainable params: 16,400,685
    Non-trainable params: 0
    Total mult-adds (G): 50.95
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 3388.57
    Params size (MB): 65.60
    Estimated Total Size (MB): 3459.42
    """
    model = CoreVAE(depths=[2, 2, 8, 2], dims=[160, 320, 640, 1280], **kwargs)
    return model


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 4
    CHANNELS = 10
    HEIGHT = 128
    WIDTH = 128

    model = CoreVAE_nano(
        input_dim=10,
        output_dim=1,
    )

    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )

    sd = model.state_dict()
    torch.save(sd, 'test.pt')