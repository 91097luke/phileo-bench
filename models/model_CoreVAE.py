import torch
import torch.nn as nn
from utils.training_utils import get_activation, get_normalization, SE_Block
from models.model_CoreCNN import CoreCNNBlock, CoreEncoderBlock, CoreAttentionBlock


class CoreDecoderBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu", padding="same"):
        super(CoreDecoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_blocks = activation
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.match_channels = CoreCNNBlock(self.in_channels, self.out_channels, norm=self.norm,
                                           activation=self.activation_blocks, padding=self.padding)
        # self.attention = CoreAttentionBlock(self.in_channels, self.in_channels, norm=self.norm,
        #                                     activation=self.activation_blocks, padding=self.padding)

        self.blocks = []
        for _ in range(self.depth):
            block = CoreCNNBlock(self.out_channels, self.out_channels, norm=self.norm,
                                 activation=self.activation_blocks, padding=self.padding)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        x = self.upsample(x)
        # attn_s, attn_c = self.attention(x, skip)
        # x = torch.cat([x, (skip * attn_s) + (skip + attn_c)], dim=1)
        x = self.match_channels(x)

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x


class CoreVAE(nn.Module):
    def __init__(self, *,
        input_dim=10,
        output_dim=1,
        img_size=128,
        latent_dim=512,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
    ):
        super(CoreVAE, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.dims = [v // 2 for v in self.dims]

        self.latent_dim = latent_dim
        self.linear_dim = int(((img_size // (2 ** 4)) ** 2) * self.dims[-1]) # 2 ** 4 because of 4 downsamples
        self.img_size = img_size

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = nn.Sequential(
            CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
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

        self.bridge = nn.Sequential(
            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding),
        )

        self.head = nn.Sequential(
            CoreCNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

        self.linear_encode = nn.Sequential(
            nn.Linear(self.linear_dim, self.latent_dim * 2),
        )

        self.linear_decode = nn.Sequential(
            nn.Linear(self.latent_dim, self.linear_dim),
            nn.LayerNorm(self.linear_dim),
            nn.Mish(),
        )
    def sample(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = logvar.exp()
        z = eps * std + mu
        return z

    def gaussian_latent(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear_encode(x)
        mu, logvar = x.chunk(2, dim=1)
        x = self.sample(mu, logvar)
        x = self.linear_decode(x)
        x = x.reshape(-1, self.dims[-1], self.img_size // (2 ** 4), self.img_size // (2 ** 4)) # 2 ** 4 because of 4 downsamples
        return x, mu, logvar

    # x = self.linear_encode(x.reshape(-1, self.linear_dim))
    # mu, logvar = x.chunk(2, dim=1)
    # x = self.sample(mu, logvar)
    # x = self.linear_decode(x)
    # x = x.reshape(-1, self.dims[-1], self.img_size // (2 ** 4), self.img_size // (2 ** 4)) # 2 ** 4 because of 4 downsamples
    def forward_body(self, x):
        skip_connections = []

        x = self.stem(x)
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.bridge(x)
        x, mu, logvar = self.gaussian_latent(x)

        for block in self.decoder_blocks:
            # skip = skip_connections.pop()
            x = block(x)
        return x, mu, logvar

    def forward(self, x):

        x, mu, logvar = self.forward_body(x)
        x = self.head(x)

        return x, mu, logvar


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
    model = CoreVAE(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 4
    CHANNELS = 10
    HEIGHT = 128
    WIDTH = 128

    model = CoreVAE(
        input_dim=10,
        output_dim=1,
    )

    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )