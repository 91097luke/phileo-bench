import sys; sys.path.append("../")
import torch
import torch.nn as nn
from utils.cnn_utils import get_activation, get_normalization

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)

        return x * y.expand_as(x)
class CoreCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm="batch", activation="relu", padding="same", residual=True):
        super(CoreCNNBlock, self).__init__()

        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeeze = SE_Block(self.out_channels)

        self.match_channels = nn.Identity()
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(norm, out_channels),
            )

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=self.out_channels)
        self.norm2 = get_normalization(norm, self.out_channels)
        
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=1)
        self.norm3 = get_normalization(norm, self.out_channels)


    def forward(self, x):
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))

        x = x * self.squeeze(x)

        if self.residual:
            x = x + self.match_channels(identity)

        x = self.activation(x)

        return x

class CoreEncoderBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, norm="batch", activation="relu", padding="same"):
        super(CoreEncoderBlock, self).__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.blocks = []
        for i in range(self.depth):
            _in_channels = self.in_channels if i == 0 else self.out_channels
            block = CoreCNNBlock(_in_channels, self.out_channels, norm=self.norm, activation=self.activation, padding=self.padding)

            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)

        before_downsample = x
        x = self.downsample(x)

        return x, before_downsample


class CoreAttentionBlock(nn.Module):
    def __init__(self,
        lower_channels,
        higher_channels, *,
        norm="batch",
        activation="relu",
        padding="same",
    ):
        super(CoreAttentionBlock, self).__init__()

        self.lower_channels = lower_channels
        self.higher_channels = higher_channels
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.expansion = 4
        self.reduction = 4

        if self.lower_channels != self.higher_channels:
            self.match = nn.Sequential(
                nn.Conv2d(self.higher_channels, self.lower_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(self.norm, self.lower_channels),
            )

        self.compress = nn.Conv2d(self.lower_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)
        self.attn_c_reduction = nn.Linear(self.lower_channels * (self.reduction ** 2), self.lower_channels * self.expansion)
        self.attn_c_extention = nn.Linear(self.lower_channels * self.expansion, self.lower_channels)

    def forward(self, x, skip):
        if x.size(1) != skip.size(1):
            x = self.match(x)
        x = x + skip
        x = self.activation(x)

        attn_spatial = self.compress(x)
        attn_spatial = self.sigmoid(attn_spatial)

        attn_channel = self.attn_c_pool(x)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.attn_c_reduction(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.attn_c_extention(attn_channel)
        attn_channel = attn_channel.reshape(x.size(0), x.size(1), 1, 1)
        attn_channel = self.sigmoid(attn_channel)

        return attn_spatial, attn_channel


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
        self.match_channels = CoreCNNBlock(self.in_channels * 2, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)
        self.attention = CoreAttentionBlock(self.in_channels, self.in_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)

        self.blocks = []
        for _ in range(self.depth):
            block = CoreCNNBlock(self.out_channels, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)
            self.blocks.append(block)

        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, skip): # y is the skip connection
        x = self.upsample(x)
        attn_s, attn_c = self.attention(x, skip)
        x = torch.cat([x, (skip * attn_s) + (skip + attn_c)], dim=1)
        x = self.match_channels(x)

        for i in range(self.depth):
            x = self.blocks[i](x)

        return x


class CoreUnet(nn.Module):
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="relu", norm="batch", padding="same"):
        super(CoreUnet, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation
        self.norm = norm
        self.padding = padding

        self.dims = [v // 2 for v in self.dims]

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

    def forward(self, x):
        skip_connections = []
        
        x = self.stem(x)
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        x = self.bridge(x)

        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            x = block(x, skip)

        x = self.head(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)

        return x



class CoreEncoder(nn.Module):
    def __init__(self, *, input_dim=10, output_dim=1, depths=None, dims=None, clamp_output=False, clamp_min=0.0, clamp_max=1.0, activation="relu", norm="batch", padding="same"):
        super(CoreEncoder, self).__init__()

        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.clamp_output = clamp_output
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.activation = activation
        self.norm = norm
        self.padding = padding

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.stem = CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding)

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

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], self.output_dim),
        )

    def forward(self, x):
        x = self.stem(x)

        for block in self.encoder_blocks:
            x, _ = block(x)

        x = self.head(x)

        if self.clamp_output:
            x = torch.clamp(x, self.clamp_min, self.clamp_max)

        return x

def CoreUnet_atto(**kwargs):
    model = CoreUnet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def CoreUnet_femto(**kwargs):
    model = CoreUnet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def CoreUnet_pico(**kwargs):
    model = CoreUnet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def CoreUnet_nano(**kwargs):
    model = CoreUnet(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def CoreUnet_tiny(**kwargs):
    model = CoreUnet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def CoreUnet_base(**kwargs):
    model = CoreUnet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def CoreUnet_large(**kwargs):
    model = CoreUnet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def CoreUnet_huge(**kwargs):
    model = CoreUnet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def Core_atto(**kwargs):
    model = CoreEncoder(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def Core_femto(**kwargs):
    model = CoreEncoder(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def Core_pico(**kwargs):
    model = CoreEncoder(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def Core_nano(**kwargs):
    model = CoreEncoder(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def Core_tiny(**kwargs):
    model = CoreEncoder(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def Core_base(**kwargs):
    model = CoreEncoder(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def Core_large(**kwargs):
    model = CoreEncoder(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def Core_huge(**kwargs):
    model = CoreEncoder(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = CoreUnet_femto(
        input_dim=10,
        output_dim=1,
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )

# ====================================================================================================
# Total params: 2,326,921
# Trainable params: 2,326,921
# Non-trainable params: 0
# Total mult-adds (G): 16.61
# ====================================================================================================
# Input size (MB): 5.24
# Forward/backward pass size (MB): 1992.11
# Params size (MB): 9.31
# Estimated Total Size (MB): 2006.66
# ====================================================================================================