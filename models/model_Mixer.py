import torch
import torch.nn as nn


class ScaleSkip2D(nn.Module):
    def __init__(self, channels, drop_p=0.1):
        super(ScaleSkip2D, self).__init__()

        self.channels = channels

        self.y_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.y_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))
        
        self.x_skipscale = nn.Parameter(torch.ones(1, self.channels, 1, 1))
        self.x_skipbias = nn.Parameter(torch.zeros(1, self.channels, 1, 1))

        if drop_p > 0.:
            self.dropout = nn.Dropout2d(drop_p)
        else:
            self.dropout = None

    def forward(self, x, y):
        if self.dropout:
            x = self.dropout(x)
            y = self.dropout(y)

        y = self.y_skipscale * y + self.y_skipbias
        x = self.x_skipscale * x + self.x_skipbias
        
        return x + y


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, apply_residual=True):
        super(CNNBlock, self).__init__()

        self.apply_residual = apply_residual and in_channels == out_channels
        self.out_channels = out_channels

        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=1, bias=False)

        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
            self.match_norm = nn.BatchNorm2d(out_channels)
        else:
            self.match_channels = None

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation(x)

        x = self.conv3(x)

        if self.match_channels:
            identity = self.match_channels(identity)
            identity = self.match_norm(identity)

        if self.apply_residual:
            x = identity + x

        x = self.activation(x)

        return x


class MLPMixerLayer(nn.Module):
    def __init__(self,
        embed_dims,
        patch_size=16,
        chw=(10, 64, 64),
        expansion=2,
        drop_n=0.0,
    ):
        super(MLPMixerLayer, self).__init__()
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.chw = chw
        self.expansion = expansion

        self.num_patches_height = self.chw[1] // self.patch_size
        self.num_patches_width = self.chw[2] // self.patch_size
        self.num_patches = self.num_patches_height * self.num_patches_width
        self.tokens = round((self.chw[1] * self.chw[2]) / self.num_patches)

        self.bn1 = nn.BatchNorm2d(self.num_patches)
        self.mix_channel = nn.Sequential( # B, P, T, C
            nn.Linear(self.embed_dims, int(self.embed_dims * self.expansion)),
            nn.ReLU6(),
            nn.Dropout(drop_n) if drop_n > 0. else nn.Identity(),
            nn.Linear(int(self.embed_dims * self.expansion), self.embed_dims),
        )

        self.bn2 = nn.BatchNorm2d(self.num_patches)
        self.mix_patch = nn.Sequential( # B, C, T, P
            nn.Linear(self.num_patches, int(self.num_patches * self.expansion)),
            nn.ReLU6(),
            nn.Dropout(drop_n) if drop_n > 0. else nn.Identity(),
            nn.Linear(int(self.num_patches * self.expansion), self.num_patches),
        )

        self.bn3 = nn.BatchNorm2d(self.num_patches)
        self.mix_token = nn.Sequential( # B, P, C, T
            nn.Linear(self.tokens, int(self.tokens * self.expansion)),
            nn.ReLU6(),
            nn.Dropout(drop_n) if drop_n > 0. else nn.Identity(),
            nn.Linear(int(self.tokens * self.expansion), self.tokens),
        )


    def patchify_batch(self, tensor):
        B, C, _H, _W = tensor.shape
        patch_size = self.patch_size
        num_patches_height = self.num_patches_height
        num_patches_width = self.num_patches_width
        num_patches = self.num_patches

        # Reshape and extract patches
        reshaped = tensor.reshape(B, C, num_patches_height, patch_size, num_patches_width, patch_size)
        transposed = reshaped.permute(0, 2, 4, 1, 3, 5)
        final_patches = transposed.reshape(B, num_patches, C, patch_size ** 2)

        return final_patches


    def unpatchify_batch(self, patches):
        B, _P, C, _T = patches.shape
        _C, H, W = self.chw
        patch_size = self.patch_size
        num_patches_height = self.num_patches_height
        num_patches_width = self.num_patches_width

        # Reverse the patchify process
        reshaped = patches.reshape(B, num_patches_height, num_patches_width, C, patch_size, patch_size)
        transposed = reshaped.permute(0, 3, 1, 4, 2, 5)
        final_tensor = transposed.reshape(B, C, H, W)

        return final_tensor

    def forward(self, x):
        x = self.patchify_batch(x) # B, P, C, T

        x = self.bn1(x)
        mix_channel = x.permute(0, 1, 3, 2) # B, P, T, C
        mix_channel = self.mix_channel(mix_channel)
        mix_channel = mix_channel.permute(0, 1, 3, 2) # B, P, C, T 
        x = x + mix_channel

        x = self.bn2(x)
        mix_patch = x.permute(0, 2, 3, 1) # B, C, T, P
        mix_patch = self.mix_patch(mix_patch)
        mix_patch = mix_patch.permute(0, 3, 1, 2) # B, P, C, T
        x = x + mix_patch

        x = self.bn3(x)
        mix_token = self.mix_token(x)
        x = x + mix_token

        x = self.unpatchify_batch(x) # B, C, H, W

        return x


class Mixer(nn.Module):
    def __init__(self,
        chw,
        output_dim,
        embedding_dims=[64, 64, 64, 64],
        patch_sizes=[16, 8, 4, 2],
        expansion=2,
        drop_n=0.0,
        drop_p=0.0,
        softmax_output=False,
    ):
        super(Mixer, self).__init__()
        self.chw = chw
        self.output_dim = output_dim
        self.embedding_dims = embedding_dims
        self.patch_sizes = patch_sizes
        self.expansion = expansion
        self.drop_n = drop_n
        self.drop_p = drop_p
        self.std = .02
        self.softmax_output = softmax_output
        self.class_boundary = max(patch_sizes)

        assert isinstance(self.embedding_dims, list), "embedding_dims must be a list."
        assert isinstance(self.patch_sizes, list), "patch_sizes must be a list."
        assert len(self.embedding_dims) == len(self.patch_sizes), "embedding_dims and patch_sizes must be the same length."

        self.stem = nn.Sequential(
            CNNBlock(chw[0], self.embedding_dims[0], apply_residual=False),
            CNNBlock(self.embedding_dims[0], self.embedding_dims[0]),
            CNNBlock(self.embedding_dims[0], self.embedding_dims[0]),
        )

        self.chw_2 = (self.embedding_dims[0], self.chw[1] + self.class_boundary, self.chw[2])

        self.mixer_layers = []
        self.matcher_layers = []
        self.skip_layers = []
        self.skip_layers_2 = []
        for i, v in enumerate(patch_sizes):
            if self.embedding_dims[i] != self.embedding_dims[i - 1] and i < len(patch_sizes) - 1 and i != 0:
                self.matcher_layers.append(
                    nn.Conv2d(self.embedding_dims[i - 1], self.embedding_dims[i], 1, padding=0)
                )
            else:
                self.matcher_layers.append(nn.Identity())

            self.mixer_layers.append(
                MLPMixerLayer(
                    self.embedding_dims[i],
                    patch_size=v,
                    chw=self.chw_2,
                    expansion=self.expansion,
                    drop_n=drop_n,
                )
            )

            # We only add skip-connections at every second block, starting at the second block
            if i != 0 and i % 2 == 0:
                self.skip_layers.append(
                    ScaleSkip2D(self.embedding_dims[i], drop_p=drop_p)
                )
                if self.embedding_dims[i] != self.embedding_dims[0]:
                    self.skip_layers_2.append(
                        nn.Conv2d(self.embedding_dims[0], self.embedding_dims[i], 1, padding=0)
                    )
                else:
                    self.skip_layers_2.append(nn.Identity())
            else:
                self.skip_layers.append(nn.Identity())
                self.skip_layers_2.append(nn.Identity())

        self.matcher_layers = nn.ModuleList(self.matcher_layers)
        self.mixer_layers = nn.ModuleList(self.mixer_layers)
        self.skip_layers = nn.ModuleList(self.skip_layers)
        self.skip_layers_2 = nn.ModuleList(self.skip_layers_2)

        self.head = nn.Sequential(
            CNNBlock(self.embedding_dims[-1], self.embedding_dims[-1]),
            CNNBlock(self.embedding_dims[-1], self.embedding_dims[-1]),
            nn.Conv2d(self.embedding_dims[-1], self.output_dim, 1, padding=0),
        )
        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),
        #     nn.Linear(self.embedding_dims[-1], self.output_dim),
        # )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = self.std
            torch.nn.init.trunc_normal_(m.weight, std=std, a=-std * 2, b=std * 2)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_body(self, identity):
        skip = self.stem(identity)
        skip = torch.nn.functional.pad(skip, (0, 0, self.class_boundary, 0), mode="constant", value=0.0)

        x = skip

        for i, layer in enumerate(self.mixer_layers):
            x = self.matcher_layers[i](x)

            # Only add skip-connections after the first layer
            if i != 0 and i % 2 == 0:
                skip_match = self.skip_layers_2[i](skip)
                x = self.skip_layers[i](layer(x), skip_match)
            else:
                x = layer(x)

        x = x[:, :, self.class_boundary:, :]

        return x

    def forward(self, identity):
        x = self.forward_body(identity)
        x = self.head(x)

        if self.softmax_output:
            x = torch.softmax(x, dim=1)

        return x

if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 16
    CHANNELS = 10
    HEIGHT = 128
    WIDTH = 128

    torch.set_default_device("cuda")

    model = Mixer(
        chw=(10, 128, 128),
        output_dim=113,
        # embedding_dims=[32, 32],
        # patch_sizes=[16, 8],
        # embedding_dims=32,
        # expansion=2,
        # drop_n=0.0,
        # drop_p=0.0,
    )

    # model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
