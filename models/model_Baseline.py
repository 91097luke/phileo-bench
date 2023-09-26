import torch
import torch.nn as nn
from utils.training_utils import get_activation, get_normalization


class BaselineNet(nn.Module):
    def __init__(self, *, input_dim=10, output_dim=1, activation="relu", norm="batch", padding="same"):
        super(BaselineNet, self).__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding
        self.activation = get_activation(activation)

        self.size = 256

        self.net = nn.Sequential(
            nn.Conv2d(self.input_dim, self.size, kernel_size=5, padding=self.padding),
            nn.Conv2d(self.size, self.size, kernel_size=5, padding=self.padding),
            get_normalization(self.norm, self.size),
            self.activation,
            nn.Conv2d(self.size, self.size, kernel_size=3, padding=self.padding),
            nn.Conv2d(self.size, self.size, kernel_size=3, padding=self.padding),
            get_normalization(self.norm, self.size),
            self.activation,
            nn.Conv2d(self.size, self.size, kernel_size=1, padding=self.padding),
            nn.Conv2d(self.size, self.size, kernel_size=1, padding=self.padding),
            get_normalization(self.norm, self.size),
            self.activation,
            nn.Conv2d(self.size, 1, kernel_size=1, padding=self.padding),
        )

    def initialize_weights(self, std=0.02):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.net(x)

        return x



if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 32
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    model = BaselineNet(
        input_dim=10,
        output_dim=1,
    )
    
    model(torch.randn((BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)))

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
