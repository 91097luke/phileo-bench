from torchvision import models
from torch import nn
from torch.nn import functional as F

class Resnet50(nn.Module):
    def __init__(self, in_channels, ):
        super(Resnet50, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.backbone.conv1.out_channels,
                                        kernel_size=self.backbone.conv1.kernel_size,
                                        stride=self.backbone.conv1.stride, padding=self.backbone.conv1.padding,
                                        bias=self.backbone.conv1.bias)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x

class Resnet18(nn.Module):
    def __init__(self, in_channels, ):
        super(Resnet18, self).__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=self.backbone.conv1.out_channels,
                                        kernel_size=self.backbone.conv1.kernel_size,
                                        stride=self.backbone.conv1.stride, padding=self.backbone.conv1.padding,
                                        bias=self.backbone.conv1.bias)
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        x = F.normalize(x, p=2, dim=1)
        return x


