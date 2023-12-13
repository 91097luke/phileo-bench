import torch
import torch.nn as nn
from models.model_CoreCNN_versions import Core_nano
from collections import OrderedDict


class GeoPretrainedFeatureExtractor(nn.Module):
    def __init__(self,
                 checkpoint,
                 input_channels,
                 freeze_body=True,
                 ):
        self.freeze_body = freeze_body
        super(GeoPretrainedFeatureExtractor, self).__init__()

        self.coreunet = Core_nano(input_dim=input_channels, output_dim=1)

        unet_weights, encoder_weights = self.load_encoder_weights(checkpoint=checkpoint, unet=self.coreunet)
        self.coreunet.load_state_dict(unet_weights)
        self.coreunet.head[2] = torch.nn.Identity()

        if self.freeze_body:
            for name, param in self.coreunet.named_parameters():
                if name in encoder_weights.keys():
                    param.requires_grad = False


    def load_encoder_weights(self, checkpoint, unet):
        model_sd = unet.state_dict()
        shared_weights = OrderedDict()
        checkpoint = torch.load(checkpoint)
        for k, name in zip(checkpoint.keys(), model_sd.keys()):
            if k.startswith('head'):
                continue
            v = checkpoint[k]
            if checkpoint[k].size() == model_sd[name].size():
                shared_weights[name] = v
            else:
                raise ValueError(f"weights of pretrained encoder layer {k} are not compatible with model layer {name}")
        model_sd.update(shared_weights)

        return model_sd, shared_weights
    def forward(self, x):
        x = self.coreunet.forward(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x

if __name__ == '__main__':
    m = GeoPretrainedFeatureExtractor(checkpoint='/phileo_data/GeoAware_results/trained_models/12102023_CoreEncoder_LEO_geoMvMF_augm/CoreEncoder_last_19.pt', input_channels=10)
    t = torch.randn((1, 10, 128, 128))
    o = m(t)