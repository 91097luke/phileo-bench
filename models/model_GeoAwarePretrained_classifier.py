import torch
import torch.nn as nn
from models.model_Mixer import Mixer, CNNBlock
from models.model_CoreCNN import CoreEncoder
from collections import OrderedDict


class CoreEncoderGeoPretrained_Classifier(nn.Module):
    def __init__(self,
                checkpoint,
                core_encoder_kwargs,
                freeze_body=True,
                ):
        self.freeze_body = freeze_body
        super(CoreEncoderGeoPretrained_Classifier, self).__init__()
        
        self.model = CoreEncoder(**core_encoder_kwargs)

        shared_weights = checkpoint.copy()
        for k in checkpoint.keys():
            if k.startswith('head'):
                del shared_weights[k]


        # load pre-trained model
        msg = self.model.load_state_dict(shared_weights, strict=False)
        print(msg)

        if freeze_body:
            for name, param in self.model.named_parameters():
                if not name.startswith('head'):
                    param.requires_grad = False

    def forward(self, identity):
        x = self.model.forward(identity)
        # x = self.head(x)
        return x


def get_core_encoder_kwargs(output_dim, input_dim, core_size, full_unet=True, **kwargs):
    core_kwargs = {'input_dim':input_dim, 'output_dim':output_dim, 'norm':'batch', 'padding':'same', 'activation':'relu'}

    if core_size=='core_nano':
        core_kwargs['depths']=[2, 2, 8, 2]
        core_kwargs['dims']=[80, 160, 320, 640]

    elif core_size=='core_tiny':
        core_kwargs['depths']=[3, 3, 9, 3]
        core_kwargs['dims']=[96, 192, 384, 768]

    elif core_size=='core_base':
        core_kwargs['depths']=[3, 3, 27, 3]
        core_kwargs['dims']=[128, 256, 512, 1024]

    else:
        raise ValueError
    
    if full_unet:
        core_kwargs['dims'] = [v * 2 for v in core_kwargs['dims']]

   
    core_kwargs.update(kwargs) 
    return core_kwargs



if __name__ == '__main__':

    input = torch.rand((8,10,128,128))
    # sd = torch.load('Mixer_last_10.pt')
    # mixer_kwargs = {'chw':(10,128,128), 'output_dim':641, 'embedding_dims':[128] * 4 * 2, 'patch_sizes':[16, 8, 4, 2] * 2, 'expansion':2.0}
    # mixer_kwargs = get_mixer_kwargs(chw=(10,128,128), output_dim=641, mixer_size='mixer_nano')
    # model = MixerGeoPretrained(output_dim=1,checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=True)

    sd_1 = torch.load('/home/lcamilleri/git_repos/phileo-testbed/models/test.pt')
    core_kwargs = get_core_encoder_kwargs(output_dim=1, input_dim=10, core_size='core_nano')
    model = CoreEncoderGeoAutoEncoder(1, checkpoint=sd_1 ,core_encoder_kwargs=core_kwargs)


    #model.load_state_dict(sd)
    out = model(input)
    print(out.shape)