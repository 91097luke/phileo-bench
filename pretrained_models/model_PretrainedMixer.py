import torch
import torch.nn as nn
from models.model_Mixer import Mixer, CNNBlock
from models.model_CoreCNN import CoreEncoder, CoreCNNBlock
from models.model_Mixer_versions import Mixer_tiny


class MixerGeoAware(Mixer):
    def __init__(self, **kwargs):

        super(MixerGeoAware, self).__init__(**kwargs)

        assert isinstance(self.embedding_dims, list), "embedding_dims must be a list."
        assert isinstance(self.patch_sizes, list), "patch_sizes must be a list."
        assert len(self.embedding_dims) == len(self.patch_sizes), "embedding_dims and patch_sizes must be the same length."

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.embedding_dims[-1], self.output_dim),
        )

class CoreEncoderGeoAware(CoreEncoder):
    def __init__(self, **kwargs):

        super(CoreEncoderGeoAware, self).__init__(**kwargs)

        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], self.output_dim),
        )
    


class MixerGeoPretrained(nn.Module):
    def __init__(self,
                output_dim,
                checkpoint,
                mixer_kwargs,
                freeze_body=True,
                ):
        self.freeze_body = freeze_body
        super(MixerGeoPretrained, self).__init__()

        self.geomixer = MixerGeoAware(**mixer_kwargs)
        self.geomixer.load_state_dict(checkpoint)
        
        self.head = nn.Sequential(
            CNNBlock(mixer_kwargs['embedding_dims'][-1], mixer_kwargs['embedding_dims'][-1]),
            CNNBlock(mixer_kwargs['embedding_dims'][-1], mixer_kwargs['embedding_dims'][-1]),
            nn.Conv2d(mixer_kwargs['embedding_dims'][-1], output_dim, 1, padding=0),
        )
        if self.freeze_body:
            for name, param in self.geomixer.named_parameters():
                param.requires_grad = False
                if name.startswith('mixer_layers.6'):
                    param.requires_grad  = True
                    print(name)
                if name.startswith('mixer_layers.7'):
                    param.requires_grad  = True
                    print(name)
                if name.startswith('skip_layers.6'):
                    param.requires_grad  = True                
                    print(name)
                if name.startswith('head'):
                    param.requires_grad  = True                
                    print(name)
            # for param in self.geomixer.parameters():
            #     param.requires_grad = False

    def forward(self, identity):
        x = self.geomixer.forward_body(identity)
        x = self.head(x)
        return x
    


class CoreEncoderGeoPretrained(nn.Module):
    def __init__(self,
                output_dim,
                checkpoint,
                core_encoder_kwargs,
                freeze_body=True,
                ):
        self.freeze_body = freeze_body
        super(CoreEncoderGeoPretrained, self).__init__()
        self.geocore = CoreEncoderGeoAware(**core_encoder_kwargs)
        self.geocore.load_state_dict(checkpoint)
        
        self.head = nn.Sequential(
            CNNBlock(core_encoder_kwargs['dims'][-1], core_encoder_kwargs['dims'][-1]),
            CNNBlock(core_encoder_kwargs['dims'][-1], core_encoder_kwargs['dims'][-1]),
            nn.Conv2d(core_encoder_kwargs['dims'][-1], output_dim, 1, padding=0),
        )
        if self.freeze_body:
            for param in self.geocore.parameters():
                param.requires_grad = False

    def forward(self, identity):
        x = self.geocore.forward_body(identity)
        x = self.head(x)
        return x
    




def get_mixer_kwargs(chw, output_dim, mixer_size, **kwargs):
    mixer_kwargs = {'chw':chw, 'output_dim':output_dim}
    
    if mixer_size=='mixer_nano':
        mixer_kwargs['embedding_dims'] = [128] * 4 * 2
        mixer_kwargs['patch_sizes'] = [16, 8, 4, 2] * 2
        mixer_kwargs['expansion'] = 2.0
    
    elif mixer_size=='mixer_tiny':
        mixer_kwargs['embedding_dims'] = [128] * 4 * 3
        mixer_kwargs['patch_sizes'] = [16, 8, 4, 2] * 3
        mixer_kwargs['expansion'] = 2.2
    
    elif mixer_size=='mixer_base':
        mixer_kwargs['embedding_dims'] =[128] * 4 * 4
        mixer_kwargs['patch_sizes'] = [16, 8, 4, 2] * 4
        mixer_kwargs['expansion'] = 4

    else:
        raise ValueError
    
    mixer_kwargs.update(kwargs)

    return mixer_kwargs



def get_core_encoder_kwargs(output_dim, input_dim, core_size, **kwargs):
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
   
    core_kwargs.update(kwargs) 
    return core_kwargs



if __name__ == '__main__':

    input = torch.rand((8,10,128,128))
    sd = torch.load('Mixer_last_10.pt')
    mixer_kwargs = {'chw':(10,128,128), 'output_dim':641, 'embedding_dims':[128] * 4 * 2, 'patch_sizes':[16, 8, 4, 2] * 2, 'expansion':2.0}
    mixer_kwargs = get_mixer_kwargs(chw=(10,128,128), output_dim=641, mixer_size='mixer_nano')
    model = MixerGeoPretrained(output_dim=1,checkpoint=sd, mixer_kwargs=mixer_kwargs, freeze_body=True)

    # sd = torch.load('CoreEncoder_last_8.pt')
    # core_kwargs = get_core_encoder_kwargs(output_dim=641, input_dim=10, core_size='core_nano')
    # model = CoreEncoderGeoPretrained(1, checkpoint=sd, core_encoder_kwargs=core_kwargs)


    #model.load_state_dict(sd)
    out = model(input)
    print(out.shape)