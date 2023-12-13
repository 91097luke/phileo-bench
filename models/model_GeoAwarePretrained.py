import torch
import torch.nn as nn
from models.model_Mixer import Mixer, CNNBlock
from models.model_CoreCNN import CoreEncoder, CoreCNNBlock, CoreUnet, CoreUnet_combined
from models.model_Mixer_versions import Mixer_tiny
from collections import OrderedDict


class MixerGeoPretrained(nn.Module):
    def __init__(self,
                output_dim,
                checkpoint,
                mixer_kwargs,
                freeze_body=True,
                ):
        self.freeze_body = freeze_body
        super(MixerGeoPretrained, self).__init__()
        assert output_dim == mixer_kwargs['output_dim'], f"output dim {output_dim} but mixer will output {mixer_kwargs['output_dim']}"

        self.geomixer = Mixer(**mixer_kwargs) #MixerGeoAware(**mixer_kwargs)
        model_dict = self.geomixer.state_dict()

        pretrained_dict = {k: v for k, v in checkpoint.items() if not k.startswith('head')}

        model_dict.update(pretrained_dict) 

        self.geomixer.load_state_dict(model_dict)

        if self.freeze_body:
            for name, param in self.geomixer.named_parameters():
                if name in pretrained_dict.keys():
                    param.requires_grad = False
                    print(name)

    def forward(self, identity):
        x = self.geomixer.forward(identity)
        # x = self.head(x)
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
        
        self.coreunet = CoreUnet(**core_encoder_kwargs)

        assert output_dim == core_encoder_kwargs['output_dim'], f"output dim {output_dim} but core_unet will output {core_encoder_kwargs['output_dim']}"
        
        unet_weights, encoder_weights = self.load_encoder_weights(checkpoint=checkpoint, unet=self.coreunet)
        self.coreunet.load_state_dict(unet_weights)

        if self.freeze_body:
            for name, param in self.coreunet.named_parameters():
                if name in encoder_weights.keys():
                    param.requires_grad = False

    def load_encoder_weights(self,checkpoint, unet):
        model_sd = unet.state_dict()
        shared_weights = OrderedDict()
        for k, name in zip(checkpoint.keys(),model_sd.keys()):
            if k.startswith('head'):
                continue
            v = checkpoint[k]
            if checkpoint[k].size() == model_sd[name].size():
                shared_weights[name]=v
            else:
                raise ValueError(f"weights of pretrained encoder layer {k} are not compatible with model layer {name}")
        model_sd.update(shared_weights)

        return model_sd, shared_weights

    def forward(self, identity):
        x = self.coreunet.forward(identity)
        # x = self.head(x)
        return x


class CoreEncoderGeoAutoEncoder(nn.Module):
    def __init__(self,
                 output_dim,
                 checkpoint,
                 core_encoder_kwargs,
                 freeze_body=True,
                 ):
        self.freeze_body = freeze_body
        super(CoreEncoderGeoAutoEncoder, self).__init__()

        self.coreunet = CoreUnet(**core_encoder_kwargs)

        assert output_dim == core_encoder_kwargs[
            'output_dim'], f"output dim {output_dim} but core_unet will output {core_encoder_kwargs['output_dim']}"

        for k in ['decoder_blocks.0.match_channels.match_channels.0.weight',
                  'decoder_blocks.0.match_channels.conv1.weight',
                  'decoder_blocks.1.match_channels.match_channels.0.weight',
                  'decoder_blocks.1.match_channels.conv1.weight',
                  'decoder_blocks.2.match_channels.match_channels.0.weight',
                  'decoder_blocks.2.match_channels.conv1.weight',
                  'decoder_blocks.3.match_channels.conv1.weight',
                  'head.1.weight',
                  'head.1.bias']:

            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint[k]

        self.coreunet.load_state_dict(checkpoint, strict=False)

        if self.freeze_body:
            for name, param in self.coreunet.named_parameters():
                if name.startswith('stem') or name.startswith('encoder'):
                    param.requires_grad = False

    def forward(self, identity):
        x = self.coreunet.forward(identity)
        # x = self.head(x)
        return x


class CoreEncoderGeoPretrained_combined(nn.Module):
    def __init__(self,
                 output_dim,
                 checkpoint_1,
                 checkpoint_2,
                 core_encoder_kwargs,
                 freeze_body=True,
                 ):
        self.freeze_body = freeze_body
        super(CoreEncoderGeoPretrained_combined, self).__init__()

        self.coreunet_combined = CoreUnet_combined(**core_encoder_kwargs)

        assert output_dim == core_encoder_kwargs[
            'output_dim'], f"output dim {output_dim} but core_unet will output {core_encoder_kwargs['output_dim']}"

        unet_weights, encoder_weights = self.load_encoder_weights(checkpoint_1=checkpoint_1, checkpoint_2=checkpoint_2,
                                                                  unet=self.coreunet_combined)
        self.coreunet_combined.load_state_dict(unet_weights)

        if self.freeze_body:
            for name, param in self.coreunet_combined.named_parameters():
                if name in encoder_weights.keys():
                    param.requires_grad = False

    def load_encoder_weights(self, checkpoint_1, checkpoint_2, unet):
        model_sd = unet.state_dict()
        shared_weights = OrderedDict()

        for i, checkpoint in enumerate([checkpoint_1, checkpoint_2]):
            for k in checkpoint.keys():
                if k.startswith('head'):
                    continue
                v = checkpoint[k]
                k_split = k.split('.')
                if k.startswith('stem'):
                    name = '.'.join([f"{k_split[0]}_{i+1}.0"] + k_split[1:])
                else:
                    name = '.'.join([f"{k_split[0]}_{i+1}"] + k_split[1:])
                if checkpoint[k].size() == model_sd[name].size():
                    shared_weights[name] = v
                else:
                    raise ValueError(f"weights of pretrained encoder layer {k} are not compatible with model layer {name}")
        model_sd.update(shared_weights)

        return model_sd, shared_weights

    def forward(self, identity):
        x = self.coreunet_combined.forward(identity)
        # x = self.head(x)
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