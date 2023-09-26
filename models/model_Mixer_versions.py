import torch
from models.model_Mixer import Mixer

def Mixer_atto(**kwargs):
    """
    Total params: 3,659,905
    Trainable params: 3,659,905
    Non-trainable params: 0
    Total mult-adds (M): 949.69
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 646.45
    Params size (MB): 14.64
    Estimated Total Size (MB): 663.71
    """    
    model = Mixer(embedding_dims=[32] * 4, patch_sizes=[16, 8, 4, 2], expansion=1, **kwargs)
    return model

def Mixer_femto(**kwargs):
    """
    Total params: 5,528,509
    Trainable params: 5,528,509
    Non-trainable params: 0
    Total mult-adds (G): 3.51
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 1418.20
    Params size (MB): 22.11
    Estimated Total Size (MB): 1442.93
    """
    model = Mixer(embedding_dims=[64] * 4, patch_sizes=[16, 8, 4, 2], expansion=1.5, **kwargs)
    return model

def Mixer_pico(**kwargs):
    """
    Total params: 9,283,037
    Trainable params: 9,283,037
    Non-trainable params: 0
    Total mult-adds (G): 53.30
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 9194.44
    Params size (MB): 37.12
    Estimated Total Size (MB): 9234.18
    """
    model = Mixer(embedding_dims=[128] * 7, patch_sizes=[16, 8, 4, 2, 16, 8, 4], expansion=2.0, **kwargs)
    return model

def Mixer_nano(**kwargs):
    """
    Total params: 15,909,809
    Trainable params: 15,909,809
    Non-trainable params: 0
    Total mult-adds (G): 53.41
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 10201.07
    Params size (MB): 63.62
    Estimated Total Size (MB): 10267.31
    """
    model = Mixer(embedding_dims=[128] * 4 * 2, patch_sizes=[16, 8, 4, 2] * 2, expansion=2.0, **kwargs)
    return model

def Mixer_tiny(**kwargs):
    """
    Total params: 25,715,447
    Trainable params: 25,715,447
    Non-trainable params: 0
    Total mult-adds (G): 53.57
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 14770.04
    Params size (MB): 102.84
    Estimated Total Size (MB): 14875.50
    """
    model = Mixer(embedding_dims=[128] * 4 * 3, patch_sizes=[16, 8, 4, 2] * 3, expansion=2.2, **kwargs)
    return model

def Mixer_base(**kwargs):
    """
    Total params: 61,152,801
    Trainable params: 61,152,801
    Non-trainable params: 0
    Total mult-adds (G): 54.13
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 26307.20
    Params size (MB): 244.58
    Estimated Total Size (MB): 26554.40
    """    
    model = Mixer(embedding_dims=[128] * 4 * 4, patch_sizes=[16, 8, 4, 2] * 4, expansion=4, **kwargs)
    return model

def Mixer_large(**kwargs):
    """
    Total params: 121,489,729
    Trainable params: 121,489,729
    Non-trainable params: 0
    Total mult-adds (G): 55.10
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 50466.39
    Params size (MB): 485.89
    Estimated Total Size (MB): 50954.90
    """
    model = Mixer(embedding_dims=[128] * 4 * 8, patch_sizes=[16, 8, 4, 2] * 8, expansion=4, **kwargs)
    return model

def Mixer_huge(**kwargs):
    """ 
    Total params: 453,284,609
    Trainable params: 453,284,609
    Non-trainable params: 0
    Total mult-adds (G): 60.41
    ==========================================================================================
    Input size (MB): 2.62
    Forward/backward pass size (MB): 171262.35
    Params size (MB): 1812.94
    Estimated Total Size (MB): 173077.91
    """
    model = Mixer(embedding_dims=[128] * 4 * 24, patch_sizes=[16, 8, 4, 2] * 24, expansion=5, **kwargs)
    return model


if __name__ == "__main__":
    from torchinfo import summary

    BATCH_SIZE = 16
    CHANNELS = 10
    HEIGHT = 64
    WIDTH = 64

    torch.set_default_device("cuda")

    model = Mixer_nano(chw=(10, 64, 64), output_dim=1)

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )
