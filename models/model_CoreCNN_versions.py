from models.model_CoreCNN import CoreUnet, CoreEncoder


def CoreUnet_atto(**kwargs):
    """
    Total params: 3,953,065
    Trainable params: 3,953,065
    Non-trainable params: 0
    Total mult-adds (G): 11.63
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 1601.07
    Params size (MB): 15.81
    Estimated Total Size (MB): 1622.13
    """
    model = CoreUnet(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def CoreUnet_femto(**kwargs):
    """
    Total params: 5,684,477
    Trainable params: 5,684,477
    Non-trainable params: 0
    Total mult-adds (G): 16.64
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 1920.80
    Params size (MB): 22.74
    Estimated Total Size (MB): 1948.78
    """
    model = CoreUnet(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def CoreUnet_pico(**kwargs):
    """
    Total params: 10,087,589
    Trainable params: 10,087,589
    Non-trainable params: 0
    Total mult-adds (G): 29.36
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 2560.25
    Params size (MB): 40.35
    Estimated Total Size (MB): 2605.84
    """
    model = CoreUnet(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def CoreUnet_nano(**kwargs):
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
    model = CoreUnet(depths=[2, 2, 8, 2], dims=[80*2, 160*2, 320*2, 640*2], **kwargs)
    return model

def CoreUnet_tiny(**kwargs):
    """
    Total params: 26,114,741
    Trainable params: 26,114,741
    Non-trainable params: 0
    Total mult-adds (G): 90.77
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 5066.45
    Params size (MB): 104.46
    Estimated Total Size (MB): 5176.15
    """
    model = CoreUnet(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def CoreUnet_base(**kwargs):
    """
    Total params: 61,429,957
    Trainable params: 61,429,957
    Non-trainable params: 0
    Total mult-adds (G): 282.26
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 9474.24
    Params size (MB): 245.72
    Estimated Total Size (MB): 9725.20
    """
    model = CoreUnet(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def CoreUnet_large(**kwargs):
    """
    Total params: 138,010,277
    Trainable params: 138,010,277
    Non-trainable params: 0
    Total mult-adds (G): 633.25
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 14210.13
    Params size (MB): 552.04
    Estimated Total Size (MB): 14767.42
    """
    model = CoreUnet(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def CoreUnet_huge(**kwargs):
    """
    Total params: 463,234,997
    Trainable params: 463,234,997
    Non-trainable params: 0
    Total mult-adds (T): 2.12
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 26049.88
    Params size (MB): 1852.94
    Estimated Total Size (MB): 27908.06
    """
    model = CoreUnet(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model

def Core_atto(**kwargs):
    """
    Total params: 3,841,441
    Trainable params: 3,841,441
    Non-trainable params: 0
    Total mult-adds (G): 27.52
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 1604.83
    Params size (MB): 15.37
    Estimated Total Size (MB): 1625.44
    """
    model = CoreEncoder(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def Core_femto(**kwargs):
    """
    Total params: 5,523,169
    Trainable params: 5,523,169
    Non-trainable params: 0
    Total mult-adds (G): 39.52
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 1925.80
    Params size (MB): 22.09
    Estimated Total Size (MB): 1953.14
    """
    model = CoreEncoder(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def Core_pico(**kwargs):
    """
    Total params: 9,799,681
    Trainable params: 9,799,681
    Non-trainable params: 0
    Total mult-adds (G): 70.01
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 2567.73
    Params size (MB): 39.20
    Estimated Total Size (MB): 2612.17
    """
    model = CoreEncoder(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def Core_nano(**kwargs):
    """
    Total params: 17,379,041
    Trainable params: 17,379,041
    Non-trainable params: 0
    Total mult-adds (G): 125.99
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 3461.50
    Params size (MB): 69.52
    Estimated Total Size (MB): 3536.26
    """
    model = CoreEncoder(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model

def Core_tiny(**kwargs):
    """
    Total params: 32,963,137
    Trainable params: 32,963,137
    Non-trainable params: 0
    Total mult-adds (G): 229.80
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 5286.65
    Params size (MB): 131.85
    Estimated Total Size (MB): 5423.75
    """
    model = CoreEncoder(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model

def Core_base(**kwargs):
    """
    Total params: 106,486,529
    Trainable params: 106,486,529
    Non-trainable params: 0
    Total mult-adds (G): 795.34
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 10675.25
    Params size (MB): 425.95
    Estimated Total Size (MB): 11106.44
    """
    model = CoreEncoder(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model

def Core_large(**kwargs):
    """
    Total params: 239,343,745
    Trainable params: 239,343,745
    Non-trainable params: 0
    Total mult-adds (T): 1.79
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 16012.88
    Params size (MB): 957.37
    Estimated Total Size (MB): 16975.50
    """
    model = CoreEncoder(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    return model

def Core_huge(**kwargs):
    """
    Total params: 803,694,145
    Trainable params: 803,694,145
    Non-trainable params: 0
    Total mult-adds (T): 6.00
    =========================================================================================================
    Input size (MB): 5.24
    Forward/backward pass size (MB): 29356.95
    Params size (MB): 3214.78
    Estimated Total Size (MB): 32576.97
    """
    model = CoreEncoder(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model


if __name__ == '__main__':
    from torchinfo import summary

    BATCH_SIZE = 4
    CHANNELS = 10
    HEIGHT = 96
    WIDTH = 96

    model = CoreUnet_nano(
        input_dim=10,
        output_dim=1,
    )

    summary(
        model,
        input_size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH),
    )