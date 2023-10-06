import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import buteo as beo


class TiledMSE(nn.Module):
    """
    Calculates the MSE at full image level and at the pixel level and weights the two.
    result = (sum_mse * (1 - bias)) + (mse * bias)
    """
    def __init__(self, bias=0.8, scale_term=1.0):
        super(TiledMSE, self).__init__()
        self.bias = bias
        self.scale_term = scale_term

    def forward(self, y_pred, y_true):
        y_true = (y_true + 1) * self.scale_term
        y_pred = (y_pred + 1) * self.scale_term

        y_pred_sum = torch.sum(y_pred, dim=(2, 3)) / (y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])
        y_true_sum = torch.sum(y_true, dim=(2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])

        sum_mse = torch.mean((y_pred_sum - y_true_sum) ** 2, dim=1).mean()
        mse = torch.mean((y_pred - y_true) ** 2, dim=1).mean()

        weighted = (sum_mse * (1 - self.bias)) + (mse * self.bias)
        
        return weighted 


class TiledMAPE(nn.Module):
    """
    Calculates the MSE at full image level and at the pixel level and weights the two.
    result = (sum_mse * (1 - bias)) + (mse * bias)
    """
    def __init__(self, beta=0.1, bias=0.8):
        super(TiledMAPE, self).__init__()
        self.beta = beta
        self.bias = bias
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        y_pred_sum = torch.sum(y_pred, dim=(2, 3)) / (y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])
        y_true_sum = torch.sum(y_true, dim=(2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])

        mape_sum = torch.mean(torch.abs((y_true_sum - y_pred_sum) / (y_true_sum + self.eps + self.beta)), dim=1).mean()
        mape = torch.mean(torch.abs((y_true - y_pred) / (y_true + self.eps + self.beta)), dim=1).mean()

        weighted = (mape_sum * (1 - self.bias)) + (mape * self.bias)
        
        return weighted 


class TiledMAPE2(nn.Module):
    """
    Calculates the MSE at full image level and at the pixel level and weights the two.
    result = (sum_mse * (1 - bias)) + (mse * bias)
    """
    def __init__(self, beta=0.1, bias=0.8):
        super(TiledMAPE2, self).__init__()
        self.beta = beta
        self.bias = bias
        self.eps = 1e-6

    def forward(self, y_pred, y_true):
        eps = torch.Tensor([self.eps]).to(y_pred.device)
        y_true_sum = torch.sum(y_true, dim=(2, 3)) / (y_true.shape[1] * y_true.shape[2] * y_true.shape[3])

        abs_diff = torch.abs(y_true - y_pred)
        abs_diff_sum = torch.sum(abs_diff, dim=(2, 3)) / (y_pred.shape[1] * y_pred.shape[2] * y_pred.shape[3])

        wape = torch.mean(abs_diff_sum / torch.maximum(y_true_sum + self.beta, eps), dim=1).mean()
        mape = torch.mean(abs_diff / torch.maximum(y_true + self.beta, eps), dim=1).mean()

        weighted = (wape * (1 - self.bias)) + (mape * self.bias)
        
        return weighted 


def drop_path(x, keep_prob = 1.0, inplace = False):
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1) 
    mask = x.new_empty(mask_shape).bernoulli_(keep_prob)
    mask.div_(keep_prob)

    if inplace:
        x.mul_(mask)
    else:
        x = x * mask

    return x


class DropPath(nn.Module):
    def __init__(self, p = 0.5, inplace = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x):
        if self.training and self.p > 0:
            x = drop_path(x, self.p, self.inplace)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p})"


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 

        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]

            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer """
    def __init__(self, dim, channel_first=False):
        super().__init__()
        self.channel_first = channel_first
        if self.channel_first:
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        else:
            self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        if self.channel_first:
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        else:
            Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)

        return self.gamma * (x * Nx) + self.beta + x


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    # print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs

    return schedule


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


class SE_BlockV2(nn.Module):
    # The is a custom implementation of the ideas presented in the paper:
    # https://www.sciencedirect.com/science/article/abs/pii/S0031320321003460
    def __init__(self, channels, reduction=16, activation="relu"):
        super(SE_BlockV2, self).__init__()

        self.channels = channels
        self.reduction = reduction
        self.activation = get_activation(activation)
   
        self.fc_spatial = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(channels, channels, kernel_size=2, stride=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.fc_reduction = nn.Linear(in_features=channels * (4 * 4), out_features=channels // self.reduction)
        self.fc_extention = nn.Linear(in_features=channels // self.reduction , out_features=channels)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        identity = x
        x = self.fc_spatial(identity)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_reduction(x)
        x = self.activation(x)
        x = self.fc_extention(x)
        x = self.sigmoid(x)
        x = x.reshape(x.size(0), x.size(1), 1, 1)

        return x


class SE_BlockV3(nn.Module):
    """ Squeeze and Excitation block with spatial and channel attention. """
    def __init__(self, channels, reduction_c=2, reduction_s=8, activation="relu", norm="batch", first_layer=False):
        super(SE_BlockV3, self).__init__()

        self.channels = channels
        self.first_layer = first_layer
        self.reduction_c = reduction_c if not first_layer else 1
        self.reduction_s = reduction_s
        self.activation = get_activation(activation)
   
        self.fc_pool = nn.AdaptiveAvgPool2d(reduction_s)
        self.fc_conv = nn.Conv2d(self.channels, self.channels, kernel_size=2, stride=2, groups=self.channels, bias=False)
        self.fc_norm = get_normalization(norm, self.channels)

        self.linear1 = nn.Linear(in_features=self.channels * (reduction_s // 2 * reduction_s // 2), out_features=self.channels // self.reduction_c)
        self.linear2 = nn.Linear(in_features=self.channels // self.reduction_c, out_features=self.channels)

        self.activation_output = nn.Softmax(dim=1) if first_layer else nn.Sigmoid()


    def forward(self, x):
        identity = x

        x = self.fc_pool(x)
        x = self.fc_conv(x)
        x = self.fc_norm(x)
        x = self.activation(x)
        x = x.reshape(x.size(0), -1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        if self.first_layer:
            x = self.activation_output(x) * x.size(1)
        else:
            x = self.activation_output(x)
            
        x = identity * x.reshape(x.size(0), x.size(1), 1, 1)

        return x


def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU6(inplace=False)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name

    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name

    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name

    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name

    elif activation_name == "selu":
        return nn.SELU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name

    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name

    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name

    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")


def get_normalization(normalization_name, num_channels, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        # return LayerNorm(num_channels)
        return nn.LayerNorm(num_channels)
    elif normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}")


def convert_torch_to_float(tensor):
    if torch.is_tensor(tensor):
        return float(tensor.detach().cpu().numpy().astype(np.float32))
    elif isinstance(tensor, np.ndarray) and tensor.size == 1:
        return float(tensor.astype(np.float32))
    elif isinstance(tensor, float):
        return tensor
    elif isinstance(tensor, int):
        return float(tensor)
    else:
        raise ValueError("Cannot convert tensor to float")


import yaml
class AttrDict(dict):

    def __init__(self, *args, **kwargs):

        super(AttrDict, self).__init__(*args, **kwargs)

        self.__dict__ = self


def read_yaml(path):

    f = open(path)

    params = yaml.load(f, Loader=yaml.Loader)

    return AttrDict(params)

from typing import Union, List, Tuple, Optional



class MultiArray_1D(beo.MultiArray):
    def __init__(self,
        array_list: List[Union[np.ndarray, np.memmap]],
        shuffle: bool = False,
        random_sampling: bool = False,
        seed: int = 42,
        _idx_start: Optional[int] = None,
        _idx_end: Optional[int] = None,
        _is_subarray: bool = False
    ):
        self.array_list = array_list
        self.is_subarray = _is_subarray
        
        assert isinstance(self.array_list, list), "Input should be a list of numpy arrays."
        assert len(self.array_list) > 0, "Input list is empty. Please provide a list with numpy arrays."
        assert all(isinstance(item, (np.ndarray, np.memmap)) for item in self.array_list), "Input list should only contain numpy arrays."

        self.cumulative_sizes = [i for i in range(len(self.array_list))]

        self._idx_start = int(_idx_start) if _idx_start is not None else 0
        self._idx_end = int(_idx_end) if _idx_end is not None else int(self.cumulative_sizes[-1])

        assert isinstance(self._idx_start, int), "Minimum length should be an integer."
        assert isinstance(self._idx_end, int), "Maximum length should be an integer."
        assert self._idx_start < self._idx_end, "Minimum length should be smaller than maximum length."

        self.total_length = len(array_list)-1 #int(min(self.cumulative_sizes[-1], self._idx_end - self._idx_start))  # Store length for faster access

        if shuffle and random_sampling:
            raise ValueError("Cannot use both shuffling and resevoir sampling at the same time.")

        # Shuffling
        self.seed = seed
        self.shuffle = shuffle
        self.shuffle_indices = None
        self.random_sampling = random_sampling
        self.rng = np.random.default_rng(seed)

        if self.shuffle:
            self.shuffle_indices = self.rng.permutation(range(self._idx_start, self._idx_end))
    
    
    def _load_item(self, idx: int):
        """ Load an item from the array list. """
        # array_idx = np.searchsorted(self.cumulative_sizes, idx, side='right') - 1

        # calculated_idx = idx - self.cumulative_sizes[array_idx]
        # if calculated_idx < 0 or calculated_idx >= self.array_list[array_idx].shape[0]:
        #     raise IndexError(f'Index {idx} out of bounds for MultiArray with length {self.__len__()}')

        output = self.array_list[idx]# [calculated_idx]

        return output