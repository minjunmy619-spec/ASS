import inspect
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel
try:
    from ..layers import activations, normalizations
    from ..utils.signal_processing import calculate_band_widths
except ImportError:
    class _LayerNormalization4D(nn.Module):
        def __init__(self, normalized_shape, eps=1e-8):
            super().__init__()
            channels = normalized_shape[0] if isinstance(normalized_shape, tuple) else normalized_shape
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

        def forward(self, x):
            mean = x.mean(dim=1, keepdim=True)
            centered = x - mean
            var = (centered * centered).mean(dim=1, keepdim=True)
            inv_std = torch.rsqrt(var + self.eps)
            return centered * inv_std * self.weight + self.bias

    class _Registry:
        def __init__(self, mapping):
            self.mapping = mapping

        def get(self, name):
            if name is None:
                return nn.Identity
            if name not in self.mapping:
                raise KeyError(f"Unsupported registry key: {name}")
            return self.mapping[name]

    activations = _Registry(
        {
            "prelu": nn.PReLU,
            "relu": nn.ReLU,
            "identity": nn.Identity,
        }
    )
    normalizations = _Registry(
        {
            "LayerNormalization4D": _LayerNormalization4D,
            "identity": nn.Identity,
        }
    )

    def calculate_band_widths(enc_dim, sample_rate):
        bandwidth_25 = int(np.floor(25 / (sample_rate / 2.0) * enc_dim))
        bandwidth_100 = int(np.floor(100 / (sample_rate / 2.0) * enc_dim))
        bandwidth_250 = int(np.floor(250 / (sample_rate / 2.0) * enc_dim))
        bandwidth_500 = int(np.floor(500 / (sample_rate / 2.0) * enc_dim))
        band_width = [bandwidth_25] * 40
        band_width += [bandwidth_100] * 10
        band_width += [bandwidth_250] * 8
        band_width += [bandwidth_500] * 8
        band_width.append(enc_dim - np.sum(band_width))
        return band_width


def GlobLN(nOut):
    # return nn.GroupNorm(1, nOut, eps=1e-8)
    return nn.LayerNorm(1, eps=1e-8) #normalize channel dim


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)

class Conv2dNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn, nOut, (1, kSize), stride=(1, stride), padding=(0, padding), bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        input = input.unsqueeze(2)
        output = self.conv(input)
        output = self.norm(output)
        output = self.act(output)
        output = output.squeeze(2)
        return output

class LayerNorm2DOnChannel(nn.Module):
    def __init__(self, nOut, eps: float = 1e-8):
        super(LayerNorm2DOnChannel, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, nOut, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, nOut, 1, 1))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        return centered * inv_std * self.weight + self.bias


class BasicLayerNormLastDim(nn.Module):
    def __init__(self, normalized_size: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_size))
        self.bias = nn.Parameter(torch.zeros(normalized_size))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        var = (centered * centered).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(var + self.eps)
        shape = [1] * x.dim()
        shape[-1] = self.weight.numel()
        return centered * inv_std * self.weight.view(shape) + self.bias.view(shape)

class PrjConv2dNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        
        self.norm = LayerNorm2DOnChannel(nOut, eps=1e-8)
        self.act = nn.PReLU()

    def forward(self, input):   
        output = self.conv(input)
        output = self.norm(output)
        output = self.act(output) 
        return output
    
class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class Conv2dNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn, nOut, (1, kSize), stride=(1, stride), padding=(0, padding), bias=bias, groups=groups
        )
        self.norm = LayerNorm2DOnChannel(nOut) #GlobLN(nOut)

    def forward(self, input):
        # input = input.unsqueeze(2)
        output = self.conv(input)
        output = self.norm(output)
        # output = output.squeeze(2)
        return output

class FreqConv2dNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn, nOut, (kSize, 1), stride=(stride, 1), padding=(padding, 0), bias=bias, groups=groups
        )
        self.norm = LayerNorm2DOnChannel(nOut) #GlobLN(nOut)

    def forward(self, input):
        # input = input.unsqueeze(2)
        output = self.conv(input)
        output = self.norm(output)
        # output = output.squeeze(2)
        return output


def compute_freq_level_sizes(nband: int, depth: int):
    sizes = [nband]
    for _ in range(1, depth):
        sizes.append((sizes[-1] + 1) // 2)
    return sizes


class StaticFreqResize2D(nn.Module):
    """
    Static frequency-axis resize for [B, C, F, T].
    - downsample: fixed depthwise conv with stride on the frequency axis
    - upsample: repeat_interleave + constant slicing on the frequency axis
    """

    def __init__(self, channels: int, source_bins: int, target_bins: int):
        super().__init__()
        self.channels = channels
        self.source_bins = source_bins
        self.target_bins = target_bins

        if source_bins == target_bins:
            self.mode = "identity"
        elif source_bins > target_bins:
            self.mode = "downsample"
            current_bins = source_bins
            self.downsample_steps = 0
            while current_bins > target_bins:
                current_bins = (current_bins + 1) // 2
                self.downsample_steps += 1
            if current_bins != target_bins:
                raise ValueError(
                    f"StaticFreqResize2D only supports ceil-halving downsample chains, got {source_bins} -> {target_bins}"
                )
            weight = torch.full((channels, 1, 3, 1), 1.0 / 3.0)
            self.register_buffer("step_weight", weight)
        else:
            self.mode = "upsample"
            current_bins = source_bins
            self.upsample_steps = 0
            while current_bins < target_bins:
                current_bins *= 2
                self.upsample_steps += 1
            if current_bins < target_bins:
                raise ValueError(
                    f"StaticFreqResize2D could not build an upsample chain for {source_bins} -> {target_bins}"
                )

    def forward(self, x):
        if self.mode == "identity":
            return x
        if self.mode == "downsample":
            for _ in range(self.downsample_steps):
                x = F.conv2d(
                    x,
                    self.step_weight,
                    bias=None,
                    stride=(2, 1),
                    padding=(1, 0),
                    groups=self.channels,
                )
            return x
        for _ in range(self.upsample_steps):
            x = x.repeat_interleave(2, dim=2)
        return x[:, :, :self.target_bins, :]

class FrameConv2dNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn, nOut, (1, kSize), stride=(1, stride), padding=(0, padding), bias=bias, groups=groups
        )
        self.norm = LayerNorm2DOnChannel(nOut) #GlobLN(nOut)

    def forward(self, input):
        # input = input.unsqueeze(2)
        output = self.conv(input)
        output = self.norm(output)
        # output = output.squeeze(2)
        return output


class CausalFrameConv2dNorm(nn.Module):
    """
    Time-causal 2D conv on [B, C, F, T].
    Only pads on the left of the time dimension.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.lookback = (kSize - 1) * dilation
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (1, kSize),
            stride=(1, stride),
            padding=(0, 0),
            dilation=(1, dilation),
            bias=bias,
            groups=groups,
        )
        self.norm = LayerNorm2DOnChannel(nOut)

    def forward(self, input):
        if self.lookback > 0:
            input = F.pad(input, (self.lookback, 0, 0, 0))
        output = self.conv(input)
        return self.norm(output)

    
class ATTConvActNorm(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ATTConvActNorm, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        # if self.padding is None:
        #     self.padding = 0 if self.stride >= 1 else "same" #FIXME cmj self.padding = 0 if self.stride >= 1 else "same"
        self.padding = 0

        
        conv = nn.Conv2d if is2d else nn.Conv1d

        self.conv = conv(
            in_channels=self.in_chan,
            out_channels=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        # if self.xavier_init:
        #     nn.init.xavier_uniform_(self.conv.weight)
        

        self.act = activations.get(self.act_type)()
        if is2d and self.norm_type == "LayerNormalization4D":
            self.norm = LayerNorm2DOnChannel(self.out_chan)
        else:
            self.norm = normalizations.get(self.norm_type)(
                (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
            )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

class ATTConvActNormOnFrame(nn.Module):
    def __init__(
        self,
        in_chan: int = 1,
        out_chan: int = 1,
        kernel_size: int = -1,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        padding: int = None,
        norm_type: str = None,
        act_type: str = None,
        n_freqs: int = -1,
        xavier_init: bool = False,
        bias: bool = True,
        is2d: bool = False,
        *args,
        **kwargs,
    ):
        super(ATTConvActNormOnFrame, self).__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.padding = padding
        self.norm_type = norm_type
        self.act_type = act_type
        self.n_freqs = n_freqs
        self.xavier_init = xavier_init
        self.bias = bias

        # if self.padding is None:
        #     self.padding = 0 if self.stride >= 1 else "same" #FIXME cmj self.padding = 0 if self.stride >= 1 else "same"
        self.padding = 0

        
        # conv = nn.Conv2d if is2d else nn.Conv1d
        conv = Unified2DAttention if is2d else nn.Conv1d        

        self.conv = conv(
            in_channels=self.in_chan,
            out_channels=self.out_chan,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias,
        )

        # if self.xavier_init:
        #     nn.init.xavier_uniform_(self.conv.weight)
        

        self.act = activations.get(self.act_type)()
        if is2d and self.norm_type == "LayerNormalization4D":
            self.norm = LayerNorm2DOnChannel(self.out_chan)
        else:
            self.norm = normalizations.get(self.norm_type)(
                (self.out_chan, self.n_freqs) if self.norm_type == "LayerNormalization4D" else self.out_chan
            )

    def forward(self, x: torch.Tensor):
        output = self.conv(x)
        output = self.act(output)
        output = self.norm(output)
        return output

    def get_config(self):
        encoder_args = {}

        for k, v in (self.__dict__).items():
            if not k.startswith("_") and k != "training":
                if not inspect.ismethod(v):
                    encoder_args[k] = v

        return encoder_args

class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class DilatedConv2dNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=(1,1), d=(1,1), groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=(((kSize[0] - 1) // 2) * d[0], ((kSize[1] - 1) // 2) * d[1]),
            groups=groups,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        # self.norm = GlobLN(nOut)
        self.norm = LayerNorm2DOnChannel(nOut, eps=1e-8) #nn.LayerNorm(1, eps=1e-8) #normalize channel dim

    def forward(self, input):        
        output = self.conv(input)
        output = self.norm(output)        
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = Conv2dNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_size, hidden_size, (1, 5), 1, (0, 2), bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = Conv2dNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(2)
        x = self.dwconv(x)
        x = x.squeeze(2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FrequencyMlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = Conv2dNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_size, hidden_size, (5, 1), 1, (2, 0), bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = Conv2dNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)       
        x = self.dwconv(x)        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FrameMlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1):
        super().__init__()
        self.fc1 = Conv2dNorm(in_features, hidden_size, 1, bias=False)
        self.dwconv = nn.Conv2d(
            hidden_size, hidden_size, (1, 5), 1, (0, 2), bias=True, groups=hidden_size
        )
        self.act = nn.ReLU()
        self.fc2 = Conv2dNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)       
        x = self.dwconv(x)        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CausalFrameMlp(nn.Module):
    def __init__(self, in_features, hidden_size, drop=0.1, kernel_size=5, dilation=1):
        super().__init__()
        self.fc1 = Conv2dNorm(in_features, hidden_size, 1, bias=False)
        self.lookback = (kernel_size - 1) * dilation
        self.dwconv = nn.Conv2d(
            hidden_size,
            hidden_size,
            (1, kernel_size),
            1,
            (0, 0),
            dilation=(1, dilation),
            bias=True,
            groups=hidden_size,
        )
        self.act = nn.ReLU()
        self.fc2 = Conv2dNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        if self.lookback > 0:
            x = F.pad(x, (self.lookback, 0, 0, 0))
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StatefulCausalFrameMlp(nn.Module):
    """
    Causal frame MLP with explicit state on the depthwise temporal conv.
    This keeps single-frame runtime efficient while allowing exact chunk
    training/inference equivalence.
    """

    def __init__(self, in_features, hidden_size, drop=0.1, kernel_size=5, dilation=1):
        super().__init__()
        self.fc1 = Conv2dNorm(in_features, hidden_size, 1, bias=False)
        self.lookback = (kernel_size - 1) * dilation
        self.dwconv = nn.Conv2d(
            hidden_size,
            hidden_size,
            (1, kernel_size),
            1,
            (0, 0),
            dilation=(1, dilation),
            bias=True,
            groups=hidden_size,
        )
        self.act = nn.ReLU()
        self.fc2 = Conv2dNorm(hidden_size, in_features, 1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, state=None):
        x = self.fc1(x)
        if self.lookback > 0:
            if state is None:
                state = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], self.lookback)
            combined = torch.cat([state, x], dim=-1)
            next_state = combined[:, :, :, -self.lookback:].detach()
        else:
            combined = x
            next_state = x[:, :, :, :0].detach()

        x = self.dwconv(combined)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x, next_state
       
class InjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = Conv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = Conv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = Conv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        global_act = self.act(global_act)
        global_act = global_act.unsqueeze(2)
        sig_act = F.interpolate(global_act, size=(1, T), mode="nearest")
        sig_act = sig_act.squeeze(2)
        # sig_act = self.act(global_act)

        global_feat = self.global_embedding(x_g)
        global_feat = global_feat.unsqueeze(2)
        global_feat = F.interpolate(global_feat, size=(1, T), mode="nearest")
        global_feat = global_feat.squeeze(2)

        out = local_feat * sig_act + global_feat
        return out

class FrequencyInjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, source_bins: int = None, target_bins: int = None) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = FreqConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = FreqConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = FreqConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()
        self.global_resizer = None
        if source_bins is not None and target_bins is not None:
            self.global_resizer = StaticFreqResize2D(oup, source_bins, target_bins)

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        local_feat = self.local_embedding(x_l)       

        global_act = self.global_act(x_g)
        global_act = self.act(global_act)
        if self.global_resizer is not None:
            sig_act = self.global_resizer(global_act)
        else:
            B, N, nBand, T = x_l.shape
            sig_act = AdaptiveAvgPool2DOnLastDim()(global_act, (nBand, T))

        global_feat = self.global_embedding(x_g)
        if self.global_resizer is not None:
            global_feat = self.global_resizer(global_feat)
        else:
            B, N, nBand, T = x_l.shape
            global_feat = AdaptiveAvgPool2DOnLastDim()(global_feat, (nBand, T))
        
        out = local_feat * sig_act + global_feat
        return out

class FrameInjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = FrameConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = FrameConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = FrameConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, nBand, T = list(x_l.size())
        local_feat = self.local_embedding(x_l)        

        global_act = self.global_act(x_g)
        global_act = self.act(global_act)

        # sig_act = F.interpolate(global_act, size=(nBand, T), mode="nearest")
        # sig_act = AdaptiveAvgPool2DOnLastDim()(global_act, (nBand, T)) #FIXME already in the same size
        sig_act = global_act

        global_feat = self.global_embedding(x_g)
        # print(f"Frame local_feat: {local_feat.shape} vs global_act:{global_act.shape} vs global_feat: {global_feat.shape}, interploation with: ({nBand},{T})")

        # global_feat = F.interpolate(global_feat, size=(nBand, T), mode="nearest")
        # global_feat = AdaptiveAvgPool2DOnLastDim()(global_feat, (nBand, T)) #FIXME already in the same size
    
        out = local_feat * sig_act + global_feat
        return out


class CausalFrameInjectionMultiSum(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1, dilation: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.receptive_field = (kernel - 1) * dilation
        self.local_embedding = CausalFrameConv2dNorm(inp, oup, kernel, groups=groups, dilation=dilation, bias=False)
        self.global_embedding = CausalFrameConv2dNorm(inp, oup, kernel, groups=groups, dilation=dilation, bias=False)
        self.global_act = CausalFrameConv2dNorm(inp, oup, kernel, groups=groups, dilation=dilation, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        local_feat = self.local_embedding(x_l)
        global_act = self.act(self.global_act(x_g))
        global_feat = self.global_embedding(x_g)
        return local_feat * global_act + global_feat
        
class InjectionMulti(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")
        # sig_act = self.act(global_act)

        out = local_feat * sig_act
        return out

class UConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=5,
                    stride=2,
                    groups=in_channels,
                    d=1,
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(InjectionMultiSum(in_channels, in_channels))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.globalatt = Mlp(in_channels, in_channels, drop=0.1)
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(InjectionMultiSum(in_channels, in_channels, 5))

    def forward(self, x, ):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )

        # print(f"output[-1].shape[-1]: {output[-1].shape[-1]}")

        for fea in output:            
            global_f = global_f + F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
            # global_f = global_f + fea
        global_f = self.globalatt(global_f)  # [B, N, T]

        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        return self.res_conv(expanded) + residual

class AdaptiveAvgPool1DCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool1DCustom, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        shape_x = x.shape
        if(shape_x[-1] < self.output_size):
            paddzero = torch.zeros((shape_x[0], shape_x[1], self.output_size - shape_x[-1]))
            paddzero = paddzero.to(x)
            x = torch.cat((x, paddzero), axis=-1)

        # stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        # kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        # avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))

        stride_size = math.floor(x.shape[-1] / self.output_size)
        kernel_size = math.floor(x.shape[-1] - (self.output_size - 1) * stride_size)
        # print(f"AdaptiveAvgPool1dCustom x shape: {x.shape}, stride: {stride_size}@{type(stride_size)}, kernel: {kernel_size}@{type(kernel_size)}, output_size: {self.output_size}")

        avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride_size)
        x = avg(x)
        return x

class AdaptiveAvgPool2DCustom(nn.Module):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2DCustom, self).__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        shape_x = x.shape
        if(shape_x[-1] < self.output_size):
            paddzero = torch.zeros((shape_x[0], shape_x[1], self.output_size - shape_x[-1]))
            paddzero = paddzero.to(x)
            x = torch.cat((x, paddzero), axis=-1)

        # stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        # kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size
        # avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))

        stride_size = math.floor(x.shape[-1] / self.output_size)
        kernel_size = math.floor(x.shape[-1] - (self.output_size - 1) * stride_size)
        # print(f"AdaptiveAvgPool1dCustom x shape: {x.shape}, stride: {stride_size}@{type(stride_size)}, kernel: {kernel_size}@{type(kernel_size)}, output_size: {self.output_size}")

        avg = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride_size))
        x = x.unsqueeze(2)
        x = avg(x)
        x = x.squeeze(2)
        return x

class AdaptiveAvgPool2DOnLastDim(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPool2DOnLastDim, self).__init__()
        # self.output_size = np.array(output_size)
        
    def forward(self, x: torch.Tensor, output_size):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, 1, output_size)
        '''
        if torch.onnx.is_in_onnx_export():
            return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
   
        # Convert output_size to tensor for calculations
        output_size_tensor = torch.tensor(output_size, device=x.device)
        
        # Pad if necessary using regular Python control flow
        if x.shape[-1] < output_size[-1]:
            shape_x = x.shape
            pad_size = output_size[-1] - shape_x[-1]
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], pad_size), device=x.device)
            x = F.interpolate(x, output_size, mode="bilinear")
            # x = x.repeat(1, 1, 1, output_size[-1])
            # xx = torch.cat((x, paddzero), axis=-1)
            # x = torch.cat((x, x.clone()), axis=-1)
            # print(f"AdaptiveAvgPool2DOnLastDim H padding :{pad_size}=> x {x.shape} => xx: {xx.shape} => xy: {xy.shape}  => yy: {yy.shape}")

            # x = F.pad(x, (0, pad_size, 0, 0), mode='constant', value=x.mean().item())
            # x = F.pad(x, (0, pad_size, 0, 0), mode='constant', value=0.0)

        ####padd on height
        if x.shape[-2] < output_size[-2]:
            shape_x = x.shape
            pad_size = output_size[-2] - shape_x[-2]
            paddzero = torch.zeros((shape_x[0], shape_x[1], pad_size, shape_x[3]), device=x.device)
            x = F.interpolate(x, output_size, mode="bilinear")
            # x = x.repeat(1, 1, output_size[-2], 1)
            # xx = torch.cat((x, paddzero), axis=-2)
            # xy = torch.cat((x, x.clone()), axis=-2)
            # print(f"AdaptiveAvgPool2DOnLastDim H padding :{pad_size}=> x {x.shape} => xx: {xx.shape} => xy: {xy.shape}  => yy: {yy.shape}")

            # x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=x.mean().item())
            # x = F.pad(x, (0, 0, 0, pad_size), mode='constant', value=0.0)
        if x.shape[-2]==output_size[-2] and x.shape[-1]==output_size[-1]:
            return x
        
        # Calculate adaptive pooling parameters
        x_shape = torch.tensor(x.shape[-2:], device=x.device)
        stride_size = torch.floor(x_shape / output_size_tensor).to(torch.int32)
        kernel_size = x_shape - (output_size_tensor - 1) * stride_size
        
        avg = nn.AvgPool2d(kernel_size=kernel_size.tolist(), stride=stride_size.tolist())        
        x = avg(x)

        return x

class AdaptiveAvgPool2DOnLastDimNotWorking(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPool2DOnLastDim, self).__init__()
        
    def forward(self, x: torch.Tensor, output_size):
        '''
        Args:
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, output_height, output_width)
        '''
        # Get input dimensions
        b, c, h, w = x.shape
        
        # For ONNX compatibility, we need to avoid adaptive pooling
        # Instead, we'll use interpolation when upscaling and avg_pool2d when downscaling
        
        if output_size[0] > h or output_size[1] > w:
            print("F.interpolate")
            # If output size is greater than or equal to input size, use interpolation (upscaling)
            # Pad if necessary
            pad_w = max(0, output_size[1] - w)
            pad_h = max(0, output_size[0] - h)
            
            if pad_w > 0 or pad_h > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)
            
            # Use bilinear interpolation for upscaling
            x = F.interpolate(x, size=(output_size[0], output_size[1]), mode='bilinear', align_corners=False)
        else:
            print("F.avg_pool2d")
            # If output size is less than input size, use avg_pool2d (downscaling)
            # Pad if necessary
            pad_w = max(0, output_size[1] - w)
            pad_h = max(0, output_size[0] - h)
            
            if pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, 0), mode='constant', value=0.0)
            if pad_h > 0:
                x = F.pad(x, (0, 0, 0, pad_h), mode='constant', value=0.0)
            
            # Calculate kernel sizes to achieve the desired output size
            h_padded = x.shape[2]
            w_padded = x.shape[3]
            
            # Calculate stride and kernel sizes for average pooling
            stride_h = max(1, h_padded // output_size[0])
            stride_w = max(1, w_padded // output_size[1])
            
            # Calculate kernel sizes
            kernel_h = h_padded - (output_size[0] - 1) * stride_h
            kernel_w = w_padded - (output_size[1] - 1) * stride_w
            
            # Ensure kernel sizes are positive
            kernel_h = max(1, kernel_h)
            kernel_w = max(1, kernel_w)
            
            # Ensure stride is not larger than kernel
            stride_h = min(stride_h, kernel_h)
            stride_w = min(stride_w, kernel_w)
            
            # Use average pooling with calculated parameters
            x = F.avg_pool2d(x, kernel_size=(kernel_h, kernel_w), stride=(stride_h, stride_w))
            
            # If result is not exactly the target size, interpolate to match
            if x.shape[2] != output_size[0] or x.shape[3] != output_size[1]:
                x = F.interpolate(x, size=(output_size[0], output_size[1]), mode='bilinear', align_corners=False)
        
        return x
    
class FreqUConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True, nband=1):
        super().__init__()
        self.proj_1x1 = PrjConv2dNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.level_bins = compute_freq_level_sizes(nband, upsampling_depth)
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConv2dNorm(
                in_channels, in_channels, kSize=(5, 1), stride=(1,1), groups=in_channels, d=(1, 1)
            )
        )
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                DilatedConv2dNorm(
                    in_channels,
                    in_channels,
                    kSize=(5, 1),
                    stride=(2, 1),
                    groups=in_channels,
                    d=(1, 1),
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(
                FrequencyInjectionMultiSum(
                    in_channels,
                    in_channels,
                    source_bins=self.level_bins[-1],
                    target_bins=self.level_bins[i],
                )
            )

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.globalatt = FrequencyMlp(in_channels, in_channels, drop=0.1)
        self.global_resizers = nn.ModuleList([
            StaticFreqResize2D(in_channels, self.level_bins[i], self.level_bins[-1])
            for i in range(self.depth - 1)
        ])
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            if i == self.depth - 2:
                source_bins = self.level_bins[i - 1]
            else:
                source_bins = self.level_bins[i + 1]
            self.last_layer.append(
                FrequencyInjectionMultiSum(
                    in_channels,
                    in_channels,
                    5,
                    source_bins=source_bins,
                    target_bins=self.level_bins[i],
                )
            )

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        # print(f"======FreqUConvBlock")
        residual = x.clone() # B, N, nband, T
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        global_f = self.global_resizers[0](output[0])
        for idx, fea in enumerate(output[1:-1], start=1):
            global_f = global_f + self.global_resizers[idx](fea)
        global_f = global_f + output[-1]

        global_f = self.globalatt(global_f)  # (B, N, nBand, T)


        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            # print(f"loc_glo_fus:{idx}, {global_f.shape}")
            local = output[idx]            
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            # print(f"last_layer: {i}")
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1]) #FIXME i - 1 change to i +1
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        
        res_output = self.res_conv(expanded)
        # print(f"FreqUConvBlock=======")
        return res_output + residual

class TimeUConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True):
        super().__init__()
        self.proj_1x1 = PrjConv2dNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConv2dNorm(
                in_channels, in_channels, kSize=(1, 5), stride=(1,1), groups=in_channels, d=(1, 1)
            )
        )
        for i in range(1, upsampling_depth):
            self.spp_dw.append(
                DilatedConv2dNorm(
                    in_channels,
                    in_channels,
                    kSize=(1, 5),
                    stride=(1, 2),
                    groups=in_channels,
                    d=(1, 1),
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(FrameInjectionMultiSum(in_channels, in_channels)) 

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.globalatt = FrameMlp(in_channels, in_channels, drop=0.1)
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(FrameInjectionMultiSum(in_channels, in_channels, 5))

    def forward(self, x):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        residual = x.clone() # B, N, nband, T
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        # Do the downsampling process from the previous level
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        last_output=output[-1]
        last_output_size_on_frame = (last_output.shape[-2], last_output.shape[-1]) #(T, nBand) on last output
        # print(f"Frame last_output_size_on_frame: {last_output_size_on_frame}")

        fea0 = output[0]
        fea0_pooled = AdaptiveAvgPool2DOnLastDim()(fea0, last_output_size_on_frame)
        global_f= fea0_pooled
        for fea in output[1:]:
            # print(f"Frame fea shape: {fea.shape}")            
            fea_pooled = AdaptiveAvgPool2DOnLastDim()(fea, last_output_size_on_frame)
            # print(f"Frame fea_pooled shape: {fea_pooled.shape}")
            global_f = global_f + fea_pooled
        # print(f"overall global_f: {global_f.shape}")
        global_f = self.globalatt(global_f)  # [B, N, nBand, T]
        # print(f"global_f after att: {global_f.shape}")

        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]
            # print(f"Frame local shape: {local.shape}")
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        
        res_output = self.res_conv(expanded)
        # print(f"Frame res_output: {res_output.shape}")
        
        return res_output + residual
    
class Unified2DAttentionOnFrameAsym(nn.Module):
    """
    Unified2DAttentionOnFrameAsym: K, Q using same channels, but V using different channels
    """
    def __init__(self, in_channels, hid_channels=4, freq_bins=1, window_size=50, n_heads=4, value_hid_channels=None, act_type: str = "prelu",  norm_type: str = "LayerNormalization4D", need_streaming=True):
        super().__init__()
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.hid_channels=hid_channels
        self.window_size = window_size
        self.num_heads = n_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.need_streaming=need_streaming

        self.v_hid_channels = value_hid_channels if value_hid_channels is not None else hid_channels
        self.total_hidd_channels = self.num_heads * (self.hid_channels * 2 + self.v_hid_channels)
        self.head_dim = self.hid_channels * self.freq_bins
        self.v_head_dim = self.v_hid_channels * self.freq_bins
        self.attn_scale = float(self.head_dim ** -0.5)

        # Keep the projection block aligned with the official TIGER attention:
        # conv -> act -> norm. n_freqs intentionally stays 1, matching tiger.py.
        self.qkv_conv = ATTConvActNorm(
            in_chan=self.in_channels,
            out_chan=self.total_hidd_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=1,
            is2d=True,
        )
        self.proj_conv = ATTConvActNorm(
            in_chan=self.num_heads * self.v_hid_channels,
            out_chan=self.in_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=1,
            is2d=True,
        )

    def get_sliding_window_mask(self, seq_len, window_size):    
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask_too_old = torch.tril(torch.ones(seq_len, seq_len), -window_size - 1)
        final_mask = mask - mask_too_old
        return final_mask.float().masked_fill(final_mask == 0, float("-inf")).masked_fill(final_mask == 1, 0.0)

    def forward(self, x, past_kv=None, past_valid_mask=None):
        """
        x: [B, C, F, T]
        - T=1: export / runtime single-frame fast path
        - T>1: training chunk path with exact causal-with-cache semantics
        """
        B, inpC, F, T = x.shape
        if not torch.onnx.is_in_onnx_export():
            assert F == self.freq_bins, f"expected {self.freq_bins} bands, got {F}"

        x = x.transpose(-2, -1).contiguous()  # [B, C, 1, F]
        qkv = self.qkv_conv(x)
        qkv = qkv.permute(0, 2, 1, 3).reshape(B, T, self.num_heads, -1).transpose(1, 2)

        q = qkv[:, :, :, 0:self.head_dim]
        k = qkv[:, :, :, self.head_dim:2 * self.head_dim]
        v = qkv[:, :, :, 2 * self.head_dim:]

        if past_kv is None:
            past_kv = x.new_zeros(B, self.num_heads, self.window_size, self.head_dim + self.v_head_dim)
        if past_valid_mask is None:
            past_valid_mask = x.new_zeros(B, 1, self.window_size, 1)

        prev_k = past_kv[:, :, :, 0:self.head_dim]
        prev_v = past_kv[:, :, :, self.head_dim:]
        current_valid = x.new_ones(B, 1, T, 1)

        if torch.onnx.is_in_onnx_export() or T == 1:
            new_k = torch.cat([prev_k[:, :, T:, :], k], dim=2)
            new_v = torch.cat([prev_v[:, :, T:, :], v], dim=2)
            next_valid_mask = torch.cat([past_valid_mask[:, :, T:, :], current_valid], dim=2)

            head_contexts = []
            invalid_mask = 1.0 - next_valid_mask.reshape(B, 1, self.window_size)
            for head_idx in range(self.num_heads):
                q_vec = q[:, head_idx, :, :]
                k_mat = new_k[:, head_idx, :, :].transpose(1, 2).contiguous()
                v_mat = new_v[:, head_idx, :, :]

                attn_scores = torch.bmm(q_vec, k_mat) * self.attn_scale
                attn_scores = attn_scores + invalid_mask * (-1e4)
                attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
                head_contexts.append(torch.bmm(attn_weights, v_mat))
            context = torch.stack(head_contexts, dim=1)
            next_state = torch.cat([new_k, new_v], dim=3)
        else:
            total_k = torch.cat([prev_k, k], dim=2)
            total_v = torch.cat([prev_v, v], dim=2)
            total_valid_mask = torch.cat([past_valid_mask, current_valid], dim=2)

            positions = torch.arange(self.window_size + T, device=x.device).unsqueeze(0)
            query_ids = torch.arange(T, device=x.device).unsqueeze(1)
            allowed = (positions >= (query_ids + 1)) & (positions <= (query_ids + self.window_size))
            full_mask = torch.where(
                allowed,
                x.new_zeros(1),
                x.new_full((), -1e4),
            )
            head_contexts = []
            invalid_mask = (1.0 - total_valid_mask.reshape(B, 1, self.window_size + T)) * (-1e4)
            for head_idx in range(self.num_heads):
                q_vec = q[:, head_idx, :, :]
                k_mat = total_k[:, head_idx, :, :].transpose(1, 2).contiguous()
                v_mat = total_v[:, head_idx, :, :]

                attn_scores = torch.bmm(q_vec, k_mat) * self.attn_scale
                attn_scores = attn_scores + full_mask.unsqueeze(0)
                attn_scores = attn_scores + invalid_mask
                attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
                head_contexts.append(torch.bmm(attn_weights, v_mat))
            context = torch.stack(head_contexts, dim=1)

            next_k = total_k[:, :, -self.window_size:, :]
            next_v = total_v[:, :, -self.window_size:, :]
            next_valid_mask = total_valid_mask[:, :, -self.window_size:, :]
            next_state = torch.cat([next_k, next_v], dim=3)

        out = context.reshape(B, self.num_heads, T, self.v_hid_channels * self.freq_bins)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.v_hid_channels, self.freq_bins)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = self.proj_conv(out)
        return out.transpose(-2, -1).contiguous(), next_state, next_valid_mask

class Unified2DAttentionOnFreqAsym(nn.Module):
    """
    Unified2DAttentionOnFreqAsym: K, Q using same channels, but V using different channels
    """
    def __init__(self, in_channels, hid_channels=4, freq_bins=1, window_size=50, n_heads=4, value_hid_channels=None, act_type: str = "prelu",  norm_type: str = "LayerNormalization4D", need_streaming=True):
        super().__init__()
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.hid_channels=hid_channels
        self.window_size = window_size
        self.num_heads = n_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.need_streaming=need_streaming

        self.v_hid_channels = value_hid_channels if value_hid_channels is not None else hid_channels
        self.total_hidd_channels = self.num_heads * (self.hid_channels * 2 + self.v_hid_channels)
        self.head_dim = self.hid_channels
        self.v_head_dim = self.v_hid_channels
        self.attn_scale = float(self.head_dim ** -0.5)

        self.qkv_conv = ATTConvActNorm(
            in_chan=self.in_channels,
            out_chan=self.total_hidd_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=1,
            is2d=True,
        )
        self.proj_conv = ATTConvActNorm(
            in_chan=self.num_heads * self.v_hid_channels,
            out_chan=self.in_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=1,
            is2d=True,
        )
    
    def get_sliding_window_mask(self, seq_len, window_size):    
        # 1. 首先生成标准的因果掩码 (下三角)
        mask = torch.tril(torch.ones(seq_len, seq_len))
        
        # 2. 生成一个偏移的下三角矩阵来抹掉“太远”的历史
        # i - j <= window_size => j >= i - window_size
        mask_too_old = torch.tril(torch.ones(seq_len, seq_len), -window_size - 1)
        
        # 3. 最终掩码：因果区域 减去 太老的区域
        final_mask = mask - mask_too_old
        
        # 转换为常用的补丁格式: 0 表示关注, -inf 表示屏蔽
        return final_mask.float().masked_fill(final_mask == 0, float('-inf')).masked_fill(final_mask == 1, 0.0)

    def forward(self, x, past_kv=None):
        """
        x: [B, C, F, T]
        - training: T >= 1
        - inferencing: T = 1 in the exportable streaming cell
        """
        B, inpC, F, T = x.shape
        if not torch.onnx.is_in_onnx_export():
            assert F == self.freq_bins, f"expected {self.freq_bins} bands, got {F}"

        qkv = self.qkv_conv(x)
        head_chunks = torch.chunk(qkv, self.num_heads, dim=1)
        head_outputs = []

        for head_chunk in head_chunks:
            q = head_chunk[:, 0:self.head_dim, :, :]
            k = head_chunk[:, self.head_dim:2 * self.head_dim, :, :]
            v = head_chunk[:, 2 * self.head_dim:, :, :]

            time_contexts = []
            for t in range(T):
                q_t = q[:, :, :, t].transpose(1, 2).contiguous()  # [B, F, D]
                k_t = k[:, :, :, t].contiguous()  # [B, D, F]
                v_t = v[:, :, :, t].transpose(1, 2).contiguous()  # [B, F, V]

                attn_scores = torch.bmm(q_t, k_t) * self.attn_scale
                attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1)
                ctx_t = torch.bmm(attn_weights, v_t)  # [B, F, V]
                time_contexts.append(ctx_t.transpose(1, 2).unsqueeze(-1))

            head_outputs.append(torch.cat(time_contexts, dim=-1))

        out = torch.cat(head_outputs, dim=1)
        out = self.proj_conv(out)
        return out, None
    
class MultiHeadSelfAttention2DOnFrame(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "prelu",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        time_window=5,
        *args,
        **kwargs,
    ):
        super(MultiHeadSelfAttention2DOnFrame, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim
        self.time_window=time_window

        assert self.in_chan % self.n_head == 0

        self.Queries = nn.ModuleList()
        self.Keys = nn.ModuleList()
        self.Values = nn.ModuleList()

        for _ in range(self.n_head):
            self.Queries.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
                # nn.Conv2d(
                #     self.in_chan,
                #     self.hid_chan,
                #     kernel_size=1,
                # )
            )
            self.Keys.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.hid_chan,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
                # nn.Conv2d(
                #     self.in_chan,
                #     self.hid_chan,
                #     kernel_size=1,
                # )
            )
            self.Values.append(
                ATTConvActNorm(
                    in_chan=self.in_chan,
                    out_chan=self.in_chan // self.n_head,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.n_freqs,
                    is2d=True,
                )
                # nn.Conv2d(
                #     self.in_chan,
                #     self.in_chan // self.n_head,
                #     kernel_size=1,
                # )
            )

        self.attn_concat_proj = ATTConvActNorm(
            in_chan=self.in_chan,
            out_chan=self.in_chan,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.n_freqs,
            is2d=True,
        )
        # self.attn_concat_proj = nn.Conv2d(
        #     self.in_chan,
        #     self.in_chan,
        #     kernel_size=1            
        # )

    def forward(self, x: torch.Tensor, past_k=None, past_v=None, is_streaming=False):
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        print(f"x shape: {x.shape}")

        batch_size, _, time, freq = x.size()
        residual = x

        if is_streaming: #inference by streaming
            
            all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
            all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
            all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]          
            print(f"all_Q[0]: {all_Q[0].shape}, all_K[0]: {all_K[0].shape}, all_V[0]: {all_V[0].shape}")

            all_Q = [q.transpose(1, 2).reshape(batch_size, time, -1) for q in all_Q]  # [B, T, E, F]
            all_K = [k.transpose(1, 2).reshape(batch_size, time, -1) for k in all_K]  # [B, T, E, F]
            all_V = [v.transpose(1, 2).reshape(batch_size, time, -1) for v in all_V]  # [B, T, C/n_head, F]          
            print(f"After transpose and reshape: all_Q[0]: {all_Q[0].shape}, all_K[0]: {all_K[0].shape}, all_V[0]: {all_V[0].shape}")

            Q = torch.stack(all_Q, dim=1)  # [B, n_heads, T, E*F]    B' = B*n_head
            K = torch.stack(all_K, dim=1)  # [B, n_heads, T, E*F]
            V = torch.stack(all_V, dim=1)  # [B, n_heads, T, C*F/n_head]
            
            Q = Q.reshape(batch_size*self.n_head, time, -1)
            K = K.reshape(batch_size*self.n_head, time, -1)
            V = V.reshape(batch_size*self.n_head, time, -1)


            # Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
            # K = torch.cat(all_K, dim=0)  # [B', E, T, F]
            # V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]            

            # Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
            # K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
            # V = V.transpose(1, 2)  # [B', T, C/n_head, F]


            old_shape = V.shape
            V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
            emb_dim = Q.shape[-1]  # C*F/n_head

            kB, kT, kDim = K.shape
            vB, vT, vDim = V.shape
            print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

            if past_k is None:
                past_k = torch.zeros(batch_size, self.n_head, self.window_size, kDim).to(x.device)
                past_v = torch.zeros(batch_size, self.n_head, self.window_size, vDim).to(x.device)
            
            K = K.reshape(batch_size, self.n_head, time, kDim) #[B, E, T, F]
            V = V.reshape(batch_size, self.n_head, time, vDim)

            new_k = torch.cat([past_k[:, :, 1:, :], K], dim=2)  
            new_v = torch.cat([past_v[:, :, 1:, :], V], dim=2)
            
            K = new_k.reshape(kB, self.time_window, kDim)
            V = new_v.reshape(vB, self.time_window, vDim)

            attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
            attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
            V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
            V = V.reshape(old_shape)  # [B', T, C/n_head, F]
            V = V.transpose(1, 2)  # [B', C/n_head, T, F]
            emb_dim = V.shape[1]  # C/n_head

            x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
            x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

            x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
            x = self.attn_concat_proj(x)  # [B, C, T, F]

            x = x + residual

            if self.dim == 4:
                x = x.transpose(-2, -1).contiguous()

            return x, new_k, new_v
        else:     

            all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
            all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
            all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]

            Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
            K = torch.cat(all_K, dim=0)  # [B', E, T, F]
            V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]

            Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
            K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
            V = V.transpose(1, 2)  # [B', T, C/n_head, F]
            old_shape = V.shape
            V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
            emb_dim = Q.shape[-1]  # C*F/n_head

            attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]

            # 应用 Causal Mask (下三角阵)
            # 屏蔽掉右上角（未来信息）
            mask = torch.triu(torch.ones(time, time), diagonal=1).bool().to(x.device)
            attn_mat.masked_fill_(mask, float('-inf'))

            attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
            V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]
            V = V.reshape(old_shape)  # [B', T, C/n_head, F]
            V = V.transpose(1, 2)  # [B', C/n_head, T, F]
            emb_dim = V.shape[1]  # C/n_head

            x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
            x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]

            x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]
            x = self.attn_concat_proj(x)  # [B, C, T, F]

            x = x + residual

            if self.dim == 4:
                x = x.transpose(-2, -1).contiguous()

            return x, None, None

class Recurrent(nn.Module):
    def __init__(
        self, 
        out_channels=128,
        in_channels=512,
        nband=8,
        upsampling_depth=3,
        n_head=4,
        att_hid_chan=4,
        kernel_size: int = 8, 
        stride: int = 1,
        _iter=4
    ):
        super().__init__()
        self.nband = nband

        self.freq_path = nn.ModuleList([
            FreqUConvBlock(out_channels, in_channels, upsampling_depth, nband=nband),
            # MultiHeadSelfAttention2D(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu", norm_type="LayerNormalization4D", dim=4),
            Unified2DAttentionOnFrequency(out_channels, hid_channels=att_hid_chan, freq_bins=1, window_size=50, num_heads=4, need_streaming=False),
            normalizations.get("LayerNormalization4D")((out_channels, 1))
        ])
        
        self.frame_path = nn.ModuleList([
            TimeUConvBlock(out_channels, in_channels, upsampling_depth),
            # MultiHeadSelfAttention2DOnFrame(out_channels, 1, n_head=n_head, hid_chan=att_hid_chan, act_type="prelu", norm_type="LayerNormalization4D", dim=4),
            Unified2DAttentionOnFrame(out_channels, hid_channels=att_hid_chan, freq_bins=1, window_size=50, num_heads=4, need_streaming=True),
            normalizations.get("LayerNormalization4D")((out_channels, 1))
        ])
        
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def freq_time_interp_process(self, x, past_kv=None):
        # Process Frequency Path
        residual_1 = x #x.clone()  #FIXME cmj
        # print(f"freq_time_process: Frequency: {x.shape}")
        freq_fea = self.freq_path[0](x) # B, N, nband, T 
        freq_fea = self.freq_path[1](freq_fea) # B, N, nband, T
        freq_fea = self.freq_path[2](freq_fea) # B, N, nband, T       
        x = freq_fea + residual_1 # B, N, nband, T
        
        # Process Frame Path
        residual_2 = x #x.clone()      #FIXME cmj   
        # print(f"freq_time_process: Frame: {x.shape}")
        frame_fea = self.frame_path[0](x) # B, N, nband, T
        frame_fea = self.frame_path[1](frame_fea) # B, N, nband, T
        frame_fea = self.frame_path[2](frame_fea) # B, N, nband, T
        x = frame_fea + residual_2 # B, N, nband, T
        return x
    
    def forward(self, x, past_kv=None,):
        # B, nband, N, T
        B, nband, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous() # B, N, nband, T
        mixture = x #x.clone()  #FIXME cmj

        x = self.freq_time_interp_process(x, B, nband, N, T) # B, N, nband, T
        for i in range(1, self.iter):
            print(f"freq_time_process iter-{i}...")
            x = self.freq_time_interp_process(self.concat_block(mixture + x), B, nband, N, T) # B, N, nband, T
                
        return x.permute(0, 2, 1, 3).contiguous() # B, nband, N, T


#input: (B, C, F, T)  
class StatefullLookaheadConv2dOld(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(1, 1), dilation=(1, 1), lookahead=0, groups=1):
        super().__init__()
        self.k_f, self.k_t=kernel_size
        self.d_f, self.d_t=dilation
        self.lookahead = lookahead
        self.receptive_field_width = (self.k_t - 1) * self.d_t #real receptive_field_width should be: (self.k_t - 1) * self.d_t+1
        self.lookback = self.receptive_field_width - self.lookahead
        self.padding_f = ((self.k_f-1)*self.d_f)//2 #receptive_field_width_on_frequency//2

        self.conv=nn.Conv2d(in_ch, out_ch, kernel_size, dilation=dilation, padding=(self.padding_f, 0), groups=groups)

    def forward(self, x, state=None): #input: (B, C, F, T)， state:(B, C, F, lookback+lookahead) == (B, C, F, receptive_field_width)
        B, C, F, T = x.shape
        if state is None:
            state=torch.zeros(B, C, F, self.lookback+self.lookahead, device = x.device)
        # print(f"StatefullLookaheadConv2d: x {x.shape}, state {state.shape}")
        combined_x=torch.cat([state, x], dim=-1) #B, C, F, (self.lookback+self.lookahead)+T

        x = self.conv(combined_x)  #B, C, F, T
        udp_state=combined_x[:, :, :, -(self.lookback+self.lookahead):].detach() #updated state

        return x, udp_state


class StatefullLookaheadConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=(1, 1), dilation=(1, 1), lookahead=0, groups=1):
        super().__init__()
        self.k_f, self.k_t=kernel_size
        self.d_f, self.d_t=dilation
        self.lookahead = lookahead
        self.receptive_field_t = (self.k_t - 1) * self.d_t + 1
        self.lookback = self.receptive_field_t - 1 - self.lookahead
        assert self.lookback>=0, "Lookahead has exceed the receptive_field"
        self.padding_f = ((self.k_f-1) * self.d_f)//2 #receptive_field_width_on_frequency//2

        self.conv=nn.Conv2d(in_ch, out_ch, kernel_size, dilation=dilation, padding=(self.padding_f, 0), groups=groups)

    def forward(self, x, state=None): #input: (B, C, F, T)， state:(B, C, F, lookback+lookahead) == (B, C, F, receptive_field_width-1)
        # print(f"StatefullLookaheadConv2d x: {x.shape}, state: {state.shape}")
        B, C, _, T = x.shape #B, C, F, T
        if state is not None: #inferencing mode
            combined_x=torch.cat([state, x], dim=-1) #B, C, F, (self.lookback+self.lookahead)+T
            #FIXME padd to (left, right, top, bottom)
            # padded_combined_x = F.pad(combined_x, (0, 0, self.padding_f, self.padding_f))  #Note Make import onnx failed
            
            out = self.conv(combined_x)  #B, C, F, T
            udp_state=combined_x[:, :, :, T:].detach() #updated state
        else:
            padded_x=F.pad(x, (self.lookback, self.lookahead, self.padding_f, self.padding_f))
            out = self.conv(padded_x)
            udp_state = None
       
        return out, udp_state

class StatefulDilatedConv2dNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, d=(1,1), groups=1, lookahead=0):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = StatefullLookaheadConv2d(
            nIn,
            nOut,
            kernel_size=kSize,            
            dilation=d,
            groups=groups,
            lookahead=lookahead
        )

        self.norm = LayerNorm2DOnChannel(nOut, eps=1e-8) #nn.LayerNorm(1, eps=1e-8) #normalize channel dim

    def forward(self, input, state=None):        
        output, new_state = self.conv(input, state)
        output = self.norm(output)        
        return output, new_state


class StatefulTimeUConvBlock(nn.Module):
    """
    This class defines the block which performs successive downsampling and
    upsampling in order to be able to analyze the input features in multiple
    resolutions.
    """

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True, hidden_channels=None, dilations=(1, 2, 4), kernel_size=3, fusion_kernel_size=1, global_kernel_size=5):
        super().__init__()
        assert len(dilations) == 3, "StatefulTimeUConvBlock currently expects exactly 3 dilations/states"
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.dilations = tuple(dilations)
        self.kernel_size = kernel_size
        self.global_kernel_size = global_kernel_size
        self.state_widths = tuple((self.kernel_size - 1) * d for d in self.dilations)
        self.global_state_width = self.global_kernel_size - 1
        self.proj_1x1 = PrjConv2dNormAct(out_channels, self.hidden_channels, 1, stride=1, groups=1)
        self.depth = len(self.dilations)
        self.spp_dw = nn.ModuleList([
            StatefulDilatedConv2dNorm(
                self.hidden_channels,
                self.hidden_channels,
                kSize=(1, self.kernel_size),
                groups=self.hidden_channels,
                d=(1, dilation),
                lookahead=0,
            )
            for dilation in self.dilations
        ])
        
        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(FrameInjectionMultiSum(self.hidden_channels, self.hidden_channels)) 

        self.res_conv = nn.Conv2d(self.hidden_channels, out_channels, 1)

        self.globalatt = StatefulCausalFrameMlp(
            self.hidden_channels,
            self.hidden_channels,
            drop=0.1,
            kernel_size=self.global_kernel_size,
        )
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(FrameInjectionMultiSum(self.hidden_channels, self.hidden_channels, fusion_kernel_size))

    def forward(self, x, state_0=None, state_1=None, state_2=None, global_state=None):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        # print(f"StatefulTimeUConvBlock: x {x.shape}")
        residual = x.clone() # B, N, nband, T
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        x = self.proj_1x1(x)

        output = []
        new_states = []
        layer_states = [state_0, state_1, state_2]
        current = x
        for layer_idx, layer in enumerate(self.spp_dw):
            current, new_state = layer(current, layer_states[layer_idx])
            output.append(current)
            new_states.append(new_state)

        """
        last_output=output[-1]
        last_output_size_on_frame = (last_output.shape[-2], last_output.shape[-1]) #(T, nBand) on last output
       
        fea0 = output0
        fea0_pooled = AdaptiveAvgPool2DOnLastDim()(fea0, last_output_size_on_frame)
        global_f= fea0_pooled

        for fea in output[1:]:           
            fea_pooled = AdaptiveAvgPool2DOnLastDim()(fea, last_output_size_on_frame)           
            global_f = global_f + fea_pooled
        """
        global_f=output[0]
        for fea in output[1:]:     
            global_f = global_f + fea
        ##########################################

        global_f, new_global_state = self.globalatt(global_f, global_state)  # [B, N, nBand, T]
     
        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]           
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1]) #FIXME may need change to self.last_layer[i](x_fused[i], x_fused[i + 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        
        res_output = self.res_conv(expanded)
        
        return res_output + residual, new_states[0], new_states[1], new_states[2], new_global_state


class ContextTimeUConvBlock(nn.Module):
    """
    A chunk-streaming time block that caches a single per-block context tensor.

    The cached context lives in the projected hidden space, so deployment only
    needs one `ctx_i` tensor per iteration instead of exposing every internal
    convolution state.
    """

    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        upsampling_depth=4,
        model_T=True,
        hidden_channels=None,
        dilations=(1, 2, 4),
        kernel_size=3,
        fusion_kernel_size=1,
        global_kernel_size=5,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.dilations = tuple(dilations)
        self.kernel_size = kernel_size
        self.fusion_kernel_size = fusion_kernel_size
        self.global_kernel_size = global_kernel_size
        self.depth = len(self.dilations)

        self.proj_1x1 = PrjConv2dNormAct(out_channels, self.hidden_channels, 1, stride=1, groups=1)
        self.spp_dw = nn.ModuleList([
            StatefulDilatedConv2dNorm(
                self.hidden_channels,
                self.hidden_channels,
                kSize=(1, self.kernel_size),
                groups=self.hidden_channels,
                d=(1, dilation),
                lookahead=0,
            )
            for dilation in self.dilations
        ])
        self.loc_glo_fus = nn.ModuleList([
            CausalFrameInjectionMultiSum(self.hidden_channels, self.hidden_channels, 1)
            for _ in range(self.depth)
        ])
        self.res_conv = nn.Conv2d(self.hidden_channels, out_channels, 1)
        self.globalatt = CausalFrameMlp(
            self.hidden_channels,
            self.hidden_channels,
            drop=0.1,
            kernel_size=self.global_kernel_size,
        )
        self.last_layer = nn.ModuleList([
            CausalFrameInjectionMultiSum(
                self.hidden_channels,
                self.hidden_channels,
                kernel=self.fusion_kernel_size,
            )
            for _ in range(max(0, self.depth - 1))
        ])

        conv_context = sum((self.kernel_size - 1) * dilation for dilation in self.dilations)
        global_context = max(0, self.global_kernel_size - 1)
        fusion_context = max(0, self.depth - 1) * max(0, self.fusion_kernel_size - 1)
        self.context_size = conv_context + global_context + fusion_context

    def forward(self, x, ctx=None):
        batch_size, _, nband, chunk_T = x.shape
        projected = self.proj_1x1(x)

        if ctx is None:
            ctx = projected.new_zeros(batch_size, self.hidden_channels, nband, self.context_size)
        elif self.context_size > 0 and not torch.onnx.is_in_onnx_export():
            assert ctx.shape[-1] == self.context_size, f"expected ctx width {self.context_size}, got {ctx.shape[-1]}"

        if self.context_size > 0:
            projected_full = torch.cat([ctx, projected], dim=-1)
        else:
            projected_full = projected

        output = []
        current = projected_full
        for layer in self.spp_dw:
            current, _ = layer(current, None)
            output.append(current)

        global_f = output[0]
        for fea in output[1:]:
            global_f = global_f + fea
        global_f = self.globalatt(global_f)

        x_fused = []
        for idx in range(self.depth):
            x_fused.append(self.loc_glo_fus[idx](output[idx], global_f))

        expanded = x_fused[-1]
        for i in range(self.depth - 2, -1, -1):
            expanded = self.last_layer[i](x_fused[i], expanded)

        res_output = self.res_conv(expanded)
        valid_output = res_output[:, :, :, -chunk_T:] + x
        if self.context_size > 0:
            next_ctx = projected_full[:, :, :, -self.context_size:].detach()
        else:
            next_ctx = projected_full[:, :, :, :0].detach()

        return valid_output, next_ctx

class RecurrentKV(nn.Module):
    def __init__(
        self, 
        out_channels=128, 
        in_channels=512, 
        nband=8,
        f_upsampling_depth=5,
        t_upsampling_depth=3,
        n_heads=4,
        att_hid_chan=4,
        kernel_size: int = 8, 
        stride: int = 1,
        _iter=4,
        need_streaming=False,
        kv_window_size=18, #FIXME should be total lookback size=18?
        att_val_hid_chan=None,
        time_hidden_channels=None,
        time_dilations=(1, 2, 4),
        time_kernel_size=3,
        time_fusion_kernel_size=1,
    ):
        super().__init__()
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.n_heads=n_heads
        self.nband = nband
        self.att_hid_chan=att_hid_chan
        self.att_val_hid_chan = att_val_hid_chan if att_val_hid_chan is not None else att_hid_chan
        self.kv_window_size=kv_window_size
        self.need_streaming=need_streaming

        print(f"")

        self.freq_path = nn.ModuleList([
            FreqUConvBlock(out_channels, in_channels, f_upsampling_depth, nband=nband),
            Unified2DAttentionOnFreqAsym(out_channels, hid_channels=att_hid_chan, freq_bins=nband, window_size=self.kv_window_size, n_heads=n_heads, value_hid_channels=self.att_val_hid_chan, need_streaming=False),
            LayerNorm2DOnChannel(out_channels)
        ])
        
        self.frame_path = nn.ModuleList([           
            StatefulTimeUConvBlock(
                out_channels,
                in_channels,
                t_upsampling_depth,
                hidden_channels=time_hidden_channels,
                dilations=time_dilations,
                kernel_size=time_kernel_size,
                fusion_kernel_size=time_fusion_kernel_size,
            ),
            Unified2DAttentionOnFrameAsym(out_channels, hid_channels=att_hid_chan, freq_bins=nband, window_size=self.kv_window_size, n_heads=n_heads, value_hid_channels=self.att_val_hid_chan, need_streaming=need_streaming),
            LayerNorm2DOnChannel(out_channels)
        ])
        
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def freq_time_interp_process(self, x, past_kv=None, past_valid_mask=None, state_0=None, state_1=None, state_2=None, global_state=None):
        #x: (B, N, nband, T)
        
        # Process Frequency Path
        residual_1 = x.clone()

        freq_fea = self.freq_path[0](x) # B, N, nband, T
        freq_fea, _ = self.freq_path[1](freq_fea) # B, N, nband, T

        freq_fea = self.freq_path[2](freq_fea) # B, N, nband, T
       
        x2 = freq_fea + residual_1 #(B, N, nband, T) #NOTE cmj make NPU convertion failed
        # x = freq_fea+x
       
        # Process Frame Path
        residual_2 = x2.clone()

        frame_fea, new_state_0, new_state_1, new_state_2, new_global_state = self.frame_path[0](x2, state_0, state_1, state_2, global_state) #(B, N, nband, T)
        frame_fea, new_kv, new_valid_mask = self.frame_path[1](frame_fea, past_kv=past_kv, past_valid_mask=past_valid_mask) # B, N, nband, T
        frame_fea = self.frame_path[2](frame_fea) # B, N, nband, T

        out = frame_fea + residual_2 #B, N, nband, T #Make NPU compilation failed

        return out, new_kv, new_valid_mask, new_state_0, new_state_1, new_state_2, new_global_state
    
    def forward(self, x, past_kvs=None, past_valid_mask=None, prev_states_0=None, prev_states_1=None, prev_states_2=None, prev_global_states=None):
        # B, nband, N, T
        B, nband, N, T = x.shape
        if not torch.onnx.is_in_onnx_export():
            assert T >= 1, "RecurrentKV expects at least one frame"
        x = x.permute(0, 2, 1, 3).contiguous() # B, N, nband, T
        mixture = x.clone()  #FIXME cmj

        state_dim = self.frame_path[0].hidden_channels
        kv_dim = (self.att_hid_chan + self.att_val_hid_chan) * nband

        if past_kvs is None:
            past_kvs = x.new_zeros(B, self.n_heads, self.kv_window_size, kv_dim * self.iter)
        if past_valid_mask is None:
            past_valid_mask = x.new_zeros(B, 1, self.kv_window_size, 1)
        if prev_states_0 is None:
            prev_states_0 = x.new_zeros(B, state_dim * self.iter, nband, self.frame_path[0].state_widths[0])
        if prev_states_1 is None:
            prev_states_1 = x.new_zeros(B, state_dim * self.iter, nband, self.frame_path[0].state_widths[1])
        if prev_states_2 is None:
            prev_states_2 = x.new_zeros(B, state_dim * self.iter, nband, self.frame_path[0].state_widths[2])
        if prev_global_states is None:
            prev_global_states = x.new_zeros(B, state_dim * self.iter, nband, self.frame_path[0].global_state_width)

        new_states_0 = []
        new_states_1 = []
        new_states_2 = []
        new_global_states = []
        new_kvs = []
        new_valid_mask = past_valid_mask

        for i in range(self.iter):
            prev_state_0 = prev_states_0[:, i * state_dim:(i + 1) * state_dim, :, :]
            prev_state_1 = prev_states_1[:, i * state_dim:(i + 1) * state_dim, :, :]
            prev_state_2 = prev_states_2[:, i * state_dim:(i + 1) * state_dim, :, :]
            prev_global_state = prev_global_states[:, i * state_dim:(i + 1) * state_dim, :, :]
            past_kv = past_kvs[:, :, :, i * kv_dim:(i + 1) * kv_dim]

            if i == 0:
                iter_input = x
            else:
                iter_input = self.concat_block(mixture + x)

            x, new_kv, new_valid_mask, new_state_0, new_state_1, new_state_2, new_global_state = self.freq_time_interp_process(
                iter_input,
                past_kv=past_kv,
                past_valid_mask=past_valid_mask,
                state_0=prev_state_0,
                state_1=prev_state_1,
                state_2=prev_state_2,
                global_state=prev_global_state,
            )
            new_states_0.append(new_state_0)
            new_states_1.append(new_state_1)
            new_states_2.append(new_state_2)
            new_global_states.append(new_global_state)
            new_kvs.append(new_kv)

        new_states_0 = torch.cat(new_states_0, dim=1)
        new_states_1 = torch.cat(new_states_1, dim=1)
        new_states_2 = torch.cat(new_states_2, dim=1)
        new_global_states = torch.cat(new_global_states, dim=1)
        new_kvs = torch.cat(new_kvs, dim=-1)

        return x.permute(0, 2, 1, 3).contiguous(), new_kvs, new_valid_mask, new_states_0, new_states_1, new_states_2, new_global_states # B, nband, N, T


class RecurrentKVCtx(nn.Module):
    """
    Streaming separator variant that keeps one hidden-space context tensor per
    iteration instead of exposing internal layer states.
    """

    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        nband=8,
        f_upsampling_depth=5,
        t_upsampling_depth=3,
        n_heads=4,
        att_hid_chan=4,
        kernel_size: int = 8,
        stride: int = 1,
        _iter=4,
        need_streaming=False,
        kv_window_size=18,
        att_val_hid_chan=None,
        time_hidden_channels=None,
        time_dilations=(1, 2, 4),
        time_kernel_size=3,
        time_fusion_kernel_size=1,
        time_global_kernel_size=5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.nband = nband
        self.att_hid_chan = att_hid_chan
        self.att_val_hid_chan = att_val_hid_chan if att_val_hid_chan is not None else att_hid_chan
        self.kv_window_size = kv_window_size
        self.need_streaming = need_streaming
        self.iter = _iter

        self.freq_path = nn.ModuleList([
            FreqUConvBlock(out_channels, in_channels, f_upsampling_depth, nband=nband),
            Unified2DAttentionOnFreqAsym(
                out_channels,
                hid_channels=att_hid_chan,
                freq_bins=nband,
                window_size=self.kv_window_size,
                n_heads=n_heads,
                value_hid_channels=self.att_val_hid_chan,
                need_streaming=False,
            ),
            LayerNorm2DOnChannel(out_channels),
        ])
        self.frame_path = nn.ModuleList([
            ContextTimeUConvBlock(
                out_channels,
                in_channels,
                t_upsampling_depth,
                hidden_channels=time_hidden_channels,
                dilations=time_dilations,
                kernel_size=time_kernel_size,
                fusion_kernel_size=time_fusion_kernel_size,
                global_kernel_size=time_global_kernel_size,
            ),
            Unified2DAttentionOnFrameAsym(
                out_channels,
                hid_channels=att_hid_chan,
                freq_bins=nband,
                window_size=self.kv_window_size,
                n_heads=n_heads,
                value_hid_channels=self.att_val_hid_chan,
                need_streaming=need_streaming,
            ),
            LayerNorm2DOnChannel(out_channels),
        ])
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU(),
        )

    def freq_time_interp_process(self, x, past_kv=None, past_valid_mask=None, time_ctx=None):
        residual_1 = x.clone()
        freq_fea = self.freq_path[0](x)
        freq_fea, _ = self.freq_path[1](freq_fea)
        freq_fea = self.freq_path[2](freq_fea)
        x2 = freq_fea + residual_1

        residual_2 = x2.clone()
        frame_fea, next_ctx = self.frame_path[0](x2, ctx=time_ctx)
        frame_fea, new_kv, new_valid_mask = self.frame_path[1](
            frame_fea,
            past_kv=past_kv,
            past_valid_mask=past_valid_mask,
        )
        frame_fea = self.frame_path[2](frame_fea)
        out = frame_fea + residual_2
        return out, new_kv, new_valid_mask, next_ctx

    def forward(self, x, past_kvs=None, past_valid_mask=None, time_ctx=None):
        B, nband, N, T = x.shape
        if not torch.onnx.is_in_onnx_export():
            assert T >= 1, "RecurrentKVCtx expects at least one frame"
        x = x.permute(0, 2, 1, 3).contiguous()
        mixture = x.clone()

        kv_dim = (self.att_hid_chan + self.att_val_hid_chan) * nband
        ctx_dim = self.frame_path[0].hidden_channels
        ctx_width = self.frame_path[0].context_size

        if past_kvs is None:
            past_kvs = x.new_zeros(B, self.n_heads, self.kv_window_size, kv_dim * self.iter)
        if past_valid_mask is None:
            past_valid_mask = x.new_zeros(B, 1, self.kv_window_size, 1)
        if time_ctx is None:
            time_ctx = x.new_zeros(B, ctx_dim * self.iter, nband, ctx_width)

        new_ctxs = []
        new_kvs = []
        new_valid_mask = past_valid_mask

        for i in range(self.iter):
            prev_ctx = time_ctx[:, i * ctx_dim:(i + 1) * ctx_dim, :, :]
            past_kv = past_kvs[:, :, :, i * kv_dim:(i + 1) * kv_dim]

            if i == 0:
                iter_input = x
            else:
                iter_input = self.concat_block(mixture + x)

            x, new_kv, new_valid_mask, next_ctx = self.freq_time_interp_process(
                iter_input,
                past_kv=past_kv,
                past_valid_mask=past_valid_mask,
                time_ctx=prev_ctx,
            )
            new_ctxs.append(next_ctx)
            new_kvs.append(new_kv)

        new_kvs = torch.cat(new_kvs, dim=-1)
        new_ctxs = torch.cat(new_ctxs, dim=1)

        return x.permute(0, 2, 1, 3).contiguous(), new_kvs, new_valid_mask, new_ctxs


class NPUFriendlyFreqTimeStageCtx(nn.Module):
    """
    A larger, non-shared Freq+Time stage for NPU deployment.
    Compared with the iterative shared-parameter separator, this stage trades
    more parameters for fewer repeated graph copies in the exported model.
    """

    def __init__(
        self,
        out_channels=192,
        in_channels=1024,
        nband=8,
        f_upsampling_depth=5,
        t_upsampling_depth=3,
        n_heads=4,
        att_hid_chan=8,
        kv_window_size=4,
        att_val_hid_chan=None,
        time_hidden_channels=32,
        time_dilations=(1, 2, 4),
        time_kernel_size=3,
        time_fusion_kernel_size=1,
        time_global_kernel_size=5,
    ):
        super().__init__()
        self.freq_block = FreqUConvBlock(out_channels, in_channels, f_upsampling_depth, nband=nband)
        self.freq_attn = Unified2DAttentionOnFreqAsym(
            out_channels,
            hid_channels=att_hid_chan,
            freq_bins=nband,
            window_size=kv_window_size,
            n_heads=n_heads,
            value_hid_channels=att_val_hid_chan,
            need_streaming=False,
        )
        self.freq_norm = LayerNorm2DOnChannel(out_channels)

        self.time_block = ContextTimeUConvBlock(
            out_channels,
            in_channels,
            t_upsampling_depth,
            hidden_channels=time_hidden_channels,
            dilations=time_dilations,
            kernel_size=time_kernel_size,
            fusion_kernel_size=time_fusion_kernel_size,
            global_kernel_size=time_global_kernel_size,
        )
        self.frame_attn = Unified2DAttentionOnFrameAsym(
            out_channels,
            hid_channels=att_hid_chan,
            freq_bins=nband,
            window_size=kv_window_size,
            n_heads=n_heads,
            value_hid_channels=att_val_hid_chan,
            need_streaming=True,
        )
        self.frame_norm = LayerNorm2DOnChannel(out_channels)

    def forward(self, x, past_kv=None, past_valid_mask=None, time_ctx=None):
        residual_1 = x.clone()
        freq_fea = self.freq_block(x)
        freq_fea, _ = self.freq_attn(freq_fea)
        freq_fea = self.freq_norm(freq_fea)
        x2 = freq_fea + residual_1

        residual_2 = x2.clone()
        frame_fea, next_ctx = self.time_block(x2, ctx=time_ctx)
        frame_fea, new_kv, new_valid_mask = self.frame_attn(
            frame_fea,
            past_kv=past_kv,
            past_valid_mask=past_valid_mask,
        )
        frame_fea = self.frame_norm(frame_fea)
        out = frame_fea + residual_2
        return out, new_kv, new_valid_mask, next_ctx


class NPUFriendlyStackedSeparatorCtx(nn.Module):
    """
    Explicitly stacked, non-shared online separator.
    This keeps the TIGER freq/time alternation but reduces exported node count
    by using a small number of large stages instead of many repeated shared
    iterations.
    """

    def __init__(
        self,
        out_channels=192,
        in_channels=1024,
        nband=8,
        num_stages=2,
        f_upsampling_depth=5,
        t_upsampling_depth=3,
        n_heads=4,
        att_hid_chan=8,
        kv_window_size=4,
        att_val_hid_chan=None,
        time_hidden_channels=32,
        time_dilations=(1, 2, 4),
        time_kernel_size=3,
        time_fusion_kernel_size=1,
        time_global_kernel_size=5,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.n_heads = n_heads
        self.nband = nband
        self.att_hid_chan = att_hid_chan
        self.att_val_hid_chan = att_val_hid_chan if att_val_hid_chan is not None else att_hid_chan
        self.kv_window_size = kv_window_size
        self.num_stages = num_stages

        self.stages = nn.ModuleList([
            NPUFriendlyFreqTimeStageCtx(
                out_channels=out_channels,
                in_channels=in_channels,
                nband=nband,
                f_upsampling_depth=f_upsampling_depth,
                t_upsampling_depth=t_upsampling_depth,
                n_heads=n_heads,
                att_hid_chan=att_hid_chan,
                kv_window_size=kv_window_size,
                att_val_hid_chan=self.att_val_hid_chan,
                time_hidden_channels=time_hidden_channels,
                time_dilations=time_dilations,
                time_kernel_size=time_kernel_size,
                time_fusion_kernel_size=time_fusion_kernel_size,
                time_global_kernel_size=time_global_kernel_size,
            )
            for _ in range(num_stages)
        ])
        self.mix_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, 1, groups=1),
                nn.PReLU(),
            )
            for _ in range(max(0, num_stages - 1))
        ])

    def forward(self, x, past_kvs=None, past_valid_mask=None, time_ctx=None):
        B, nband, N, T = x.shape
        if not torch.onnx.is_in_onnx_export():
            assert T >= 1, "NPUFriendlyStackedSeparatorCtx expects at least one frame"
        x = x.permute(0, 2, 1, 3).contiguous()
        mixture = x.clone()

        kv_dim = (self.att_hid_chan + self.att_val_hid_chan) * nband
        ctx_dim = self.stages[0].time_block.hidden_channels
        ctx_width = self.stages[0].time_block.context_size

        if past_kvs is None:
            past_kvs = x.new_zeros(B, self.n_heads, self.kv_window_size, kv_dim * self.num_stages)
        if past_valid_mask is None:
            past_valid_mask = x.new_zeros(B, 1, self.kv_window_size, 1)
        if time_ctx is None:
            time_ctx = x.new_zeros(B, ctx_dim * self.num_stages, nband, ctx_width)

        new_ctxs = []
        new_kvs = []
        new_valid_mask = past_valid_mask

        for stage_idx, stage in enumerate(self.stages):
            prev_ctx = time_ctx[:, stage_idx * ctx_dim:(stage_idx + 1) * ctx_dim, :, :]
            past_kv = past_kvs[:, :, :, stage_idx * kv_dim:(stage_idx + 1) * kv_dim]

            if stage_idx == 0:
                stage_input = x
            else:
                stage_input = self.mix_blocks[stage_idx - 1](mixture + x)

            x, new_kv, new_valid_mask, next_ctx = stage(
                stage_input,
                past_kv=past_kv,
                past_valid_mask=past_valid_mask,
                time_ctx=prev_ctx,
            )
            new_ctxs.append(next_ctx)
            new_kvs.append(new_kv)

        new_kvs = torch.cat(new_kvs, dim=-1)
        new_ctxs = torch.cat(new_ctxs, dim=1)

        return x.permute(0, 2, 1, 3).contiguous(), new_kvs, new_valid_mask, new_ctxs


class BNBlock(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size=(1, 1), groups=1, eps=1e-8):
        super().__init__()        
        self.sgn = BasicLayerNormLastDim(n_in_channels, eps=eps)
        self.conv = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=kernel_size, groups=groups)

    def forward(self, x): #input: (B, 1, bw, T)
        x = x.permute(0, 3, 1, 2) #(B, T, 1, bw)
        x = self.sgn(x) #norm on bw(frequency) dim
        x = x.permute(0, 3, 2, 1) #(B, bw, 1, T)
        x = self.conv(x)

        return x


class MaskBlock(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, ksize, n_groups):
        super().__init__()
        self.act=nn.PReLU()
        self.conv=nn.Conv2d(n_in_channels, n_out_channels, (ksize, 1), groups=n_groups)        

    def forward(self, x):
        x = self.act(x)
        x = self.conv(x)
        return x

class TIGER(BaseModel):
    def __init__(
        self,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        att_n_head=4,
        att_hid_chan=4,
        att_kernel_size=8, 
        att_stride=1,
        win=2048, 
        stride=512,
        num_sources=2,
        sample_rate=44100,
        need_streaming=False,
        kv_window_size=18,
        pre_calc_bands=None,
        att_val_hid_chan=None,
        time_hidden_channels=None,
        time_dilations=(1, 2, 4),
        time_kernel_size=3,
        time_fusion_kernel_size=1,

    ):
        super(TIGER, self).__init__(sample_rate=sample_rate)
        
        self.sample_rate = sample_rate
        self.win = win
        self.stride = stride
        self.group = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = out_channels
        self.num_output = num_sources
        self.eps = torch.finfo(torch.float32).eps

        # Calculate band widths using utility function
        if pre_calc_bands is None:
            self.band_width = calculate_band_widths(self.enc_dim, sample_rate) #FIXME
        else:
            self.band_width = pre_calc_bands

        self.nband = len(self.band_width)
        print(f"model nband: {len(self.band_width)} => {self.band_width}")
       
        feature_extractors=[] #nn.ModuleList
        mask_decoders=[]
        for bw in self.band_width:
            fea_ex = BNBlock(bw*2, self.feature_dim, kernel_size=(1, 1), groups=1, eps=self.eps)
            feature_extractors.append(fea_ex)

            mask_dec = MaskBlock(self.feature_dim, bw*4*num_sources, 1, num_sources)
            mask_decoders.append(mask_dec)

        self.feature_extractors = nn.ModuleList(feature_extractors)
        self.mask_decoders = nn.ModuleList(mask_decoders)               
        
        self.separator = RecurrentKV(
            self.feature_dim,
            in_channels,
            self.nband,
            upsampling_depth,
            3,
            att_n_head,
            att_hid_chan,
            att_kernel_size,
            att_stride,
            num_blocks,
            need_streaming,
            kv_window_size,
            att_val_hid_chan=att_val_hid_chan,
            time_hidden_channels=time_hidden_channels,
            time_dilations=time_dilations,
            time_kernel_size=time_kernel_size,
            time_fusion_kernel_size=time_fusion_kernel_size,
        )
        self.supports_exact_chunk_training = False

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        kv_dim = (self.separator.att_hid_chan + self.separator.att_val_hid_chan) * self.nband
        state_dim = self.separator.frame_path[0].hidden_channels
        state_widths = self.separator.frame_path[0].state_widths
        global_state_width = self.separator.frame_path[0].global_state_width
        return (
            torch.zeros(batch_size, self.separator.n_heads, self.separator.kv_window_size, kv_dim * self.separator.iter, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, self.separator.kv_window_size, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, state_dim * self.separator.iter, self.nband, state_widths[0], device=device, dtype=dtype),
            torch.zeros(batch_size, state_dim * self.separator.iter, self.nband, state_widths[1], device=device, dtype=dtype),
            torch.zeros(batch_size, state_dim * self.separator.iter, self.nband, state_widths[2], device=device, dtype=dtype),
            torch.zeros(batch_size, state_dim * self.separator.iter, self.nband, global_state_width, device=device, dtype=dtype),
        )

    def _encode_subbands(self, subband_spec_RIs):
        band_feats = []
        start_i = 0
        for i, bw in enumerate(self.band_width):
            subband = subband_spec_RIs[:, :, start_i:start_i + bw * 2, :]
            subband_feature = self.feature_extractors[i](subband)
            band_feats.append(subband_feature)
            start_i += bw * 2

        subband_features = torch.cat(band_feats, dim=2)
        return subband_features.transpose(1, 2).contiguous()

    def _decode_masks(self, sep_output):
        batch_size = sep_output.shape[0]
        masked_outputs = []
        for i, bw in enumerate(self.band_width):
            subband_mask_enc = sep_output[:, i:i + 1, :, :]
            subband_mask_enc = subband_mask_enc.permute(0, 2, 3, 1).contiguous()
            subband_mask_dec = self.mask_decoders[i](subband_mask_enc)
            subband_mask_dec = subband_mask_dec.view(batch_size, 4 * self.num_output, bw, -1)
            masked_outputs.append(subband_mask_dec)

        return torch.concat(masked_outputs, dim=-2)

    def forward_cell(self, subband_spec_RIs, past_kvs=None, past_valid_mask=None, prev_states_0=None, prev_states_1=None, prev_states_2=None, prev_global_states=None):
        if not torch.onnx.is_in_onnx_export():
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects a single frame (T=1)"

        subband_features = self._encode_subbands(subband_spec_RIs)
        sep_output, new_kv, new_valid_mask, new_states_0, new_states_1, new_states_2, new_global_states = self.separator(
            subband_features,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
        )
        band_masked_output = self._decode_masks(sep_output)
        return band_masked_output, new_kv, new_valid_mask, new_states_0, new_states_1, new_states_2, new_global_states

    def forward_sequence(self, subband_spec_RIs=None, past_kvs=None, past_valid_mask=None, prev_states_0=None, prev_states_1=None, prev_states_2=None, prev_global_states=None, detach_state=False, chunk_size=8):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        batch_size, _, _, total_frames = subband_spec_RIs.shape

        if past_kvs is None or past_valid_mask is None or prev_states_0 is None or prev_states_1 is None or prev_states_2 is None or prev_global_states is None:
            past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states = self.init_streaming_state(
                batch_size,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        if not self.supports_exact_chunk_training:
            chunk_size = 1
        elif chunk_size is None or chunk_size <= 0:
            chunk_size = total_frames

        frame_outputs = []
        for t in range(0, total_frames, chunk_size):
            chunk = subband_spec_RIs[..., t:t + chunk_size]
            subband_features = self._encode_subbands(chunk)
            sep_output, past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states = self.separator(
                subband_features,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
            frame_outputs.append(self._decode_masks(sep_output))

            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                prev_states_0 = prev_states_0.detach()
                prev_states_1 = prev_states_1.detach()
                prev_states_2 = prev_states_2.detach()
                prev_global_states = prev_global_states.detach()

        return torch.cat(frame_outputs, dim=-1), past_kvs, past_valid_mask, prev_states_0, prev_states_1, prev_states_2, prev_global_states

    def forward(self, subband_spec_RIs=None, past_kvs=None, past_valid_mask=None, prev_states_0=None, prev_states_1=None, prev_states_2=None, prev_global_states=None, detach_state=False, chunk_size=8):
        """
        Export path: pass a single frame (T=1) and explicit states/caches.
        Training path: pass a longer sequence (T>1); the model will unroll the
        same 1-frame streaming cell over time.
        """        
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        if subband_spec_RIs.shape[-1] == 1:
            return self.forward_cell(
                subband_spec_RIs,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                prev_states_0=prev_states_0,
                prev_states_1=prev_states_1,
                prev_states_2=prev_states_2,
                prev_global_states=prev_global_states,
            )
        return self.forward_sequence(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
            detach_state=detach_state,
            chunk_size=chunk_size,
        )
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args


class TIGERStreamingTrainingWrapper(nn.Module):
    """
    Thin training wrapper around the exportable 1-frame TIGER cell.
    This lets long RI sequences reuse the exact same state/cache flow used in deployment.
    """

    def __init__(self, streaming_model: TIGER, detach_state: bool = False, chunk_size: int = 8):
        super().__init__()
        self.streaming_model = streaming_model
        self.detach_state = detach_state
        self.chunk_size = chunk_size

    def forward(self, subband_spec_RIs, past_kvs=None, past_valid_mask=None, prev_states_0=None, prev_states_1=None, prev_states_2=None, prev_global_states=None):
        return self.streaming_model.forward_sequence(
            subband_spec_RIs=subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            prev_states_0=prev_states_0,
            prev_states_1=prev_states_1,
            prev_states_2=prev_states_2,
            prev_global_states=prev_global_states,
            detach_state=self.detach_state,
            chunk_size=self.chunk_size,
        )


class TIGERCtx(TIGER):
    """
    TIGER variant whose time path exposes one packed context tensor per
    iteration: `ctx_i`, instead of per-layer internal states.
    """

    def __init__(
        self,
        *args,
        time_global_kernel_size=5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.separator = RecurrentKVCtx(
            self.feature_dim,
            kwargs.get("in_channels", 512),
            self.nband,
            kwargs.get("upsampling_depth", 4),
            3,
            kwargs.get("att_n_head", 4),
            kwargs.get("att_hid_chan", 4),
            kwargs.get("att_kernel_size", 8),
            kwargs.get("att_stride", 1),
            kwargs.get("num_blocks", 16),
            kwargs.get("need_streaming", False),
            kwargs.get("kv_window_size", 18),
            att_val_hid_chan=kwargs.get("att_val_hid_chan"),
            time_hidden_channels=kwargs.get("time_hidden_channels"),
            time_dilations=kwargs.get("time_dilations", (1, 2, 4)),
            time_kernel_size=kwargs.get("time_kernel_size", 3),
            time_fusion_kernel_size=kwargs.get("time_fusion_kernel_size", 1),
            time_global_kernel_size=time_global_kernel_size,
        )
        self.supports_exact_chunk_training = True

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        kv_dim = (self.separator.att_hid_chan + self.separator.att_val_hid_chan) * self.nband
        ctx_dim = self.separator.frame_path[0].hidden_channels
        ctx_width = self.separator.frame_path[0].context_size
        return (
            torch.zeros(batch_size, self.separator.n_heads, self.separator.kv_window_size, kv_dim * self.separator.iter, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, self.separator.kv_window_size, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, ctx_dim * self.separator.iter, self.nband, ctx_width, device=device, dtype=dtype),
        )

    def forward_cell(self, subband_spec_RIs, past_kvs=None, past_valid_mask=None, time_ctx=None):
        if not torch.onnx.is_in_onnx_export():
            assert subband_spec_RIs.shape[-1] == 1, "forward_cell expects a single frame (T=1)"
        subband_features = self._encode_subbands(subband_spec_RIs)
        sep_output, new_kv, new_valid_mask, new_time_ctx = self.separator(
            subband_features,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            time_ctx=time_ctx,
        )
        band_masked_output = self._decode_masks(sep_output)
        return band_masked_output, new_kv, new_valid_mask, new_time_ctx

    def forward_sequence(self, subband_spec_RIs=None, past_kvs=None, past_valid_mask=None, time_ctx=None, detach_state=False, chunk_size=8):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        batch_size, _, _, total_frames = subband_spec_RIs.shape

        if past_kvs is None or past_valid_mask is None or time_ctx is None:
            past_kvs, past_valid_mask, time_ctx = self.init_streaming_state(
                batch_size,
                device=subband_spec_RIs.device,
                dtype=subband_spec_RIs.dtype,
            )

        if chunk_size is None or chunk_size <= 0:
            chunk_size = total_frames

        frame_outputs = []
        for t in range(0, total_frames, chunk_size):
            chunk = subband_spec_RIs[..., t:t + chunk_size]
            subband_features = self._encode_subbands(chunk)
            sep_output, past_kvs, past_valid_mask, time_ctx = self.separator(
                subband_features,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                time_ctx=time_ctx,
            )
            frame_outputs.append(self._decode_masks(sep_output))

            if detach_state:
                past_kvs = past_kvs.detach()
                past_valid_mask = past_valid_mask.detach()
                time_ctx = time_ctx.detach()

        return torch.cat(frame_outputs, dim=-1), past_kvs, past_valid_mask, time_ctx

    def forward(self, subband_spec_RIs=None, past_kvs=None, past_valid_mask=None, time_ctx=None, detach_state=False, chunk_size=8):
        assert subband_spec_RIs is not None, "subband_spec_RIs is required"
        if subband_spec_RIs.shape[-1] == 1:
            return self.forward_cell(
                subband_spec_RIs,
                past_kvs=past_kvs,
                past_valid_mask=past_valid_mask,
                time_ctx=time_ctx,
            )
        return self.forward_sequence(
            subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            time_ctx=time_ctx,
            detach_state=detach_state,
            chunk_size=chunk_size,
        )


class TIGERCtxStreamingTrainingWrapper(nn.Module):
    def __init__(self, streaming_model: TIGERCtx, detach_state: bool = False, chunk_size: int = 8):
        super().__init__()
        self.streaming_model = streaming_model
        self.detach_state = detach_state
        self.chunk_size = chunk_size

    def forward(self, subband_spec_RIs, past_kvs=None, past_valid_mask=None, time_ctx=None):
        return self.streaming_model.forward_sequence(
            subband_spec_RIs=subband_spec_RIs,
            past_kvs=past_kvs,
            past_valid_mask=past_valid_mask,
            time_ctx=time_ctx,
            detach_state=self.detach_state,
            chunk_size=self.chunk_size,
        )


class TIGERDeployable(TIGER):
    """
    A more deployment-oriented preset:
    - compressed frame KV value channels
    - smaller hidden state in the time path
    - shorter causal state widths
    """

    def __init__(self, *args, **kwargs):
        out_channels = kwargs.get("out_channels", 128)
        att_hid_chan = kwargs.get("att_hid_chan", 4)
        kwargs.setdefault("kv_window_size", 4)
        kwargs.setdefault("att_val_hid_chan", att_hid_chan)
        kwargs.setdefault("time_hidden_channels", max(16, out_channels // 8))
        kwargs.setdefault("time_dilations", (1, 1, 2))
        kwargs.setdefault("time_kernel_size", 3)
        kwargs.setdefault("time_fusion_kernel_size", 1)
        super().__init__(*args, **kwargs)
        self.supports_exact_chunk_training = True


class TIGERTigerLikeApprox(TIGER):
    """
    A more TIGER-like preset that keeps the online/exportable structure but uses
    a wider time path and a larger fusion kernel to better match TIGER's local
    aggregation behavior.
    """

    def __init__(self, *args, **kwargs):
        in_channels = kwargs.get("in_channels", 512)
        att_hid_chan = kwargs.get("att_hid_chan", 4)
        kwargs.setdefault("kv_window_size", 6)
        kwargs.setdefault("att_val_hid_chan", att_hid_chan * 2)
        kwargs.setdefault("time_hidden_channels", in_channels)
        kwargs.setdefault("time_dilations", (1, 2, 3))
        kwargs.setdefault("time_kernel_size", 5)
        kwargs.setdefault("time_fusion_kernel_size", 5)
        super().__init__(*args, **kwargs)


class TIGERCtxDeployable(TIGERCtx):
    """
    Deployment-oriented ctx_i variant.
    The cached time context is stored once per iteration in hidden space.
    """

    def __init__(self, *args, **kwargs):
        out_channels = kwargs.get("out_channels", 128)
        att_hid_chan = kwargs.get("att_hid_chan", 4)
        kwargs.setdefault("kv_window_size", 4)
        kwargs.setdefault("att_val_hid_chan", att_hid_chan)
        kwargs.setdefault("time_hidden_channels", max(16, out_channels // 8))
        kwargs.setdefault("time_dilations", (1, 1, 2))
        kwargs.setdefault("time_kernel_size", 3)
        kwargs.setdefault("time_fusion_kernel_size", 1)
        kwargs.setdefault("time_global_kernel_size", 5)
        super().__init__(*args, **kwargs)


class TIGERCtxTigerLikeApprox(TIGERCtx):
    """
    A more TIGER-like ctx_i variant. It keeps a larger hidden space and a wider
    temporal kernel, but still exposes only one packed context tensor per
    iteration.
    """

    def __init__(self, *args, **kwargs):
        in_channels = kwargs.get("in_channels", 512)
        att_hid_chan = kwargs.get("att_hid_chan", 4)
        kwargs.setdefault("kv_window_size", 6)
        kwargs.setdefault("att_val_hid_chan", att_hid_chan * 2)
        kwargs.setdefault("time_hidden_channels", in_channels)
        kwargs.setdefault("time_dilations", (1, 2, 3))
        kwargs.setdefault("time_kernel_size", 5)
        kwargs.setdefault("time_fusion_kernel_size", 5)
        kwargs.setdefault("time_global_kernel_size", 5)
        super().__init__(*args, **kwargs)


class TIGERNPULargeCtx(TIGERCtx):
    """
    Larger online TIGER variant for NPU deployment.
    It keeps the TIGER-style subband encoder/decoder and freq/time alternation,
    but replaces the shared iterative separator with a small number of explicit,
    wider stages to reduce exported node count.
    """

    def __init__(
        self,
        *args,
        num_stages=2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.separator = NPUFriendlyStackedSeparatorCtx(
            self.feature_dim,
            kwargs.get("in_channels", 1024),
            self.nband,
            num_stages=num_stages,
            f_upsampling_depth=kwargs.get("upsampling_depth", 5),
            t_upsampling_depth=3,
            n_heads=kwargs.get("att_n_head", 4),
            att_hid_chan=kwargs.get("att_hid_chan", 8),
            kv_window_size=kwargs.get("kv_window_size", 4),
            att_val_hid_chan=kwargs.get("att_val_hid_chan"),
            time_hidden_channels=kwargs.get("time_hidden_channels", 32),
            time_dilations=kwargs.get("time_dilations", (1, 2, 4)),
            time_kernel_size=kwargs.get("time_kernel_size", 3),
            time_fusion_kernel_size=kwargs.get("time_fusion_kernel_size", 1),
            time_global_kernel_size=kwargs.get("time_global_kernel_size", 5),
        )
        self.supports_exact_chunk_training = True

    def init_streaming_state(self, batch_size, device=None, dtype=None):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        kv_dim = (self.separator.att_hid_chan + self.separator.att_val_hid_chan) * self.nband
        ctx_dim = self.separator.stages[0].time_block.hidden_channels
        ctx_width = self.separator.stages[0].time_block.context_size
        return (
            torch.zeros(batch_size, self.separator.n_heads, self.separator.kv_window_size, kv_dim * self.separator.num_stages, device=device, dtype=dtype),
            torch.zeros(batch_size, 1, self.separator.kv_window_size, 1, device=device, dtype=dtype),
            torch.zeros(batch_size, ctx_dim * self.separator.num_stages, self.nband, ctx_width, device=device, dtype=dtype),
        )


class TIGERNPULargeDeployable(TIGERNPULargeCtx):
    """
    Deployment preset targeting lower exported node count with a larger
    parameter budget.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("out_channels", 192)
        kwargs.setdefault("in_channels", 1024)
        kwargs.setdefault("upsampling_depth", 5)
        kwargs.setdefault("att_n_head", 4)
        kwargs.setdefault("att_hid_chan", 8)
        kwargs.setdefault("kv_window_size", 4)
        kwargs.setdefault("att_val_hid_chan", 8)
        kwargs.setdefault("time_hidden_channels", 24)
        kwargs.setdefault("time_dilations", (1, 2, 4))
        kwargs.setdefault("time_kernel_size", 3)
        kwargs.setdefault("time_fusion_kernel_size", 1)
        kwargs.setdefault("time_global_kernel_size", 5)
        kwargs.setdefault("num_stages", 2)
        super().__init__(*args, **kwargs)
