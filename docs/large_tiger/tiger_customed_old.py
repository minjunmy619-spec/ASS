import inspect
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel
from ..layers import activations, normalizations
from ..utils.signal_processing import calculate_band_widths


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
        self.norm=nn.LayerNorm(nOut, eps=eps)   

    def forward(self, x: torch.Tensor):        
        x = x.permute(0, 2, 3, 1)        
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

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
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = FreqConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = FreqConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = FreqConv2dNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g):
        """
        x_g: global features
        x_l: local features
        """
        B, N, nBand, T = x_l.shape
        local_feat = self.local_embedding(x_l)       

        global_act = self.global_act(x_g)
        global_act = self.act(global_act)
        # sig_act = F.interpolate(global_act, size=(nBand, T), mode="nearest")
        sig_act = AdaptiveAvgPool2DOnLastDim()(global_act, (nBand, T))
        # sig_act = F.adaptive_avg_pool2d(global_act, (nBand, T))

        global_feat = self.global_embedding(x_g)
        # print(f"Freq local_feat: {local_feat.shape} vs global_act:{global_act.shape} vs global_feat: {global_feat.shape}, interploation with: ({nBand},{T})")

        # global_feat = F.interpolate(global_feat, size=(nBand, T), mode="nearest")
        global_feat = AdaptiveAvgPool2DOnLastDim()(global_feat, (nBand, T))
        # global_feat = F.adaptive_avg_pool2d(global_feat, (nBand, T))

        # print(f"FrequencyInjectionMultiSum: x_l: {x_l.shape}, x_g: {x_g.shape} local_feat: {local_feat.shape} nBand={nBand} T={T} <= global_act: {global_act.shape} <= global_feat: {global_feat.shape}")
        
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
        # return F.interpolate(x, output_size)
   
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

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True):
        super().__init__()
        self.proj_1x1 = PrjConv2dNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
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
            self.loc_glo_fus.append(FrequencyInjectionMultiSum(in_channels, in_channels)) 

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.globalatt = FrequencyMlp(in_channels, in_channels, drop=0.1)
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(FrequencyInjectionMultiSum(in_channels, in_channels, 5))

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

        last_output=output[-1]

        last_output_size_on_freq = (last_output.shape[-1], last_output.shape[-2]) #(T, nBand) on last output
        
        fea0 = output[0]
        fea0 = fea0.transpose(2, 3)
        fea0_pooled = AdaptiveAvgPool2DOnLastDim()(fea0, last_output_size_on_freq)
        # fea0_pooled = F.adaptive_avg_pool2d(fea0, output_size=last_output_size_on_freq) #FIXME onnx export error

        global_f= fea0_pooled.transpose(2, 3)
        for fea in output[1 : -1]: #FIXME skip the last one, for it already in same size   
            fea = fea.transpose(2, 3) #change to B, N, T, nBand
            fea_pooled = AdaptiveAvgPool2DOnLastDim()(fea, last_output_size_on_freq)            
            global_f = global_f + fea_pooled.transpose(2, 3) 
        global_f = global_f+output[-1] #FIXME the last one already in same size

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
    
class MultiHeadSelfAttention2D(nn.Module):
    def __init__(
        self,
        in_chan: int,
        n_freqs: int,
        n_head: int = 4,
        hid_chan: int = 4,
        act_type: str = "prelu",
        norm_type: str = "LayerNormalization4D",
        dim: int = 3,
        *args,
        **kwargs,
    ):
        super(MultiHeadSelfAttention2D, self).__init__()
        self.in_chan = in_chan
        self.n_freqs = n_freqs
        self.n_head = n_head
        self.hid_chan = hid_chan
        self.act_type = act_type
        self.norm_type = norm_type
        self.dim = dim

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

    def forward(self, x: torch.Tensor):
        print(f"input: {x.shape}")
        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        print(f"input after transpose: {x.shape}")
        batch_size, _, time, freq = x.size()
        residual = x

        all_Q = [q(x) for q in self.Queries]  # [B, E, T, F]
        all_K = [k(x) for k in self.Keys]  # [B, E, T, F]
        all_V = [v(x) for v in self.Values]  # [B, C/n_head, T, F]
        print(f"heads: {len(all_Q)}: Q[0]: {all_Q[0].shape}, K[0]: {all_K[0].shape}, V[0]: {all_V[0].shape}")

        Q = torch.cat(all_Q, dim=0)  # [B', E, T, F]    B' = B*n_head
        K = torch.cat(all_K, dim=0)  # [B', E, T, F]
        V = torch.cat(all_V, dim=0)  # [B', C/n_head, T, F]
        print(f"after cat: Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

        Q = Q.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        K = K.transpose(1, 2).flatten(start_dim=2)  # [B', T, E*F]
        V = V.transpose(1, 2)  # [B', T, C/n_head, F]

        print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")
        old_shape = V.shape
        V = V.flatten(start_dim=2)  # [B', T, C*F/n_head]
        print(f"V after flatten: {V.shape}, old shape: {old_shape}")
        emb_dim = Q.shape[-1]  # C*F/n_head

        attn_mat = torch.matmul(Q, K.transpose(1, 2)) / (emb_dim**0.5)  # [B', T, T]
        print(f"attn_mat: {attn_mat.shape}")
        attn_mat = F.softmax(attn_mat, dim=2)  # [B', T, T]
        V = torch.matmul(attn_mat, V)  # [B', T, C*F/n_head]

        print(f"after att V: {V.shape}")
        V = V.reshape(old_shape)  # [B', T, C/n_head, F]
        V = V.transpose(1, 2)  # [B', C/n_head, T, F]
        emb_dim = V.shape[1]  # C/n_head

        x = V.view([self.n_head, batch_size, emb_dim, time, freq])  # [n_head, B, C/n_head, T, F]
        x = x.transpose(0, 1).contiguous()  # [B, n_head, C/n_head, T, F]
        print(f"after att transpose: {x.shape}")

        x = x.view([batch_size, self.n_head * emb_dim, time, freq])  # [B, C, T, F]

        print(f"before attn_concat_proj: {x.shape}")
        x = self.attn_concat_proj(x)  # [B, C, T, F]

        x = x + residual

        print(f"after add residual: {x.shape}")

        if self.dim == 4:
            x = x.transpose(-2, -1).contiguous()
        print(f"befor return : {x.shape}")
        return x


class Unified2DAttentionOnFrequency(nn.Module):
    def __init__(self, in_channels, hid_channels=4, freq_bins=1, window_size=50, num_heads=4, act_type: str = "prelu",  norm_type: str = "LayerNormalization4D", need_streaming=False):
        super().__init__()
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.hid_channels=hid_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.need_streaming=need_streaming
        
        # self.total_dim = in_channels * freq_bins #FIXME
        # self.total_dim = hid_channels * freq_bins
        # self.head_dim = self.total_dim // num_heads
        
        # 1x1 卷积产生 QKV [B, 3*C, F, T]
        # self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        # self.proj_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)        

        self.qkv_conv = ATTConvActNorm(
                    in_chan=self.in_channels,
                    out_chan=self.hid_channels*3,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.freq_bins,
                    is2d=True,
                )
        self.proj_conv = ATTConvActNorm(
            in_chan=self.in_channels,
            out_chan=self.in_channels,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.freq_bins,
            is2d=True,
        )

        # print(f"head_dim: {self.head_dim}")

    def forward(self, x):
        """
        x: [B, C, F, T]
        - 训练时: T > 1, past_kv 为 None, is_streaming=False
        - 推理时: T = 1, past_kv 为 [B*H, 2*D, 1, W], is_streaming=True
        """
        B, inpC, F, T = x.shape  #Make sure the input shape:(B, inpC, F, T)
        print(f"Freq x shape: {x.shape}")

        # x = x.transpose(-2, -1).contiguous() #FIXME NOT need for frequency: transpose => B, C, T, F        
        
        # 1. 投影产生 QKV 并折叠 Heads 到 Batch 维度
        qkv = self.qkv_conv(x) # [B, 3*C, T, F]
        C =self.hid_channels  #FIXME using new channels
        print(f"Freq qkv: {qkv.shape}")

        # 4D 变形: [B*num_heads, 3*head_dim, 1, T]
        # qkv_flat = qkv.reshape(B * self.num_heads, 3 * self.head_dim, 1, T)

        qkv = qkv.transpose(1,2) #[B, F, 3*C, T]
        # qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim) #[B, T, nheads, 3*C*F/nheads]=>[B, T, nheads, 3*head_dim]
        qkv = qkv.reshape(B, F, self.num_heads, -1) #[B, F, nheads, 3*C*T/nheads]=>[B, F, nheads, 3*head_dim]
        head_dim = qkv.shape[-1]//3  #FIXME caluclate here

        qkv = qkv.transpose(1, 2) #[B, nheads, F, 3*head_dim]        
        q=qkv[:, :, :, 0:head_dim] #[B, nheads, F, head_dim]
        k=qkv[:, :, :, head_dim:2*head_dim]
        v=qkv[:, :, :, 2*head_dim:]

        print(f"Freq q: {q.shape}, k: {k.shape}, v: {v.shape}")

        q_seq = q.reshape(B * self.num_heads, F, head_dim) # [B*H, F, D]
        k_seq = k.reshape(B * self.num_heads, F, head_dim).transpose(1, 2)  # [B*H, D, F]
        v_seq = v.reshape(B * self.num_heads, F, head_dim) # [B*H, F, D]
        
        # 这里的每一帧 Q 都会与之前所有的 K 计算得分: [B*H, F, F]
        attn_scores = torch.bmm(q_seq, k_seq) * (head_dim ** -0.5)
        
        attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1) # [B*H, F, F]
        context = torch.bmm(attn_weights, v_seq) # [B*H, F, D]

        # 还原回 4D: [B, C, T, F]
        out = context.transpose(1, 2).reshape(B, C, T, F)
        out = self.proj_conv(out)

        out = out.transpose(-2, -1).contiguous() #Back to (B, C, F, T)
        
        return out


class Unified2DAttentionOnFrame(nn.Module):
    def __init__(self, in_channels, hid_channels=4, freq_bins=1, window_size=50, num_heads=4, act_type: str = "prelu",  norm_type: str = "LayerNormalization4D", need_streaming=True):
        super().__init__()
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.hid_channels=hid_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.need_streaming=need_streaming
        
        # self.total_dim = in_channels * freq_bins #FIXME
        # self.total_dim = hid_channels * freq_bins
        # self.head_dim = self.total_dim // num_heads
        
        # 1x1 卷积产生 QKV [B, 3*C, F, T]
        # self.qkv_conv = nn.Conv2d(in_channels, in_channels * 3, kernel_size=1)
        # self.proj_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)        

        self.qkv_conv = ATTConvActNorm(
                    in_chan=self.in_channels,
                    out_chan=self.hid_channels*3,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.freq_bins,
                    is2d=True,
                )
        self.proj_conv = ATTConvActNorm(
            in_chan=self.in_channels,
            out_chan=self.in_channels,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.freq_bins,
            is2d=True,
        )

        

    def forward(self, x, past_kv=None):
        """
        x: [B, C, F, T]
        - 训练时: T > 1, past_kv 为 None, is_streaming=False
        - 推理时: T = 1, past_kv 为 [B*H, 2*D, 1, W], is_streaming=True
        """
        B, inpC, F, T = x.shape #Make sure input shape is (B, inpC, F, T)
        print(f"x shape: {x.shape}")

        x = x.transpose(-2, -1).contiguous() #FIXME NOT need for frequency: transpose => B, C, T, F        
        
        # 1. 投影产生 QKV 并折叠 Heads 到 Batch 维度
        qkv = self.qkv_conv(x) # [B, 3*C, T, F]
        C =self.hid_channels  #FIXME using new channels
        print(f"Frame qkv: {qkv.shape}")

        # 4D 变形: [B*num_heads, 3*head_dim, 1, T]
        # qkv_flat = qkv.reshape(B * self.num_heads, 3 * self.head_dim, 1, T)

        qkv = qkv.transpose(1, 2) #[B, T, 3*C, F]
        # qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim) #[B, T, nheads, 3*C*F/nheads]=>[B, T, nheads, 3*head_dim]
        qkv = qkv.reshape(B, T, self.num_heads, -1) #[B, T, nheads, 3*C*F/nheads]=>[B, T, nheads, 3*head_dim]
        head_dim = qkv.shape[-1]//3  #FIXME caluclate here
        print(f"Frame qkv after reshape: {qkv.shape}, head_dim: {head_dim}")

        qkv = qkv.transpose(1, 2) #[B, nheads, T, 3*head_dim]
        # qkv = qkv.reshape(B, self.num_heads, T, 3 * head_dim*F) #[B, nheads, T, 3*C*F/nheads]=>[B, nheads, T, 3*head_dim]
        q=qkv[:, :, :, 0:head_dim] #[B, nheads, T, head_dim]
        k=qkv[:, :, :, head_dim:2*head_dim]
        v=qkv[:, :, :, 2*head_dim:]

        # qkv_flat = qkv.reshape(B, self.num_heads, 3 * head_dim, F)
        # print(f"qkv_flat: {qkv_flat.shape}")       
        # q, k, v = qkv_flat[:, :, 0:head_dim, :], qkv_flat[:, :, head_dim:2*head_dim, :], qkv_flat[:, :, head_dim*2:3*head_dim, :]
        print(f"Frame q: {q.shape}, k: {k.shape}, v: {v.shape}")
        
        if not self.training and self.need_streaming:
            # --- 【推理模式：流式 + KV Cache】 ---
            if past_kv is None:
                # past_kv = torch.zeros(B * self.num_heads, head_dim * 2, 1, self.window_size).to(x.device)
                past_kv = torch.zeros(B, self.num_heads, self.window_size, head_dim * 2).to(x.device)
            
            prev_k, prev_v = past_kv[:, :, :, 0:head_dim], past_kv[:, :, :, head_dim:] #(B, n_heads, T, head_dim)
            
            new_k = torch.cat([prev_k[:, :, 1:, :], k], dim=2) #(B, n_heads, window_size, head_dim)
            new_v = torch.cat([prev_v[:, :, 1:, :], v], dim=2) #(B, n_heads, window_size, head_dim)
           
            print(f"Frame prev_k: {prev_k.shape}, k: {k.shape}, new_k: {new_k.shape}")

            # 计算单帧 Attention (1 vs W)
            # q_vec: [B*H, 1, D], k_mat: [B*H, D, W]
            q_vec = q.reshape(B * self.num_heads, 1, head_dim) #T=1
            k_mat = new_k.transpose(2, 3).reshape(B * self.num_heads, head_dim, self.window_size)
            
            attn_weights = torch.bmm(q_vec, k_mat) * (head_dim ** -0.5)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1) # [B*H, 1, W]
            
            v_mat = new_v.reshape(B*self.num_heads, self.window_size, head_dim)

            context = torch.bmm(attn_weights, v_mat) # [B*H, 1, head_dim]
            
            next_state = torch.cat([new_k, new_v], dim=3)
            
        else:
            # --- 【训练模式：全量序列 + Causal Mask】 ---
            # q, k, v 形状均为 [B*H, D, 1, T] #[B, nheads, T, head_dim]

            q_seq = q.reshape(B * self.num_heads, T, head_dim) # [B*H, T, D]
            k_seq = k.reshape(B * self.num_heads, T, head_dim).transpose(1, 2)  # [B*H, D, T]
            v_seq = v.reshape(B * self.num_heads, T, head_dim) # [B*H, T, D]
            
            # 这里的每一帧 Q 都会与之前所有的 K 计算得分: [B*H, T, T]
            attn_scores = torch.bmm(q_seq, k_seq) * (head_dim ** -0.5)
            
            # Only Frame need mask
            # 应用 Causal Mask (下三角阵)
            # 屏蔽掉右上角（未来信息）
            mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            attn_scores.masked_fill_(mask, float('-inf'))
            
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1) # [B*H, T, T]
            context = torch.bmm(attn_weights, v_seq) # [B*H, T, D]

            next_state = None # 训练时不需要输出当前状态

        # 还原回 4D: [B, C, T, F]
        out = context.transpose(1, 2).reshape(B, C, T, F)
        out = self.proj_conv(out)

        out = out.transpose(2, 3) #Reverse the transpose: (B, inpC, F, T)
        
        return out, next_state

class Unified2DAttentionOnFrameAsym(nn.Module):
    """
    Unified2DAttentionOnFrameAsym: K, Q using same channels, but V using different channels
    """
    def __init__(self, in_channels, hid_channels=4, freq_bins=1, window_size=50, n_heads=4, act_type: str = "prelu",  norm_type: str = "LayerNormalization4D", need_streaming=True):
        super().__init__()
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.hid_channels=hid_channels
        self.window_size = window_size
        self.num_heads = n_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.need_streaming=need_streaming
        
        self.v_hid_channels=self.in_channels//self.num_heads #Value tensor dims
        self.total_hidd_channels = self.num_heads*(self.hid_channels*2+self.v_hid_channels) #kq hidden channels(2*hid_channels) + v hidden channels(v_hid_channels)
        
        self.qkv_conv = ATTConvActNorm(
                    in_chan=self.in_channels,
                    out_chan=self.total_hidd_channels,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.freq_bins,
                    is2d=True,
                )
        self.proj_conv = ATTConvActNorm(
            in_chan=self.in_channels,
            out_chan=self.in_channels,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.freq_bins,
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
        - training: T > 1, past_kv is set as None, and need_streaming=False
        - inferencing: T = 1, past_kv 为 [B*H, 2*D, F, W], need_streaming=True, W is self.window_size
        """
        B, inpC, F, T = x.shape #Make sure input shape is (B, inpC, F, T) for Frame path, and (B, inpC, T, F) for Frequency path
        x = x.transpose(-2, -1).contiguous()        
       
        qkv = self.qkv_conv(x) # [B, total_hidd_channels, T, F]
        C =self.hid_channels  #FIXME using new channels      

        qkv = qkv.transpose(1, 2) #[B, T, total_hidd_channels, F]
        qkv = qkv.reshape(B, T, self.num_heads, -1) #[B, T, nheads, (self.hid_channels*2+self.v_hid_channels)*F/nheads]=>[B, T, nheads, mix_dim]
        head_dim = self.hid_channels*F  #FIXME caluclate here
        v_head_dim = self.v_hid_channels*F
        # print(f"Unified2DAttentionOnFrameAsym need_streaming:{self.need_streaming}, x: {x.shape}, head_dim: {head_dim}, v_head_dim: {v_head_dim}")
       
        qkv = qkv.transpose(1, 2) #[B, nheads, T, mix_dim]
        q=qkv[:, :, :, 0:head_dim] #[B, nheads, T, head_dim]
        k=qkv[:, :, :, head_dim:2*head_dim] #[B, nheads, T, head_dim]
        v=qkv[:, :, :, 2*head_dim:] #[B, nheads, T, v_head_dim]
         
        if not self.training and past_kv is not None:       # self.need_streaming
            # print(f"past_kv: {past_kv.shape}")
            # if past_kv is None:                
            #     past_kv = torch.zeros(B, self.num_heads, self.window_size, head_dim+v_head_dim).to(x.device)
            
            prev_k, prev_v = past_kv[:, :, :, 0:head_dim], past_kv[:, :, :, head_dim:] #(B, n_heads, T, head_dim)
            
            new_k = torch.cat([prev_k[:, :, T:, :], k], dim=2) #(B, n_heads, window_size, head_dim)
            new_v = torch.cat([prev_v[:, :, T:, :], v], dim=2) #(B, n_heads, window_size, head_dim)            
            print(f"new_k: {new_k.shape}, new_v: {new_v.shape}, past_kv: {past_kv.shape}")

            q_vec = q.reshape(B * self.num_heads, T, head_dim) #T=1
            k_mat = new_k.transpose(2, 3).reshape(B * self.num_heads, head_dim, self.window_size)
            
            # print(f"Unified2DAttentionOnFrameAsym: head_dim: {head_dim}=>{head_dim ** -0.5}")
            attn_weights = torch.bmm(q_vec, k_mat) * 0.28867512941360474 #FIXME 0.22360679774997896 replace (head_dim ** -0.5) because npu compilation error
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1) # [B*H, 1, W]
            
            v_mat = new_v.reshape(B*self.num_heads, self.window_size, v_head_dim)

            context = torch.bmm(attn_weights, v_mat) # [B*H, T, v_head_dim]            
            next_state = torch.cat([new_k.detach(), new_v.detach()], dim=3)
            
        else:
            # on training/evaluate
            q_seq = q.reshape(B * self.num_heads, T, head_dim) # [B*H, T, D]
            k_seq = k.reshape(B * self.num_heads, T, head_dim).transpose(1, 2)  # [B*H, D, T]
            v_seq = v.reshape(B * self.num_heads, T, v_head_dim) # [B*H, T, D]
            
            attn_scores = torch.bmm(q_seq, k_seq) *  (head_dim ** -0.5)
            
            # Only Frame path need masking
            # Apply Causal Masking
            # if self.training and self.need_streaming:
            if self.need_streaming: #FIXME
                # mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
                # attn_scores.masked_fill_(mask, float('-inf'))
                mask = self.get_sliding_window_mask(T, self.window_size) #FIXME for fixed length of window attention, here using add(+)
                attn_scores = attn_scores+mask
            
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1) # [B*H, T, T]
            context = torch.bmm(attn_weights, v_seq) # [B*H, T, D]

            next_state = None #During training, no need saving next_state, because it uing whole clips (not chunks)

        
        # preparing back to 4D: [B*H, D, T]
        out = context.transpose(1, 2).reshape(B, inpC, F, T)       
        out = self.proj_conv(out)
        return out, next_state

class Unified2DAttentionOnFreqAsym(nn.Module):
    """
    Unified2DAttentionOnFreqAsym: K, Q using same channels, but V using different channels
    """
    def __init__(self, in_channels, hid_channels=4, freq_bins=1, window_size=50, n_heads=4, act_type: str = "prelu",  norm_type: str = "LayerNormalization4D", need_streaming=True):
        super().__init__()
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.hid_channels=hid_channels
        self.window_size = window_size
        self.num_heads = n_heads
        self.act_type = act_type
        self.norm_type = norm_type
        self.need_streaming=need_streaming
        
        self.v_hid_channels=self.in_channels//self.num_heads #Value tensor dims
        self.total_hidd_channels = self.num_heads*(self.hid_channels*2+self.v_hid_channels) #kq hidden channels(2*hid_channels) + v hidden channels(v_hid_channels)
        
        self.qkv_conv = ATTConvActNorm(
                    in_chan=self.in_channels,
                    out_chan=self.total_hidd_channels,
                    kernel_size=1,
                    act_type=self.act_type,
                    norm_type=self.norm_type,
                    n_freqs=self.freq_bins,
                    is2d=True,
                )
        self.proj_conv = ATTConvActNorm(
            in_chan=self.in_channels,
            out_chan=self.in_channels,
            kernel_size=1,
            act_type=self.act_type,
            norm_type=self.norm_type,
            n_freqs=self.freq_bins,
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
        - training: T > 1, past_kv is set as None, and need_streaming=False
        - inferencing: T = 1, past_kv 为 [B*H, 2*D, F, W], need_streaming=True, W is self.window_size
        """
        B, inpC, F, T = x.shape #Make sure input shape is (B, inpC, F, T) for Frame path, and (B, inpC, T, F) for Frequency path
        x = x.transpose(-2, -1).contiguous()        
       
        qkv = self.qkv_conv(x) # [B, total_hidd_channels, T, F]
        C =self.hid_channels  #FIXME using new channels      

        qkv = qkv.transpose(1, 2) #[B, T, total_hidd_channels, F]
        qkv = qkv.reshape(B, T, self.num_heads, -1) #[B, T, nheads, (self.hid_channels*2+self.v_hid_channels)*F/nheads]=>[B, T, nheads, mix_dim]
        head_dim = self.hid_channels*F  #FIXME caluclate here
        v_head_dim = self.v_hid_channels*F
        # print(f"Unified2DAttentionOnFrameAsym need_streaming:{self.need_streaming}, x: {x.shape}, head_dim: {head_dim}, v_head_dim: {v_head_dim}")
       
        qkv = qkv.transpose(1, 2) #[B, nheads, T, mix_dim]
        q=qkv[:, :, :, 0:head_dim] #[B, nheads, T, head_dim]
        k=qkv[:, :, :, head_dim:2*head_dim] #[B, nheads, T, head_dim]
        v=qkv[:, :, :, 2*head_dim:] #[B, nheads, T, v_head_dim]
         
        if not self.training and past_kv is not None: #Freq Always Not pass this path
            # if past_kv is None:                
            #     past_kv = torch.zeros(B, self.num_heads, self.window_size, head_dim+v_head_dim).to(x.device)
            
            prev_k, prev_v = past_kv[:, :, :, 0:head_dim], past_kv[:, :, :, head_dim:] #(B, n_heads, T, head_dim)
            
            new_k = torch.cat([prev_k[:, :, T:, :], k], dim=2) #(B, n_heads, window_size, head_dim)
            new_v = torch.cat([prev_v[:, :, T:, :], v], dim=2) #(B, n_heads, window_size, head_dim)            
           
            q_vec = q.reshape(B * self.num_heads, T, head_dim) #T=1
            k_mat = new_k.transpose(2, 3).reshape(B * self.num_heads, head_dim, self.window_size)
            
            attn_weights = torch.bmm(q_vec, k_mat) * (head_dim ** -0.5)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1) # [B*H, 1, W]
            
            v_mat = new_v.reshape(B*self.num_heads, self.window_size, v_head_dim)

            context = torch.bmm(attn_weights, v_mat) # [B*H, T, v_head_dim]            
            next_state = torch.cat([new_k.detach(), new_v.detach()], dim=3)
            
        else:
            # on training/evaluate
            q_seq = q.reshape(B * self.num_heads, T, head_dim) # [B*H, T, D]
            k_seq = k.reshape(B * self.num_heads, T, head_dim).transpose(1, 2)  # [B*H, D, T]
            v_seq = v.reshape(B * self.num_heads, T, v_head_dim) # [B*H, T, D]
            # print(f"Unified2DAttentionOnFreqAsym: head_dim: {head_dim}=>{head_dim ** -0.5}")
            attn_scores = torch.bmm(q_seq, k_seq) * 0.5 #FIXME for NPU compilation: replacing (head_dim ** -0.5)
            
            # Only Frame path need masking
            # Apply Causal Masking
            # if self.training and self.need_streaming:
            if self.need_streaming: #FIXME
                # mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
                # attn_scores.masked_fill_(mask, float('-inf'))
                mask = self.get_sliding_window_mask(T, self.window_size) #FIXME for fixed length of window attention, here using add(+)
                attn_scores = attn_scores+mask
            
            attn_weights = torch.nn.functional.softmax(attn_scores, dim=-1) # [B*H, T, T]
            context = torch.bmm(attn_weights, v_seq) # [B*H, T, D]

            next_state = None #During training, no need saving next_state, because it uing whole clips (not chunks)

        
        # preparing back to 4D: [B*H, D, T]
        out = context.transpose(1, 2).reshape(B, inpC, F, T)       
        out = self.proj_conv(out)
        return out, next_state
    
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
            FreqUConvBlock(out_channels, in_channels, upsampling_depth),
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

    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, model_T=True):
        super().__init__()
        self.proj_1x1 = PrjConv2dNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        #lookback=(3-1)*6 + (3-1)*2+ (3-1)*1=18
        #(receiptive width from big to small)
        # self.spp_dw0 = StatefulDilatedConv2dNorm(
        #         in_channels, in_channels, kSize=(1, 3), groups=in_channels, d=(1, 6), lookahead=0  #FIXME mayneed set to not zero for first dw0
        #     )
        # self.spp_dw1 = StatefulDilatedConv2dNorm(
        #         in_channels, in_channels, kSize=(1, 3), groups=in_channels, d=(1, 2), lookahead=0
        #     )
        # self.spp_dw2 = StatefulDilatedConv2dNorm(
        #         in_channels, in_channels, kSize=(1, 3), groups=in_channels, d=(1, 1), lookahead=0
        #     )
        
        #in reverse(receiptive width from small to big)
        #lookback=(3-1)*1 + (3-1)*2 + (3-1)*4=14
        self.spp_dw0 = StatefulDilatedConv2dNorm(
                in_channels, in_channels, kSize=(1, 3), groups=in_channels, d=(1, 1), lookahead=0  #FIXME mayneed set to not zero for first dw0
            )
        self.spp_dw1 = StatefulDilatedConv2dNorm(
                in_channels, in_channels, kSize=(1, 3), groups=in_channels, d=(1, 2), lookahead=0
            )
        self.spp_dw2 = StatefulDilatedConv2dNorm(
                in_channels, in_channels, kSize=(1, 3), groups=in_channels, d=(1, 4), lookahead=0
            )
        
        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(FrameInjectionMultiSum(in_channels, in_channels)) 

        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)

        self.globalatt = FrameMlp(in_channels, in_channels, drop=0.1)
        
        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            # self.last_layer.append(FrameInjectionMultiSum(in_channels, in_channels, 5)) #Initial Tiger
            self.last_layer.append(FrameInjectionMultiSum(in_channels, in_channels, 1))  #FIXME change kernel_size to 1 for reduce the passed State parameters

    def forward(self, x, state_0=None, state_1=None, state_2=None):
        """
        :param x: input feature map
        :return: transformed feature map
        """
        # print(f"StatefulTimeUConvBlock: x {x.shape}")
        residual = x.clone() # B, N, nband, T
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        x = self.proj_1x1(x)

        output0, new_state_0 = self.spp_dw0(x, state_0)
        output1, new_state_1 = self.spp_dw1(output0, state_1)
        output2, new_state_2 = self.spp_dw2(output1, state_2)
        # output3, new_state_3 = self.spp_dw3(output2, state_3)
        output=[output0, output1, output2]

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

        global_f = self.globalatt(global_f)  # [B, N, nBand, T]        
     
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
        
        return res_output + residual, new_state_0, new_state_1, new_state_2

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
        kv_window_size=18 #FIXME should be total lookback size=18?
    ):
        super().__init__()
        self.out_channels=out_channels
        self.in_channels=in_channels
        self.n_heads=n_heads
        self.nband = nband
        self.att_hid_chan=att_hid_chan
        self.kv_window_size=kv_window_size
        self.need_streaming=need_streaming

        print(f"")

        self.freq_path = nn.ModuleList([
            FreqUConvBlock(out_channels, in_channels, f_upsampling_depth),
            Unified2DAttentionOnFreqAsym(out_channels, hid_channels=att_hid_chan, freq_bins=1, window_size=self.kv_window_size, n_heads=n_heads, need_streaming=False),
            normalizations.get("LayerNormalization4D")((out_channels, 1))
        ])
        
        self.frame_path = nn.ModuleList([           
            StatefulTimeUConvBlock(out_channels, in_channels, t_upsampling_depth),
            Unified2DAttentionOnFrameAsym(out_channels, hid_channels=att_hid_chan, freq_bins=1, window_size=self.kv_window_size, n_heads=n_heads, need_streaming=need_streaming),
            normalizations.get("LayerNormalization4D")((out_channels, 1))
        ])
        
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def freq_time_interp_process(self, x, past_kv=None, state_0=None, state_1=None, state_2=None):
        #x: (B, N, nband, T)
        
        # Process Frequency Path
        residual_1 = x.clone()

        freq_fea = self.freq_path[0](x) # B, N, nband, T
        
        freq_fea = freq_fea.transpose(2, 3) #Transpose to (B, N, T, nBand)
        freq_fea, _ = self.freq_path[1](freq_fea) # B, N, T, nBand
        freq_fea = freq_fea.transpose(2, 3) #Back to (B, N, nBand, T)

        freq_fea = self.freq_path[2](freq_fea) # B, N, nband, T
       
        x2 = freq_fea + residual_1 #(B, N, nband, T) #NOTE cmj make NPU convertion failed
        # x = freq_fea+x
       
        # Process Frame Path
        residual_2 = x2.clone()

        frame_fea, new_state_0, new_state_1, new_state_2 = self.frame_path[0](x2, state_0, state_1, state_2) #(B, N, nband, T)
        frame_fea, new_kv = self.frame_path[1](frame_fea, past_kv=past_kv) # B, N, nband, T
        frame_fea = self.frame_path[2](frame_fea) # B, N, nband, T

        out = frame_fea + residual_2 #B, N, nband, T #Make NPU compilation failed

        return out, new_kv, new_state_0, new_state_1, new_state_2
    
    def forward(self, x, past_kvs=None, prev_states_0=None, prev_states_1=None, prev_states_2=None):
        # B, nband, N, T
        B, nband, N, T = x.shape
        x = x.permute(0, 2, 1, 3).contiguous() # B, N, nband, T
        mixture = x.clone()  #FIXME cmj

        state_dim = self.in_channels
        kv_dim = self.att_hid_chan*nband + (self.out_channels//self.n_heads)*nband #FIXME only Frame path need the kv_cache
        

        if past_kvs is not None and prev_states_0 is not None:
            # print(f"x: {x.shape}, past_kvs: {past_kvs.shape}, state_dim: {state_dim}, prev_states_0: {prev_states_0.shape}, prev_states_1: {prev_states_1.shape}, prev_states_2: {prev_states_2.shape}")
        
            prev_state_0=prev_states_0[:, 0:state_dim, :, :]
            prev_state_1=prev_states_1[:, 0:state_dim, :, :]
            prev_state_2=prev_states_2[:, 0:state_dim, :, :]

            #torch.zeros(B, self.num_heads, self.window_size, head_dim+v_head_dim)
            past_kv = past_kvs[:, :, :, 0:kv_dim]

            new_states_0=[]
            new_states_1=[]
            new_states_2=[]
            new_kvs=[]
            #Iter_0       
            x, new_kv, new_state_0, new_state_1, new_state_2 = self.freq_time_interp_process(x, past_kv=past_kv, state_0=prev_state_0, state_1=prev_state_1, state_2=prev_state_2) # B, N, nband, T
            
            new_states_0.append(new_state_0)
            new_states_1.append(new_state_1)
            new_states_2.append(new_state_2)
            new_kvs.append(new_kv)

            #Iter_1/2...  
            for i in range(1, self.iter):
                prev_state_0=prev_states_0[:, i*state_dim:(i+1)*state_dim, :, :]
                prev_state_1=prev_states_1[:, i*state_dim:(i+1)*state_dim, :, :]
                prev_state_2=prev_states_2[:, i*state_dim:(i+1)*state_dim, :, :]
                past_kv = past_kvs[:, :, :, i*kv_dim:(i+1)*kv_dim]

                x, new_kv, new_state_0, new_state_1, new_state_2 = self.freq_time_interp_process(self.concat_block(mixture + x), past_kv=past_kv, state_0=prev_state_0, state_1=prev_state_1, state_2=prev_state_2) # B, N, nband, T
                new_states_0.append(new_state_0)
                new_states_1.append(new_state_1)
                new_states_2.append(new_state_2)
                new_kvs.append(new_kv)
            
            new_states_0=torch.cat(new_states_0, dim=1)
            new_states_1=torch.cat(new_states_1, dim=1)
            new_states_2=torch.cat(new_states_2, dim=1)
            new_kvs = torch.cat(new_kvs, dim=-1)
            # print(f"after cat, new_stats_0: {new_states_0.shape}, new_states_1: {new_states_1.shape}, new_states_2: {new_states_2.shape}, new_kvs: {new_kvs.shape}")

        else:
            # print(f"x: {x.shape}, state_dim: {state_dim}")        
            prev_state_0=None
            prev_state_1=None
            prev_state_2=None
            past_kv=None

            #Iter_0   
            x, new_kv, new_state_0, new_state_1, new_state_2 = self.freq_time_interp_process(x, past_kv=past_kv, state_0=prev_state_0, state_1=prev_state_1, state_2=prev_state_2) # B, N, nband, T

            #Iter_1/2...  
            for i in range(1, self.iter):
                x, new_kv, new_state_0, new_state_1, new_state_2 = self.freq_time_interp_process(self.concat_block(mixture + x), past_kv=past_kv, state_0=prev_state_0, state_1=prev_state_1, state_2=prev_state_2) # B, N, nband, T

            new_states_0=None
            new_states_1=None
            new_states_2=None
            new_kvs=None            
             
        return x.permute(0, 2, 1, 3).contiguous(), new_kvs, new_states_0, new_states_1, new_states_2 # B, nband, N, T


class BNBlock(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, kernel_size=(1, 1), groups=1, eps=1e-8):
        super().__init__()        
        self.sgn = nn.LayerNorm(n_in_channels, eps=eps)
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
        pre_calc_bands=None

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
        
        self.separator = RecurrentKV(self.feature_dim, in_channels, self.nband, upsampling_depth, 3, att_n_head, att_hid_chan, att_kernel_size, att_stride, num_blocks, need_streaming, kv_window_size)
      
    def forward(self, subband_spec_RIs=None, past_kvs=None, prev_states_0=None, prev_states_1=None, prev_states_2=None):
        """
        Forward pass of the TIGER model.
        
        Args:
            subband_features: List of subband features of shape (B, N, T)
            
        Returns:
            sep_output: Separated output of shape (B, nband, N, T)
        """        
        # bsz, C, total_bands*2, T=subband_spec_RIs.shape      #total_bands=window_size//2+1=2048//2+1=1025 
        # if not self.training:
        #     print(f"===TIGER input spec: {subband_spec_RIs.shape}")

        band_feats=[]
        start_i=0
        for i, bw in enumerate(self.band_width):
            subband = subband_spec_RIs[:, :, start_i:start_i+bw*2, :] #(B, C, bw, T)            
            subband_feature = self.feature_extractors[i](subband) #(B, fea_dim, 1, T)
            band_feats.append(subband_feature)
            start_i = start_i + bw*2

        subband_features = torch.cat(band_feats, dim=2) #(B, fea_dim, nbands, T)
        subband_features = subband_features.transpose(1, 2) #(B, nbands, fea_dim, T)

        # separator
        batch_size, nband, feature_dim, T = subband_features.shape   #torch.Size([1, 2, 132, 5]) => (bsz, nband, fea_dim, T)   
        sep_output, new_kv, new_states_0, new_states_1,  new_states_2 = self.separator(subband_features, past_kvs=past_kvs, prev_states_0=prev_states_0, prev_states_1=prev_states_1, prev_states_2=prev_states_2)  # (B, nband, fea_dim, T) or (B, nband, N, T)
         
        # Apply masks
        masked_outputs=[]
        for i, bw in enumerate(self.band_width):
            subband_mask_enc = sep_output[:, i:i+1, :, :]
            subband_mask_enc=subband_mask_enc.permute(0, 2, 3, 1) #(B, fea_dim, T, 1)            
            subband_mask_dec = self.mask_decoders[i](subband_mask_enc) #(B, 2*2*num_sources, bw, T)           
            subband_mask_dec = subband_mask_dec.view(batch_size, 4*self.num_output, bw, -1)
            masked_outputs.append(subband_mask_dec)         

        band_masked_output=torch.concat(masked_outputs, dim=-2) #(B, 12, total_bands, T)
        
        return band_masked_output, new_kv, new_states_0, new_states_1,  new_states_2
    
    def get_model_args(self):
        model_args = {"n_sample_rate": 2}
        return model_args