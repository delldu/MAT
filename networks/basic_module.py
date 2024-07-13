import sys
sys.path.insert(0, '../')
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
import todos
import pdb
#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(nn.Module):
    def __init__(self,
                 in_features = 180,          # Number of input features.
                 out_features = 180,         # Number of output features.
                 bias            = True,     # Apply additive bias before the activation function?
                 activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
                 lr_multiplier   = 1,        # Learning rate multiplier.
                 bias_init       = 0,        # Initial value for the additive bias.
                 ):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.activation = activation

        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None and self.bias_gain != 1: # False for self.bias_gain === 1, True for self.bias_gain === 0.01
            # ==> pdb.set_trace()
            b = b * self.bias_gain
        else:
            # ==> pdb.set_trace()
            pass

        if self.activation == 'linear' and b is not None:
            # ==> pdb.set_trace() for self.activation === 'linear'
            # out = torch.addmm(b.unsqueeze(0), x, w.t())
            x = x.matmul(w.t())
            out = x + b.reshape([-1 if i == x.ndim-1 else 1 for i in range(x.ndim)])
        else:
            # ==> pdb.set_trace()
            # self.activation -- 'lrelu'
            x = x.matmul(w.t())
            out = bias_act.bias_act(x, b, act=self.activation, dim=x.ndim-1)
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
                 ):
        super().__init__()

        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
            
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs['lrelu'].def_gain

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        x = conv2d_resample.conv2d_resample(x=x, w=w, f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding)

        act_gain = self.act_gain * gain

        out = bias_act.bias_act(x, self.bias, act='lrelu', gain=act_gain)
        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class ModulatedConv2d(nn.Module):
    def __init__(self,
                 in_channels,                   # Number of input channels.
                 out_channels,                  # Number of output channels.
                 kernel_size,                   # Width and height of the convolution kernel.
                 style_dim,                     # dimension of the style code
                 demodulate=True,               # perfrom demodulation
                 up=1,                          # Integer upsampling factor.
                 down=1,                        # Integer downsampling factor.
                 resample_filter=[1,3,3,1],  # Low-pass filter to apply when resampling activations.
                 ):
        super().__init__()
        self.demodulate = demodulate # True | False

        self.weight = torch.nn.Parameter(torch.randn([1, out_channels, in_channels, kernel_size, kernel_size]))
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        self.affine = FullyConnectedLayer(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style

        if self.demodulate: # True | False
            # ==> pdb.set_trace()
            decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
            weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)
        else:
            # ==> pdb.set_trace()
            pass

        weight = weight.view(batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, batch * in_channels, height, width)
        x = conv2d_resample.conv2d_resample(x=x, w=weight, f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding, groups=batch)
        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class StyleConv(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        style_dim,                      # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input? True or False
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
    ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=True,
                                    up=up,
                                    resample_filter=resample_filter)

        self.use_noise = use_noise
        self.resolution = resolution
        if use_noise: # True | False
            # ==> pdb.set_trace()
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        else:
            # ==> pdb.set_trace()
            pass

        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.act_gain = bias_act.activation_funcs['lrelu'].def_gain


    def forward(self, x, style, noise_mode='random', gain=1):
        x = self.conv(x, style)

        assert noise_mode in ['random', 'const', 'none']

        if self.use_noise: # True | False
            if noise_mode == 'random':
                xh, xw = x.size()[-2:]
                noise = torch.randn([x.shape[0], 1, xh, xw], device=x.device) \
                        * self.noise_strength
            if noise_mode == 'const':
                noise = self.noise_const * self.noise_strength
            x = x + noise
        else:
            # ==> pdb.set_trace()
            pass

        act_gain = self.act_gain * gain
        out = bias_act.bias_act(x, self.bias, act='lrelu', gain=act_gain)

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class ToRGB(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 style_dim,
                 kernel_size=1,
                 resample_filter=[1,3,3,1],
                ):
        super().__init__()

        self.conv = ModulatedConv2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size,
                                    style_dim=style_dim,
                                    demodulate=False,
                                    resample_filter=resample_filter)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        #  self.resample_filter.size() -- [4, 4]

    def forward(self, x, style, skip=None):
        x = self.conv(x, style)
        out = bias_act.bias_act(x, self.bias)

        if skip is not None:
            # ==> pdb.set_trace()
            if skip.shape != out.shape:
                # skip.shape, out.shape -- [1, 3, 128, 128], [1, 3, 256, 256]
                skip = upfirdn2d.upsample2d(skip, self.resample_filter)
            out = out + skip
        else:
            # ==> pdb.set_trace()
            pass

        return out

#----------------------------------------------------------------------------

@misc.profiled_function
def get_style_code(a, b):
    return torch.cat([a, b], dim=1)

#----------------------------------------------------------------------------
@persistence.persistent_class
class MappingNet(torch.nn.Module):
    def __init__(self,
        z_dim = 512,                # Input latent (Z) dimensionality, 0 = no latent.
        c_dim = 0,                  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim = 512,                # Intermediate latent (W) dimensionality.
        num_ws = 12,                # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = 0,        # Label embedding dimensionality, None = same as w_dim.
        layer_features  = 512,      # Number of intermediate features in the mapping layers, None = same as w_dim.
    ):
        super().__init__()
        # z_dim = 512
        # c_dim = 0
        # w_dim = 512
        # num_ws = 12
        # num_layers = 8
        # embed_features = 0
        # layer_features = 512

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.num_ws = num_ws
        self.num_layers = num_layers

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        # ==> embed_features === 0

        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        # features_list --- [512, 512, 512, 512, 512, 512, 512, 512, 512]

        for idx in range(num_layers): # num_layers -- 8
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation='lrelu', lr_multiplier=0.01)
            setattr(self, f'fc{idx}', layer)

        self.register_buffer('w_avg', torch.zeros([w_dim]))
        # self.w_avg.size() -- [512]

    def forward(self, z):
        # tensor [z] size: [1, 512], min: -2.510093, max: 3.06608, mean: 0.064815

        x = normalize_2nd_moment(z.to(torch.float32))

        # Main layers.
        for idx in range(self.num_layers): # self.num_layers === 8
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # tensor [x] size: [1, 12, 512], min: -0.035399, max: 0.030863, mean: -0.006447
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class DisFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                )
        pdb.set_trace()

    def forward(self, x):
        return self.conv(x)

#----------------------------------------------------------------------------

@persistence.persistent_class
class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=3,
                                 )
        self.conv1 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 down=2,
                                 )
        self.skip = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                down=2,
                                bias=False,
                             )
        pdb.set_trace()

    def forward(self, x):
        skip = self.skip(x, gain=np.sqrt(0.5))
        x = self.conv0(x)
        x = self.conv1(x, gain=np.sqrt(0.5))
        out = skip + x

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels
        pdb.set_trace()

    def forward(self, x):
        pdb.set_trace()

        N, C, H, W = x.shape
        with misc.suppress_tracer_warnings():  # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size),
                          torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H,
                      W)  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.

        pdb.set_trace()
        return x

