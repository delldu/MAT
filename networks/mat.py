import numpy as np
import math
import sys
sys.path.insert(0, '../')

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple

from torch_utils import misc
from torch_utils import persistence
from networks.basic_module import (
    MappingNetFC,
    LinearFC,
    FullyConnectedLayer, 
    Conv2dLayer, 
    MappingNet, 
    StyleConv, 
    ToRGB, 
    ToRGBWithSkip,
    get_style_code,
)
import todos
import pdb

@misc.profiled_function
def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2 ** stage]


@persistence.persistent_class
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features)
        self.fc2 = LinearFC(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


@misc.profiled_function
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


@misc.profiled_function
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

# xxxx_debug
@persistence.persistent_class
class Conv2dLayerPartial(nn.Module):
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
        # in_channels = 4
        # out_channels = 180
        # kernel_size = 3

        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, up, down, resample_filter)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0
        # self.weight_maskUpdater.size() -- [1, 1, 3, 3]
        #  self.slide_winsize -- 9

    def forward(self, x, mask=None):
        # tensor [x] size: [1, 4, 512, 512], min: -1.0, max: 1.0, mean: -0.063706
        if mask is not None:
            # tensor [mask] size: [1, 1, 512, 512], min: 0.0, max: 1.0, mean: 0.247547
            # ==> pdb.set_trace()
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            # ==> pdb.set_trace()
            x = self.conv(x)
            return x, None

# xxxx_debug
@persistence.persistent_class
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    """

    def __init__(self, dim, window_size, num_heads, down_ratio=1, attn_drop=0., proj_drop=0.):

        super().__init__()
        # self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = LinearFC(in_features=dim, out_features=dim)
        self.k = LinearFC(in_features=dim, out_features=dim)
        self.v = LinearFC(in_features=dim, out_features=dim)
        self.proj = LinearFC(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = self.q(norm_x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(norm_x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            pdb.set_trace()

        if mask_windows is not None:
            # ==> pdb.set_trace()
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(attn_mask_windows == 0, float(-100.0)).masked_fill(
                attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(torch.sum(mask_windows, dim=1, keepdim=True), 0, 1).repeat(1, N, 1)
        else:
            # ==> pdb.set_trace()
            pass

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows

# xxxx_debug
@persistence.persistent_class
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    """

    def __init__(self, dim, input_resolution, num_heads, down_ratio=1, window_size=7, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # dim = 180
        # input_resolution = [64, 64]
        # num_heads = 6
        # down_ratio = 1
        # window_size = 8
        # shift_size = 0
        # mlp_ratio = 2.0
        # drop = 0.0
        # attn_drop = 0.0
        # drop_path = 0.0
        # act_layer = <class 'torch.nn.modules.activation.GELU'>
        # norm_layer = <class 'torch.nn.modules.normalization.LayerNorm'>

        # self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        # self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # ==> pdb.set_trace()
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    down_ratio=down_ratio, attn_drop=attn_drop,
                                    proj_drop=drop)

        self.fuse = FullyConnectedLayer(in_features=dim * 2, out_features=dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # ==> pdb.set_trace()
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            # ==> pdb.set_trace()
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask) # None ???

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if mask is not None:
            # ==> pdb.set_trace()
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None
            # ==> pdb.set_trace()

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            # ==> pdb.set_trace()
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device))  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)
        else:
            # pdb.set_trace()
            pass

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            if mask is not None:
                mask = torch.roll(shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)
        else:
            # ==> pdb.set_trace()
            pass

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask

# xxxx_debug
@persistence.persistent_class
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            # ==> pdb.set_trace()
            mask = token2feature(mask, x_size)
        else:
            # ==> pdb.set_trace()
            pass

        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        else:
            pdb.set_trace()

        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        else:
            pass
            # pdb.set_trace()

        return x, x_size, mask

# xxxx_debug
@persistence.persistent_class
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        else:
            # ==> pdb.set_trace()
            pass

        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        else:
            # ==> pdb.set_trace()
            pass

        return x, x_size, mask


# xxxx_debug
@persistence.persistent_class
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, down_ratio=1,
                 mlp_ratio=2., drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        # dim = 180
        # input_resolution = [64, 64]
        # depth = 2
        # num_heads = 6
        # window_size = 8
        # down_ratio = 1
        # mlp_ratio = 2.0
        # drop = 0.0
        # attn_drop = 0.0
        # drop_path = [0.0, 0.007692307699471712]
        # norm_layer = <class 'torch.nn.modules.normalization.LayerNorm'>
        # downsample = None


        # pdb.set_trace()
        self.input_resolution = input_resolution

        # patch merging layer
        if downsample is not None:
            # ==> pdb.set_trace()
            self.downsample = downsample
        else:
            self.downsample = None
            # ==> pdb.set_trace()

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, down_ratio=down_ratio, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)]) # depth === 2

        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x, x_size, mask=None):
        # tensor [x] size: [1, 4096, 180], min: -136.84729, max: 598.483154, mean: -0.283396
        # x_size === (64, 64)
        # tensor [mask] size: [1, 4096, 1], min: 0.0, max: 1.0, mean: 0.663086

        if self.downsample is not None: # True
            x, x_size, mask = self.downsample(x, x_size, mask)
        else:
            # ==> pdb.set_trace()
            pass

        identity = x
        for blk in self.blocks:
            x, mask = blk(x, x_size, mask)
        if mask is not None:
            # ==> 
            mask = token2feature(mask, x_size)
        else:
            # ==> pdb.set_trace()
            pass

        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        else:
            # ==> pdb.set_trace()
            pass

        return x, x_size, mask

#----------------------------------------------------------------------------
@persistence.persistent_class
class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=1,
                                )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x

@persistence.persistent_class
class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 down=2,
                                 )
        self.conv1 = Conv2dLayer(in_channels=out_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x


@persistence.persistent_class
class Encoder(nn.Module):
    def __init__(self, 
        res_log2 = 9, 
        img_channels = 3, 
        patch_size=5, 
        channels=16, 
        drop_path_rate=0.1):
        super().__init__()
        # res_log2 = 9
        # img_channels = 3
        # patch_size = 5
        # channels = 16
        # drop_path_rate = 0.1

        self.resolution = []
        # range(res_log2, 3, -1) -- [9, 8, 7, 6, 5, 4]
        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2 ** i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i))
            else:
                block = ConvBlockDown(nf(i+1), nf(i))
            setattr(self, 'EncConv_Block_%dx%d' % (res, res), block)
        # self.resolution -- [512, 256, 128, 64, 32, 16]

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res)) # [9, 8, 7, 6, 5, 4]
            x = getattr(self, 'EncConv_Block_%dx%d' % (res, res))(x)
            out[res_log2] = x # [9, 8, 7, 6, 5, 4]

        return out


@persistence.persistent_class
class ToStyle(nn.Module):
    def __init__(self, 
        in_channels = 512, 
        out_channels = 1024, 
    ):
        super().__init__()
        # in_channels = 512
        # out_channels = 1024
        self.conv = nn.Sequential(
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, down=2),
                Conv2dLayer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, down=2),
                )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = FullyConnectedLayer(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        # tensor [x] size: [1, 512, 16, 16], min: -10.228478, max: 40.67131, mean: 0.077602
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        # tensor [x] size: [1, 1024], min: -0.370284, max: 0.784699, mean: 0.005559
        return x

# xxxx_debug
@persistence.persistent_class
class DecBlockFirstV2(nn.Module):
    def __init__(self, res, in_channels, out_channels, style_dim, img_channels):
        super().__init__()
        # res = 4
        # in_channels = 512
        # out_channels = 512
        # style_dim = 1536
        # img_channels = 3
        self.res = res
        self.conv0 = Conv2dLayer(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                )
        self.conv1 = StyleConv(in_channels=in_channels,
                              out_channels=out_channels,
                              style_dim=style_dim,
                              resolution=2**res,
                              kernel_size=3,
                              use_noise=True,
                              )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           )

    def forward(self, x, ws, gs, e_features, noise_mode='random'):
        # tensor [x] size: [1, 512, 16, 16], min: -10.228478, max: 40.67131, mean: 0.077602
        # tensor [ws] size: [1, 12, 512], min: -0.035399, max: 0.030863, mean: -0.006447
        # tensor [gs] size: [1, 1024], min: -0.370284, max: 0.784699, mean: 0.005559
        # e_features is dict:
        #     tensor [9] size: [1, 64, 512, 512], min: -2.689897, max: 8.129209, mean: -0.011228
        #     tensor [8] size: [1, 128, 256, 256], min: -7.99602, max: 19.742258, mean: 0.121683
        #     tensor [7] size: [1, 256, 128, 128], min: -14.827822, max: 47.724689, mean: 0.104518
        #     tensor [6] size: [1, 512, 64, 64], min: -15.623693, max: 53.609097, mean: 0.082812
        #     tensor [5] size: [1, 512, 32, 32], min: -15.83503, max: 75.104073, mean: -0.047926
        #     tensor [4] size: [1, 512, 16, 16], min: -10.228478, max: 40.67131, mean: 0.077602
        # noise_mode -- "const"
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + e_features[self.res] # self.res === 4
        style = get_style_code(ws[:, 0], gs) # ws[:, 0].size() -- [1, 512]
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, 1], gs)
        img = self.toRGB(x, style)

        # tensor [x] size: [1, 512, 16, 16], min: -14.745468, max: 52.303444, mean: -0.170848
        # tensor [img] size: [1, 3, 16, 16], min: -0.003557, max: 0.002705, mean: -0.000736
        return x, img

#----------------------------------------------------------------------------
# xxxx_debug
@persistence.persistent_class
class DecBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, style_dim, img_channels):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res
        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=True,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=True,
                               )
        self.toRGB = ToRGBWithSkip(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           )

    def forward(self, x, img, ws, gs, e_features, noise_mode='random'):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + e_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style, noise_mode=noise_mode)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)
        img = self.toRGB(x, style, skip=img)

        return x, img

# xxxx_debug
@persistence.persistent_class
class Decoder(nn.Module):
    def __init__(self, 
        res_log2=9, 
        # activation='lrelu', 
        style_dim=1536, 
        img_channels=3,
    ):
        super().__init__()
        self.Dec_16x16 = DecBlockFirstV2(4, nf(4), nf(4), style_dim, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res),
                    DecBlock(res, nf(res - 1), nf(res), style_dim, img_channels))
        self.res_log2 = res_log2 # 9

    def forward(self, x, ws, gs, e_features, noise_mode='random'):
        print("Decoder forward noise_mode: ", noise_mode)

        x, img = self.Dec_16x16(x, ws, gs, e_features, noise_mode=noise_mode)
        for res in range(5, self.res_log2 + 1):
            block = getattr(self, 'Dec_%dx%d' % (2 ** res, 2 ** res))
            x, img = block(x, img, ws, gs, e_features, noise_mode=noise_mode)

        return img


@persistence.persistent_class
class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, style_dim, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(in_channels=in_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               up=2,
                               use_noise=False,
                               )
        self.conv1 = StyleConv(in_channels=out_channels,
                               out_channels=out_channels,
                               style_dim=style_dim,
                               resolution=2**res,
                               kernel_size=3,
                               use_noise=False,
                               )
        self.toRGB = ToRGB(in_channels=out_channels,
                           out_channels=img_channels,
                           style_dim=style_dim,
                           )

    def forward(self, x, img, style, skip, noise_mode='random'):
        x = self.conv0(x, style, noise_mode=noise_mode)
        x = x + skip
        x = self.conv1(x, style, noise_mode=noise_mode)
        img = self.toRGB(x, style, skip=img)

        return x, img


@persistence.persistent_class
class FirstStage(nn.Module):
    def __init__(self, 
        img_channels = 3, 
        img_resolution=512, 
        dim=180, 
        w_dim=512, 
        # demodulate=True, 
    ):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(in_channels=img_channels+1, out_channels=dim, kernel_size=3)
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res)) # down_time === 3
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, down=2)
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None # ==> ratios[i] === 1
            self.tran.append(
                BasicLayer(dim=dim, input_resolution=[res, res], depth=depth, num_heads=num_heads,
                           window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                           downsample=merge)
            )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(Conv2dLayer(in_channels=dim, out_channels=dim, kernel_size=3, down=2))
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)
        self.to_style = FullyConnectedLayer(in_features=dim, out_features=dim*2)
        self.ws_style = FullyConnectedLayer(in_features=w_dim, out_features=dim)
        self.to_square = FullyConnectedLayer(in_features=dim, out_features=16*16)

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # down_time == 3, from 64 to input size
            res = res * 2
            self.dec_conv.append(DecStyleBlock(res, dim, dim, style_dim, img_channels))

    def forward(self, images_in, masks_in, ws, noise_mode='random'):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1: # # xxxx_debug
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2 # len(self.tran) == 5 ==> mid === 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid: # # xxxx_debug
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, None)
                x = x + skips[mid - i]
            else: # ==> i === mid
                x, x_size, mask = block(x, x_size, None)

                mul_map = torch.ones_like(x) * 0.5
                mul_map = F.dropout(mul_map, training=True)
                ws = self.ws_style(ws[:, -1])
                add_n = self.to_square(ws).unsqueeze(1)
                add_n = F.interpolate(add_n, size=x.size(1), mode='linear', align_corners=False).squeeze(1).unsqueeze(-1)
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(self.down_conv(token2feature(x, x_size)).flatten(start_dim=1))
                style = torch.cat([gs, ws], dim=1)

        x = token2feature(x, x_size).contiguous()
        img = None
        for i, block in enumerate(self.dec_conv): # len(self.dec_conv) === 3
            x, img = block(x, img, style, skips[len(self.dec_conv)-i-1], noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img


@persistence.persistent_class
class SynthesisNet(nn.Module):
    def __init__(self,
                 w_dim = 512,                     # Intermediate latent (W) dimensionality.
                 img_resolution = 512,            # Output image resolution.
                 img_channels   = 3,        # Number of color channels.
                 channel_base   = 32768,    # Overall multiplier for the number of channels.
                 channel_decay  = 1.0,
                 channel_max    = 512,      # Maximum number of channels in any layer.
                 ):
        super().__init__()
        # w_dim = 512
        # img_resolution = 512
        # img_channels = 3
        # channel_base = 32768
        # channel_decay = 1.0
        # channel_max = 512
        resolution_log2 = int(np.log2(img_resolution)) # ==> 9
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2 # ==> 12

        # first stage
        self.first_stage = FirstStage(img_channels, img_resolution=img_resolution, w_dim=w_dim)

        # second stage
        self.enc = Encoder(resolution_log2, img_channels, patch_size=5, channels=16)
        self.to_square = FullyConnectedLayer(in_features=w_dim, out_features=16*16)
        self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2)
        style_dim = w_dim + nf(2) * 2 # ==> 1536
        self.dec = Decoder(resolution_log2, style_dim, img_channels)


    def forward(self, images_in, masks_in, ws, noise_mode='random'):
        out_stg1 = self.first_stage(images_in, masks_in, ws, noise_mode=noise_mode)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        e_features = self.enc(x)

        fea_16 = e_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(add_n, size=fea_16.size()[-2:], mode='bilinear', align_corners=False)
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        e_features[4] = fea_16

        # style
        gs = self.to_style(fea_16)

        # decoder
        img = self.dec(fea_16, ws, gs, e_features, noise_mode=noise_mode)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        # tensor [x] size: [1, 7, 512, 512], min: -1.121117, max: 1.248556, mean: -0.007224
        # tensor [img] size: [1, 3, 512, 512], min: -1.124015, max: 1.043849, mean: 0.075933
        return img


@persistence.persistent_class
class Generator(nn.Module):
    def __init__(self,
                 z_dim,                  # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,                  # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,                  # Intermediate latent (W) dimensionality.
                 img_resolution,         # resolution of generated image
                 img_channels,           # Number of input color channels.
                 ):
        super().__init__()
        # z_dim = 512
        # c_dim = 0
        # w_dim = 512
        # img_resolution = 512
        # img_channels = 3

        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim

        self.synthesis = SynthesisNet(w_dim=w_dim,
                                      img_resolution=img_resolution,
                                      img_channels=img_channels)
        self.mapping = MappingNet(z_dim=z_dim,
                                  c_dim=c_dim,
                                  w_dim=w_dim,
                                  num_ws=self.synthesis.num_layers)

    def forward(self, images_in, masks_in, z, noise_mode='random'):
        todos.debug.output_var("images_in", images_in)
        todos.debug.output_var("masks_in", masks_in)
        todos.debug.output_var("z", z)
        # print("c", c)
        # todos.debug.output_var("truncation_cutoff", truncation_cutoff)
        # todos.debug.output_var("skip_w_avg_update", skip_w_avg_update)
        # todos.debug.output_var("noise_mode", noise_mode)

        # ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
        #                   skip_w_avg_update=skip_w_avg_update)
        ws = self.mapping(z)

        # todos.debug.output_var("ws", ws)
        # tensor [ws] size: [1, 12, 512], min: -0.035399, max: 0.030863, mean: -0.006447

        img = self.synthesis(images_in, masks_in, ws, noise_mode=noise_mode)
        return img
