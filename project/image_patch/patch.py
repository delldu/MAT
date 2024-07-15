import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from timm.models.layers import to_2tuple

# ----------------------------------------------------------------------------
from typing import Tuple
import todos

import pdb


def nf(stage, channel_base=32768, channel_decay=1.0, channel_max=512):
    NF = {512: 64, 256: 128, 128: 256, 64: 512, 32: 512, 16: 512, 8: 512, 4: 512}
    return NF[2**stage]


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = LReLuFC(in_features=in_features, out_features=hidden_features)
        self.fc2 = LinearFC(in_features=hidden_features, out_features=in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def window_partition(x, window_size: int):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    B = windows.shape[0] // (H * W // window_size // window_size)
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# xxxx_debug
class Conv2dLayerPartial(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
    ):
        super().__init__()
        # in_channels = 4
        # out_channels = 180
        # kernel_size = 3
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.up = up
        self.down = down

        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, up, down)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        # self.weight_maskUpdater.size() -- [1, 1, 3, 3]
        self.slide_winsize = kernel_size**2
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0
        #  self.slide_winsize -- 9

    def forward(self, x, mask=None):
        # tensor [x] size: [1, 4, 512, 512], min: -1.0, max: 1.0, mean: -0.063706
        if mask is not None:
            # tensor [mask] size: [1, 1, 512, 512], min: 0.0, max: 1.0, mean: 0.247547
            # ==> pdb.set_trace()
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(
                    mask,
                    self.weight_maskUpdater,
                    bias=None,
                    stride=self.down,
                    padding=self.padding,
                )
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

    def __repr__(self):
        s = f"Conv2dLayerPartial(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size},up={self.up}, down={self.down})"
        return s


# xxxx_debug
class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias."""

    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

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
        q = (
            self.q(norm_x)
            .reshape(B_, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(norm_x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 3, 1)
        )
        v = (
            self.v(x)
            .view(B_, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        else:
            pdb.set_trace()

        if mask_windows is not None:
            # ==> pdb.set_trace()
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(
                attn_mask_windows == 0, float(-100.0)
            ).masked_fill(attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(
                    torch.sum(mask_windows, dim=1, keepdim=True), 0, 1
                ).repeat(1, N, 1)
        else:
            # ==> pdb.set_trace()
            pass

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows

    def __repr__(self):
        s = f"WindowAttention(dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads})"
        return s


class SwinTransBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        # dim = 180
        # input_resolution = [64, 64]
        # num_heads = 6
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # ==> pdb.set_trace()
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads
        )
        self.fuse = LReLuFC(in_features=dim * 2, out_features=dim)
        self.mlp = MLP(in_features=dim, hidden_features=dim * 2)

        assert (self.shift_size == 0)

        # shift_size = 0, window_size=8
        # shift_size = 4, window_size=8
        # shift_size = 0, window_size=16
        # shift_size = 8, window_size=16

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

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
        else:
            # ==> pdb.set_trace()
            pass

        # no cyclic shift
        shifted_x = x
        if mask is not None:
            shifted_mask = mask
        else:
            # ==> pdb.set_trace()
            pass

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        if mask is not None:
            # ==> pdb.set_trace()
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None
            # ==> pdb.set_trace()

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, mask_windows = self.attn(
            x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device)
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)
        else:
            # pdb.set_trace()
            pass

        # no cyclic shift
        x = shifted_x
        if mask is not None:
            mask = shifted_mask
        else:
            # ==> pdb.set_trace()
            pass

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

    def __repr__(self):
        s = f"SwinTransBlock(dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size})"
        return s


class SwinTransBlockWithShift(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0):
        super().__init__()
        # dim = 180
        # input_resolution = [64, 64]
        # num_heads = 6
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads
        )
        self.fuse = LReLuFC(in_features=dim * 2, out_features=dim)
        self.mlp = MLP(in_features=dim, hidden_features=dim * 2)

        assert (self.shift_size > 0)

        # shift_size = 0, window_size=8
        # shift_size = 4, window_size=8
        # shift_size = 0, window_size=16
        # shift_size = 8, window_size=16

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        # slice(start, stop, step)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

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
        else:
            # ==> pdb.set_trace()
            pass

        # cyclic shift
        shifted_x = torch.roll(
            x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
        )
        if mask is not None:
            shifted_mask = torch.roll(
                mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            # =>pdb.set_trace()
            pass

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C
        if mask is not None:
            # ==> pdb.set_trace()
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None
            # ==> pdb.set_trace()

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows, mask_windows = self.attn(
            x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device)
        )  # nW*B, window_size*window_size, C

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
        x = torch.roll(
            shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
        )
        if mask is not None:
            mask = torch.roll(
                shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            # ==> pdb.set_trace()
            pass

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

    def __repr__(self):
        s = f"SwinTransBlockWithShift(dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, window_size={self.window_size}, shift_size={self.shift_size})"
        return s



# xxxx_debug
class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down = down
        # self.down === 2

        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            down=down,
        )

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            # ==> pdb.set_trace()
            mask = token2feature(mask, x_size)
        else:
            # ==> pdb.set_trace()
            pass

        x, mask = self.conv(x, mask)
        ratio = 1 / self.down  # self.down === 2
        x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))

        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        else:
            pass
            # pdb.set_trace()

        return x, x_size, mask

    def __repr__(self):
        s = f"PatchMerging(in_channels={self.in_channels}, out_channels={self.out_channels}, down={self.down})"
        return s


class PatchIdentity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, x_size, mask):
        return x, x_size, mask


# xxxx_debug
class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up  # up === 2

        self.conv = Conv2dLayerPartial(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            up=up,
        )

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        else:
            # ==> pdb.set_trace()
            pass

        x, mask = self.conv(x, mask)
        x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        else:
            # ==> pdb.set_trace()
            pass

        return x, x_size, mask

    def __repr__(self):
        s = f"PatchUpsampling(in_channels={self.in_channels}, out_channels={self.out_channels}, up={self.up})"
        return s


# xxxx_debug
class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage."""
    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, downsample
    ):
        super().__init__()
        # dim = 180
        # input_resolution = [64, 64]
        # depth = 2
        # num_heads = 6
        # window_size = 8

        self.input_resolution = input_resolution
        self.downsample = downsample

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth): # depth === 2
            if i % 2 == 0 or min(input_resolution) <= window_size:
                b = SwinTransBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    # shift_size=0,
                )
            else:
                b = SwinTransBlockWithShift(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                )

            self.blocks.append(b)
        
        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x, x_size, mask=None):
        # tensor [x] size: [1, 4096, 180], min: -136.84729, max: 598.483154, mean: -0.283396
        # x_size === (64, 64)
        # mask === None

        x, x_size, mask = self.downsample(x, x_size, mask)

        identity = x
        for blk in self.blocks:
            x, mask = blk(x, x_size, mask)

        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity

        return x, x_size, mask


class BasicLayerWithMask(nn.Module):
    """A basic Swin Transformer layer for one stage."""

    def __init__(
        self, dim, input_resolution, depth, num_heads, window_size, downsample
    ):
        super().__init__()
        # dim = 180
        # input_resolution = [64, 64]
        # depth = 2
        # num_heads = 6
        # window_size = 8

        self.input_resolution = input_resolution
        self.downsample = downsample

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth): # depth === 2
            if i % 2 == 0 or min(input_resolution) <= window_size:
                b = SwinTransBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    # shift_size=0,
                )
            else:
                b = SwinTransBlockWithShift(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=window_size // 2,
                )

            self.blocks.append(b)


        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3)

    def forward(self, x, x_size, mask):
        # tensor [x] size: [1, 4096, 180], min: -136.84729, max: 598.483154, mean: -0.283396
        # x_size === (64, 64)
        # tensor [mask] size: [1, 4096, 1], min: 0.0, max: 1.0, mean: 0.663086

        x, x_size, mask = self.downsample(x, x_size, mask)

        identity = x
        for blk in self.blocks:
            x, mask = blk(x, x_size, mask)

        mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        mask = feature2token(mask)

        return x, x_size, mask


# ----------------------------------------------------------------------------
class EncFromRGB(nn.Module):
    def __init__(self, in_channels, out_channels):  # res = 2, ..., resolution_log2
        super().__init__()
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        return x


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):  # res = 2, ..., resolution_log
        super().__init__()

        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            down=2,
        )
        self.conv1 = Conv2dLayer(
            in_channels=out_channels,
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


class Encoder(nn.Module):
    def __init__(self, res_log2=9, img_channels=3, patch_size=5, channels=16):
        super().__init__()
        # res_log2 = 9
        # img_channels = 3
        # patch_size = 5
        # channels = 16

        self.resolution = []
        # range(res_log2, 3, -1) -- [9, 8, 7, 6, 5, 4]
        for idx, i in enumerate(range(res_log2, 3, -1)):  # from input size to 16x16
            res = 2**i
            self.resolution.append(res)
            if i == res_log2:
                block = EncFromRGB(img_channels * 2 + 1, nf(i))
            else:
                block = ConvBlockDown(nf(i + 1), nf(i))
            setattr(self, "EncConv_Block_%dx%d" % (res, res), block)
        # self.resolution -- [512, 256, 128, 64, 32, 16]

    def forward(self, x):
        out = {}
        for res in self.resolution:
            res_log2 = int(np.log2(res))  # [9, 8, 7, 6, 5, 4]
            x = getattr(self, "EncConv_Block_%dx%d" % (res, res))(x)
            out[res_log2] = x  # [9, 8, 7, 6, 5, 4]

        return out


class ToStyle(nn.Module):
    def __init__(
        self,
        in_channels=512,
        out_channels=1024,
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
        self.fc = LReLuFC(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        # tensor [x] size: [1, 512, 16, 16], min: -10.228478, max: 40.67131, mean: 0.077602
        x = self.conv(x)
        x = self.pool(x)
        x = self.fc(x.flatten(start_dim=1))

        # tensor [x] size: [1, 1024], min: -0.370284, max: 0.784699, mean: 0.005559
        return x


# xxxx_debug
class DecBlockFirst(nn.Module):
    def __init__(self, res, in_channels, out_channels, style_dim, img_channels):
        super().__init__()
        # res = 4
        # in_channels = 512
        # out_channels = 512
        # style_dim = 1536
        # img_channels = 3
        self.res = res
        self.conv0 = Conv2dLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
        )
        self.conv1 = StyleConvWithNoise(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
        )

    def forward(self, x, ws, gs, e_features):
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
        # x = self.fc(x).view(x.shape[0], -1, 4, 4)
        x = self.conv0(x)
        x = x + e_features[self.res]  # self.res === 4
        style = get_style_code(ws[:, 0], gs)  # ws[:, 0].size() -- [1, 512]
        x = self.conv1(x, style)
        style = get_style_code(ws[:, 1], gs)
        fake_skip = torch.zeros((1, 3, 16, 16)).to(x.device)
        img = self.toRGB(x, style, skip=fake_skip) # fake_skip = None

        # tensor [x] size: [1, 512, 16, 16], min: -14.745468, max: 52.303444, mean: -0.170848
        # tensor [img] size: [1, 3, 16, 16], min: -0.003557, max: 0.002705, mean: -0.000736
        return x, img


# ----------------------------------------------------------------------------
# xxxx_debug
class DecBlock(nn.Module):
    def __init__(
        self, res, in_channels, out_channels, style_dim, img_channels
    ):  # res = 4, ..., resolution_log2
        super().__init__()
        self.res = res
        self.conv0 = StyleConvWithNoise(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
            up=2,
        )
        self.conv1 = StyleConvWithNoise(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            resolution=2**res,
            kernel_size=3,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
        )

    def forward(self, x, img, ws, gs, e_features):
        style = get_style_code(ws[:, self.res * 2 - 9], gs)
        x = self.conv0(x, style)
        x = x + e_features[self.res]
        style = get_style_code(ws[:, self.res * 2 - 8], gs)
        x = self.conv1(x, style)
        style = get_style_code(ws[:, self.res * 2 - 7], gs)

        img = self.toRGB(x, style, skip=img)

        return x, img


class Decoder(nn.Module):
    def __init__(
        self,
        res_log2=9,
        # activation='lrelu',
        style_dim=1536,
        img_channels=3,
    ):
        super().__init__()
        self.Dec_16x16 = DecBlockFirst(4, nf(4), nf(4), style_dim, img_channels)
        for res in range(5, res_log2 + 1):
            setattr(
                self,
                "Dec_%dx%d" % (2**res, 2**res),
                DecBlock(res, nf(res - 1), nf(res), style_dim, img_channels),
            )
        self.res_log2 = res_log2  # 9


    def forward(self, x, ws, gs, e_features):
        x, img = self.Dec_16x16(x, ws, gs, e_features)
        # for res in range(5, self.res_log2 + 1):
        #     block = getattr(self, "Dec_%dx%d" % (2**res, 2**res))
        #     x, img = block(x, img, ws, gs, e_features)
        x, img = self.Dec_32x32(x, img, ws, gs, e_features)
        x, img = self.Dec_64x64(x, img, ws, gs, e_features)
        x, img = self.Dec_128x128(x, img, ws, gs, e_features)
        x, img = self.Dec_256x256(x, img, ws, gs, e_features)
        x, img = self.Dec_512x512(x, img, ws, gs, e_features)

        return img


class DecStyleBlock(nn.Module):
    def __init__(self, res, in_channels, out_channels, style_dim, img_channels):
        super().__init__()
        self.res = res

        self.conv0 = StyleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            kernel_size=3,
            up=2,
        )
        self.conv1 = StyleConv(
            in_channels=out_channels,
            out_channels=out_channels,
            style_dim=style_dim,
            kernel_size=3,
        )
        self.toRGB = ToRGB(
            in_channels=out_channels,
            out_channels=img_channels,
            style_dim=style_dim,
        )


    def forward(self, x, img, style, skip):
        print("DecStyleBlock ------{")
        todos.debug.output_var("x", x)
        todos.debug.output_var("img", img)
        todos.debug.output_var("style", style)
        todos.debug.output_var("skip", skip)
        print("DecStyleBlock ------}")

        x = self.conv0(x, style)
        x = x + skip
        x = self.conv1(x, style)

        # if img is None:
        #     fake_skip = torch.zeros((1, 3, 16, 16)).to(x.device)
        # else:
        #     fake_skip = img
        img = self.toRGB(x, style, skip=img) # xxxx_1111

        return x, img


class FirstStage(nn.Module):
    def __init__(
        self,
        img_channels=3,
        img_resolution=512,
        dim=180,
        w_dim=512,
    ):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(
            in_channels=img_channels + 1, out_channels=dim, kernel_size=3
        )
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))  # down_time === 3
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(
                    in_channels=dim, out_channels=dim, kernel_size=3, down=2
                )
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1 / 2, 1 / 2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]

        self.tran = nn.ModuleList()
        mid = len(depths) // 2  # len(depths) == 5 ==> mid === 2

        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1 / ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else: # ==> ratios[i] === 1
                merge = PatchIdentity()

            if i < mid:
                self.tran.append(
                    BasicLayerWithMask(
                        dim=dim,
                        input_resolution=[res, res],
                        depth=depth,
                        num_heads=num_heads,
                        window_size=window_sizes[i],
                        downsample=merge,
                    )
                )
            else:
                self.tran.append(
                    BasicLayer(
                        dim=dim,
                        input_resolution=[res, res],
                        depth=depth,
                        num_heads=num_heads,
                        window_size=window_sizes[i],
                        downsample=merge,
                    )
                )

        # global style
        down_conv = []
        for i in range(int(np.log2(16))):
            down_conv.append(
                Conv2dLayer(in_channels=dim, out_channels=dim, kernel_size=3, down=2)
            )
        down_conv.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.down_conv = nn.Sequential(*down_conv)

        self.to_style = LReLuFC(in_features=dim, out_features=dim * 2)
        self.ws_style = LReLuFC(in_features=w_dim, out_features=dim)
        self.to_square = LReLuFC(in_features=dim, out_features=16 * 16)

        style_dim = dim * 3
        self.dec_conv = nn.ModuleList()
        for i in range(down_time):  # down_time == 3, from 64 to input size
            res = res * 2
            self.dec_conv.append(DecStyleBlock(res, dim, dim, style_dim, img_channels))
        # pdb.set_trace()

    def forward(self, images_in, masks_in, ws):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)

        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size
        skips.append(x)
        for i, block in enumerate(self.enc_conv):  # input size to 64
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:  # # xxxx_debug
                skips.append(x)

        x_size = x.size()[-2:]
        x = feature2token(x)
        mask = feature2token(mask)
        mid = len(self.tran) // 2  # len(self.tran) == 5 ==> mid === 2
        for i, block in enumerate(self.tran):  # 64 to 16
            if i < mid:  # # xxxx_debug
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
            elif i > mid:
                x, x_size, mask = block(x, x_size, mask) # xxxx_1111
                x = x + skips[mid - i]
            else:  # ==> i === mid
                x, x_size, mask = block(x, x_size, None) # xxxx_1111

                mul_map = torch.ones_like(x) * 0.5
                mul_map = F.dropout(mul_map, training=True)
                ws = self.ws_style(ws[:, -1])
                add_n = self.to_square(ws).unsqueeze(1)
                add_n = (
                    F.interpolate(
                        add_n, size=x.size(1), mode="linear", align_corners=False
                    )
                    .squeeze(1)
                    .unsqueeze(-1)
                )
                x = x * mul_map + add_n * (1 - mul_map)
                gs = self.to_style(
                    self.down_conv(token2feature(x, x_size)).flatten(start_dim=1)
                )
                style = torch.cat([gs, ws], dim=1)
        # self.tran -- BasicLayer[0] -- BasicLayer[4]

        x = token2feature(x, x_size).contiguous()

        img = torch.zeros(1, 3, 16, 16).to(x.device) # None # xxxx_1111
        for i, block in enumerate(self.dec_conv):  # len(self.dec_conv) === 3
            x, img = block(x, img, style, skips[len(self.dec_conv) - i - 1])

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img


class SynthesisNet(nn.Module):
    def __init__(
        self,
        w_dim=512,  # Intermediate latent (W) dimensionality.
        img_resolution=512,  # Output image resolution.
        img_channels=3,  # Number of color channels.
        # channel_base   = 32768,    # Overall multiplier for the number of channels.
        # channel_decay  = 1.0,
        # channel_max    = 512,      # Maximum number of channels in any layer.
    ):
        super().__init__()
        # w_dim = 512
        # img_resolution = 512
        # img_channels = 3
        # channel_base = 32768
        # channel_decay = 1.0
        # channel_max = 512
        resolution_log2 = int(np.log2(img_resolution))  # ==> 9
        assert img_resolution == 2**resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2  # ==> 12

        # first stage
        self.first_stage = FirstStage(
            img_channels, img_resolution=img_resolution, w_dim=w_dim
        )

        # second stage
        self.enc = Encoder(resolution_log2, img_channels, patch_size=5, channels=16)
        self.to_square = LReLuFC(in_features=w_dim, out_features=16 * 16)
        self.to_style = ToStyle(in_channels=nf(4), out_channels=nf(2) * 2)
        style_dim = w_dim + nf(2) * 2  # ==> 1536
        self.dec = Decoder(resolution_log2, style_dim, img_channels)

    def forward(self, images_in, masks_in, ws):
        out_stg1 = self.first_stage(images_in, masks_in, ws)

        # encoder
        x = images_in * masks_in + out_stg1 * (1 - masks_in)
        x = torch.cat([masks_in - 0.5, x, images_in * masks_in], dim=1)
        e_features = self.enc(x)

        fea_16 = e_features[4]
        mul_map = torch.ones_like(fea_16) * 0.5
        mul_map = F.dropout(mul_map, training=True)
        add_n = self.to_square(ws[:, 0]).view(-1, 16, 16).unsqueeze(1)
        add_n = F.interpolate(
            add_n, size=fea_16.size()[-2:], mode="bilinear", align_corners=False
        )
        fea_16 = fea_16 * mul_map + add_n * (1 - mul_map)
        e_features[4] = fea_16

        # style
        gs = self.to_style(fea_16) # gs === global style ?

        # decoder
        img = self.dec(fea_16, ws, gs, e_features)

        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        # tensor [x] size: [1, 7, 512, 512], min: -1.121117, max: 1.248556, mean: -0.007224
        # tensor [img] size: [1, 3, 512, 512], min: -1.124015, max: 1.043849, mean: 0.075933
        return img


class Generator(nn.Module):
    def __init__(
        self,
        z_dim=512,  # Input latent (Z) dimensionality, 0 = no latent.
        c_dim=0,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim=512,  # Intermediate latent (W) dimensionality.
        img_resolution=512,  # resolution of generated image
        img_channels=3,  # Number of input color channels.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.synthesis = SynthesisNet(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels
        )
        self.mapping = MappingNet(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.synthesis.num_layers
        )
        self.load_weights()

    def forward(self, x):
        images_in = x[:, 0:3, :, :]
        masks_in = x[:, 3:4, :, :]
        z = torch.randn(1, self.z_dim).to(x.device)

        images_in = (images_in - 0.5) * 2.0
        # images_in, masks_in, z

        # todos.debug.output_var("images_in", images_in)
        # todos.debug.output_var("masks_in", masks_in)
        # todos.debug.output_var("z", z)

        ws = self.mapping(z)

        # todos.debug.output_var("ws", ws)
        # tensor [ws] size: [1, 12, 512], min: -0.035399, max: 0.030863, mean: -0.006447

        img = self.synthesis(images_in, masks_in, ws)

        img = (img + 1.0) / 2.0

        return img.clamp(0.0, 1.0)

    def load_weights(self, model_path="models/mat.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


# ----------------------------------------------------------------------------
class MappingFC(nn.Module):
    def __init__(
        self,
        in_features=180,  # Number of input features.
        out_features=180,  # Number of output features.
    ):
        super().__init__()
        bias_init = 0  # Initial value for the additive bias.
        self.in_features = in_features
        self.out_features = out_features
        self.bias_init = bias_init

        self.weight = nn.Parameter(torch.randn([out_features, in_features]) / 0.01)
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init)))
        self.weight_gain = 0.01 / np.sqrt(in_features)
        self.bias_gain = 0.01

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        b = b * self.bias_gain

        x = x.matmul(w.t())
        out = bias_lrelu(x, b, dim=x.ndim - 1)
        return out

    def __repr__(self):
        s = f"MappingFC(in_features={self.in_features}, out_features={self.out_features}, bias_init={self.bias_init})"
        return s


class LinearFC(nn.Module):
    def __init__(
        self,
        in_features=180,  # Number of input features.
        out_features=180,  # Number of output features.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_init = bias_init

        self.weight = nn.Parameter(torch.randn([out_features, in_features]))
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init)))
        self.weight_gain = 1.0 / np.sqrt(in_features)

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias
        x = x.matmul(w.t())
        # print("LinearFC reshape: ", [-1 if i == x.ndim-1 else 1 for i in range(x.ndim)])
        out = x + b.reshape([-1 if i == x.ndim - 1 else 1 for i in range(x.ndim)])
        return out

    def __repr__(self):
        s = f"LinearFC(in_features={self.in_features}, out_features={self.out_features}, bias_init={self.bias_init})"
        return s

class LReLuFC(nn.Module):
    def __init__(
        self,
        in_features=180,  # Number of input features.
        out_features=180,  # Number of output features.
        bias_init=0,  # Initial value for the additive bias.
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_init = bias_init

        self.weight = nn.Parameter(torch.randn([out_features, in_features]))
        self.bias = nn.Parameter(torch.full([out_features], np.float32(bias_init)))
        self.weight_gain = 1.0 / np.sqrt(in_features)

    def forward(self, x):
        w = self.weight * self.weight_gain
        b = self.bias

        x = x.matmul(w.t())
        out = bias_lrelu(x, b, dim=x.ndim - 1)
        return out

    def __repr__(self):
        s = f"LReLuFC(in_features={self.in_features}, out_features={self.out_features}, bias_init={self.bias_init})"
        return s

class Conv2dLayer(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.up = up
        self.down = down

        self.register_buffer("resample_filter", setup_filter([1, 3, 3, 1]))

        self.padding = kernel_size // 2
        self.weight_gain = 1.0 / np.sqrt(in_channels * (kernel_size**2))

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels])
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        w = self.weight * self.weight_gain
        # todos.debug.output_var("x1", x)
        x = conv2d_resample(
            x=x,
            w=w,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )
        # todos.debug.output_var("x2", x)

        out = bias_lrelu(x, self.bias)
        return out

    def __repr__(self):
        s = f"Conv2dLayer(in_channels={self.in_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, up={self.up}, down={self.down}"
        return s


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        style_dim,  # dimension of the style code
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter([1, 3, 3, 1]))

        self.affine = LinearFC(style_dim, in_channels, bias_init=1)

    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        # todos.debug.output_var("x3", x)
        x = conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )
        # todos.debug.output_var("x4", x)

        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out


class ModulatedConv2dD(nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        kernel_size,  # Width and height of the convolution kernel.
        style_dim,  # dimension of the style code
        up=1,  # Integer upsampling factor.
        down=1,  # Integer downsampling factor.
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn([1, out_channels, in_channels, kernel_size, kernel_size])
        )
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size**2))
        self.padding = self.kernel_size // 2
        self.up = up
        self.down = down
        self.register_buffer("resample_filter", setup_filter([1, 3, 3, 1]))

        self.affine = LinearFC(style_dim, in_channels, bias_init=1)


    def forward(self, x, style):
        batch, in_channels, height, width = x.shape
        style = self.affine(style).view(batch, 1, in_channels, 1, 1)
        weight = self.weight * self.weight_gain * style

        # Demodulate Demodulate Demodulate !!!
        decoefs = (weight.pow(2).sum(dim=[2, 3, 4]) + 1e-8).rsqrt()
        weight = weight * decoefs.view(batch, self.out_channels, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channels, in_channels, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channels, height, width)
        # todos.debug.output_var("x5", x)
        x = conv2d_resample(
            x=x,
            w=weight,
            f=self.resample_filter,
            up=self.up,
            down=self.down,
            padding=self.padding,
        )
        # todos.debug.output_var("x6", x)

        out = x.view(batch, self.out_channels, *x.shape[2:])

        return out


class StyleConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        style_dim,  # Intermediate latent (W) dimensionality.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
    ):
        super().__init__()
        self.conv = ModulatedConv2dD(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            up=up,
        )
        self.bias = nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, style):
        x = self.conv(x, style)
        out = bias_lrelu(x, self.bias)

        return out


class StyleConvWithNoise(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        style_dim,  # Intermediate latent (W) dimensionality.
        resolution,  # Resolution of this layer.
        kernel_size=3,  # Convolution kernel size.
        up=1,  # Integer upsampling factor.
    ):
        super().__init__()
        self.conv = ModulatedConv2dD(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            up=up,
        )

        # self.resolution = resolution
        self.register_buffer("noise_const", torch.randn([resolution, resolution]))
        self.noise_strength = nn.Parameter(torch.zeros([]))
        self.bias = nn.Parameter(torch.zeros([out_channels]))

    def forward(self, x, style):
        x = self.conv(x, style)
        noise = self.noise_const * self.noise_strength
        x = x + noise
        return bias_lrelu(x, self.bias)

class ToRGB(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        kernel_size=1,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
        )
        self.bias = nn.Parameter(torch.zeros([out_channels]))
        self.register_buffer("resample_filter", setup_filter([1, 3, 3, 1]))
        #  self.resample_filter.size() -- [4, 4]

    def forward(self, x, style, skip):
        x = self.conv(x, style)
        out = bias_linear(x, self.bias)
        # if skip is not None:
        #     # if skip.shape != out.shape:
        #     #     # skip.shape, out.shape -- [1, 3, 128, 128], [1, 3, 256, 256]
        #     #     skip = upsample2d(skip, self.resample_filter)
        #     #     out = out + skip
        B, C, H, W = out.size()
        skip = F.interpolate(skip, size=(H, W), mode="bilinear", align_corners=False)
        out = out + skip

        return out


def get_style_code(a, b):
    return torch.cat([a, b], dim=1)


class MappingNet(torch.nn.Module):
    def __init__(
        self,
        z_dim=512,  # Input latent (Z) dimensionality
        c_dim=0,  # Conditioning label (C) dimensionality, 0 = no label.
        w_dim=512,  # Intermediate latent (W) dimensionality.
        num_ws=12,  # Number of intermediate latents to output
        num_layers=8,  # Number of mapping layers.
        layer_features=512,  # Number of intermediate features in the mapping layers
    ):
        super().__init__()
        self.num_ws = num_ws
        self.num_layers = num_layers

        features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]
        # features_list --- [512, 512, 512, 512, 512, 512, 512, 512, 512]

        for idx in range(num_layers):  # num_layers -- 8
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = MappingFC(in_features, out_features)
            setattr(self, f"fc{idx}", layer)

        self.register_buffer("w_avg", torch.zeros([w_dim]))
        # self.w_avg.size() -- [512]

    def forward(self, z):
        # tensor [z] size: [1, 512], min: -2.510093, max: 3.06608, mean: 0.064815

        x = normalize_2nd_moment(z.to(torch.float32))

        # Main layers.
        # for idx in range(self.num_layers): # self.num_layers === 8
        #     layer = getattr(self, f'fc{idx}')
        #     x = layer(x)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)

        x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # tensor [x] size: [1, 12, 512], min: -0.035399, max: 0.030863, mean: -0.006447
        return x


# ----------------------------------------------------------------------------
def _conv2d_wrapper(
    x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True
):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations."""
    # print(f"_conv2d_wrapper groups={groups}, flip_weight={flip_weight}")
    out_channels, in_channels_per_group, kh, kw = w.size()

    # Flip weight if requested.
    if (
        not flip_weight
    ):  # conv2d() actually performs correlation (flip_weight=True) not convolution (flip_weight=False).
        w = w.flip([2, 3])
        # ==> pdb.set_trace()
    else:
        # ==> pdb.set_trace()
        pass

    # transpose == True | False
    op = gradfix_conv_transpose2d if transpose else gradfix_conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)


# ----------------------------------------------------------------------------
def conv2d_resample(x, w, f, up=1, down=1, padding=0):
    r"""2D convolution with optional up/downsampling."""
    # x.size() -- [1, 4, 512, 512]
    # w.size() -- [180, 4, 3, 3]
    # f.size() -- [4, 4]
    # ---------------------------------
    # up = 1, down = 1, padding = 1
    # up = 1, down = 2, padding = 1
    # up = 2, down = 1, padding = 1
    # up = 1, down = 1, padding = 0
    # print(f"  conv2d_resample up={up}, down={down}")
    # todos.debug.output_var("w", w)

    assert isinstance(x, torch.Tensor) and (x.ndim == 4)
    assert isinstance(w, torch.Tensor) and (w.ndim == 4) and (w.dtype == x.dtype)
    assert f is None or (
        isinstance(f, torch.Tensor) and f.ndim in [1, 2] and f.dtype == torch.float32
    )
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    out_channels, in_channels_per_group, kh, kw = w.size()  # [180, 4, 3, 3] ?
    fw, fh = f.size()
    # fw === 4, fh === 4

    px0, px1, py0, py1 = padding, padding, padding, padding

    # Adjust padding to account for up/downsampling.
    if up > 1:
        # up == 2 ==> pdb.set_trace()
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        pdb.set_trace()
        x = upfirdn2d(
            x=x, f=f, down=[down, down, down, down], padding=[px0, px1, py0, py1]
        )
        x = _conv2d_wrapper(x=x, w=w, groups=1, flip_weight=True)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        pdb.set_trace()

        x = _conv2d_wrapper(x=x, w=w, groups=1, flip_weight=True)
        x = upfirdn2d(x=x, f=f, up=up, padding=[px0, px1, py0, py1], gain=up**2)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d(x=x, f=f, padding=[px0, px1, py0, py1])
        x = _conv2d_wrapper(x=x, w=w, stride=down, groups=1, flip_weight=True)
        return x

    # Fast path: upsampling with optional downsampling => use transpose strided convolution.
    if up > 1:
        w = w.transpose(0, 1)

        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(
            x=x,
            w=w,
            stride=up,
            padding=[pyt, pxt],
            groups=1,
            transpose=True,
            flip_weight=False,
        )
        x = upfirdn2d(
            x=x, f=f, padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt], gain=up**2
        )
        if down > 1:
            x = upfirdn2d(x=x, f=f, down=[down, down, down, down])
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(
                x=x, w=w, padding=[py0, px0], groups=1, flip_weight=True
            )

    # Fallback: Generic reference implementation.
    x = upfirdn2d(
        x=x,
        f=(f if up > 1 else None),
        up=up,
        padding=[px0, px1, py0, py1],
        gain=up**2,
    )
    x = _conv2d_wrapper(x=x, w=w, groups=1, flip_weight=True)
    if down > 1:
        x = upfirdn2d(x=x, f=f, down=[down, down, down, down])
    return x


def setup_filter(f):
    f = torch.as_tensor(f, dtype=torch.float32)
    f = f.ger(f)
    f /= f.sum()
    return f


# ----------------------------------------------------------------------------
def upfirdn2d(x, f, up=1, down=1, padding=[0, 0, 0, 0], gain=1):
    r"""Pad, upsample, filter, and downsample a batch of 2D images."""
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    if f is None:
        pdb.set_trace()
        f = torch.ones([1, 1], dtype=torch.float32, device=x.device)

    assert isinstance(f, torch.Tensor) and f.ndim in [1, 2]
    assert f.dtype == torch.float32 and not f.requires_grad
    batch_size, num_channels, in_height, in_width = x.shape

    padx0, padx1, pady0, pady1 = padding

    # Upsample by inserting zeros.
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width, 1])
    x = F.pad(x, [0, up - 1, 0, 0, 0, up - 1])
    x = x.reshape([batch_size, num_channels, in_height * up, in_width * up])

    # Pad or crop.
    x = F.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[
        :,
        :,
        max(-pady0, 0) : x.shape[2] - max(-pady1, 0),
        max(-padx0, 0) : x.shape[3] - max(-padx1, 0),
    ]

    # Setup filter.
    f = f * (gain ** (f.ndim / 2))
    f = f.to(x.dtype)
    f = f.flip(list(range(f.ndim)))

    # Convolve with the filter.
    f = f[np.newaxis, np.newaxis].repeat([num_channels, 1] + [1] * f.ndim)
    x = gradfix_conv2d(input=x, weight=f, groups=num_channels)

    # Downsample by throwing away pixels.
    x = x[:, :, ::down, ::down]
    return x


# def upsample2d(x, f, up=2, padding=0, gain=1):
#     r"""Upsample a batch of 2D images using the given 2D FIR filter."""

#     fw, fh = f.size()
#     p = [
#         padding + (fw + up - 1) // 2,
#         padding + (fw - up) // 2,
#         padding + (fh + up - 1) // 2,
#         padding + (fh - up) // 2,
#     ]
#     return upfirdn2d(x, f, up=up, padding=p, gain=gain * up * up)


def bias_lrelu(x, b, dim=1):
    r"""Fused bias and activation function."""
    assert isinstance(x, torch.Tensor)

    # act ----  lrelu, alpha= 0.2 gain= 1.4142135623730951

    # Add bias.
    assert isinstance(b, torch.Tensor) and b.ndim == 1
    assert 0 <= dim < x.ndim
    assert b.shape[0] == x.shape[dim]

    # print("bias_linear reshape -- ", [-1 if i == dim else 1 for i in range(x.ndim)])
    x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])
    # [1, -1], [1, -1, 1, 1], [1, 1, -1],

    # Evaluate activation function.
    alpha = 0.2
    x = nn.functional.leaky_relu(x, alpha)

    # Scale by gain.
    x = x * 1.4142135623730951

    return x


def bias_linear(x, b):
    r"""Fused bias and activation function."""
    dim = 1
    assert isinstance(x, torch.Tensor)
    # act -- linear, alpha= 0.0 gain= 1.0

    # Add bias.
    assert isinstance(b, torch.Tensor) and b.ndim == 1
    assert 0 <= dim < x.ndim
    assert b.shape[0] == x.shape[dim]

    # print("bias_linear reshape -- ", [-1 if i == dim else 1 for i in range(x.ndim)])
    # x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])
    x = x + b.reshape([1, -1, 1, 1])

    return x


# conv2d_gradfix
def gradfix_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return F.conv2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )


def gradfix_conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    groups=1,
    dilation=1,
):
    return F.conv_transpose2d(
        input=input,
        weight=weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        dilation=dilation,
    )
