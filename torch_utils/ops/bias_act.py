# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient bias and activation."""

import warnings
import torch
import torch.nn as nn
import pdb

#----------------------------------------------------------------------------
def bias_lrelu(x, b, dim=1):
    r"""Fused bias and activation function.
     """
    assert isinstance(x, torch.Tensor)

    # print("_bias_act_ref ---- ", "act=", act, "alpha=", alpha, "gain=", gain)
    # act ----  lrelu, alpha= 0.2 gain= 1.4142135623730951
    # act -- linear, alpha= 0.0 gain= 1.0

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
    r"""Fused bias and activation function.
     """
    dim=1     
    assert isinstance(x, torch.Tensor)
    # print("_bias_act_ref ---- ", "act=", act, "alpha=", alpha, "gain=", gain)
    # act ----  lrelu, alpha= 0.2 gain= 1.4142135623730951
    # act -- linear, alpha= 0.0 gain= 1.0

    # Add bias.
    assert isinstance(b, torch.Tensor) and b.ndim == 1
    assert 0 <= dim < x.ndim
    assert b.shape[0] == x.shape[dim]

    # print("bias_linear reshape -- ", [-1 if i == dim else 1 for i in range(x.ndim)])
    # x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])
    x = x + b.reshape([1, -1, 1, 1])

    return x
#----------------------------------------------------------------------------
