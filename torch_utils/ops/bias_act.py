# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Custom PyTorch ops for efficient bias and activation."""

# import os
import warnings
# import numpy as np
import torch
import torch.nn as nn
# import dnnlib
import pdb

#----------------------------------------------------------------------------

# activation_funcs = {
#     'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
#     'relu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
#     'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
#     'tanh':     dnnlib.EasyDict(func=lambda x, **_:         torch.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
#     'sigmoid':  dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
#     'elu':      dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
#     'selu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
#     'softplus': dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
#     'swish':    dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
# }

# def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None):
#     r"""Fused bias and activation function.
#      """
#     assert isinstance(x, torch.Tensor)
#     spec = activation_funcs[act]

#     alpha = float(alpha if alpha is not None else spec.def_alpha)
#     gain = float(gain if gain is not None else spec.def_gain)

#     print("_bias_act_ref ---- ", "act=", act, "alpha=", alpha, "gain=", gain)
#     # act ----  lrelu, alpha= 0.2 gain= 1.4142135623730951
#     # act -- linear, alpha= 0.0 gain= 1.0

#     # Add bias.
#     if b is not None:
#         assert isinstance(b, torch.Tensor) and b.ndim == 1
#         assert 0 <= dim < x.ndim
#         assert b.shape[0] == x.shape[dim]
#         x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])
#     else:
#         pdb.set_trace()

#     # Evaluate activation function.
#     alpha = float(alpha)
#     x = spec.func(x, alpha=alpha)

#     # Scale by gain.
#     gain = float(gain)
#     if gain != 1: # gain= 1.4142135623730951
#         x = x * gain
#     else:
#         # ==> pdb.set_trace()
#         pass

#     return x


def bias_lrelu(x, b=None, dim=1):
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
    x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    # Evaluate activation function.
    alpha = 0.2
    x = nn.functional.leaky_relu(x, alpha)

    # Scale by gain.
    x = x * 1.4142135623730951

    return x


def bias_linear(x, b, dim=1):
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
    x = x + b.reshape([-1 if i == dim else 1 for i in range(x.ndim)])

    return x


#----------------------------------------------------------------------------
