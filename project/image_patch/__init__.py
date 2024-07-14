"""Image/Video Patch Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021-2024(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import todos

from . import patch

import pdb

def get_trace_model():
    model =  patch.Generator() 

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_patch_model():
    """Create model."""

    seed = 240  # pick up a random number
    # random.seed(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    model = patch.Generator()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_patch.torch"):
    #     model.save("output/image_patch.torch")

    return model, device


def image_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_patch_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_rgba_tensor(filename)
        input_tensor = F.interpolate(input_tensor, size=(512, 512), mode="bilinear", align_corners=False)
        input_tensor[:, 3:4, :, :] = torch.where(input_tensor[:, 3:4, :, :] < 0.9, 0.0, 1.0)        
        # todos.data.save_tensor([input_tensor[:, 0:3, :, :]], "output/images/0003.png")
        # todos.data.save_tensor([input_tensor[:, 3:4, :, :]], "output/masks/0003.png")

        input_tensor[:, 0:3, :, :] = input_tensor[:, 0:3, :, :] * input_tensor[:, 3:4, :, :]


        orig_tensor = input_tensor.clone().detach()

        predict_tensor = todos.model.forward(model, device, input_tensor)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        todos.data.save_tensor([orig_tensor[:, 0:3, :, :], predict_tensor], output_file)
    todos.model.reset_device()
