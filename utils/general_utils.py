#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random

def reverse_index(mask, i):

    selected_indices = torch.nonzero(mask, as_tuple=False).squeeze()
    
    original_positions = selected_indices[i.long()]
    
    return original_positions

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def reparameterize(mu, cov_diag, cov_off_diag):
    epsilon = torch.randn_like(mu)

    L = torch.zeros(mu.size(0), 3, 3,device='cuda')
    L[:, 0, 0] = cov_diag[:, 0]
    L[:, 1, 1] = cov_diag[:, 1]
    L[:, 2, 2] = cov_diag[:, 2]
    L[:, 1, 0] = cov_off_diag[:, 0]
    L[:, 2, 0] = cov_off_diag[:, 1]
    L[:, 2, 1] = cov_off_diag[:, 2]
    
    z = mu + torch.bmm(L, epsilon.unsqueeze(-1)).squeeze(-1)
    return z

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def generate_chunks(scene_data, semantic, npoints=8192):
    """
    Generate chunks for training data.

    Parameters:
    - scene_list: list of scene IDs to process.
    - scene_data: dictionary where keys are scene IDs and values are arrays representing the point clouds.
    - npoints: the number of points to sample in each chunk.    
    Returns:
    - chunk_data: dictionary where keys are scene IDs and values are the generated chunks.
    """

    # print("generate new chunks...")

    coordmax = torch.max(scene_data, dim=0).values[:3]
    coordmin = torch.min(scene_data, dim=0).values[:3]

    # Generate chunks
    for _ in range(5):
        curcenter = scene_data[torch.randint(0, semantic.shape[0], (1,)), :3].squeeze()
        curmin = curcenter - torch.tensor([0.75, 0.75, 1.5],device='cuda')
        curmax = curcenter + torch.tensor([0.75, 0.75, 1.5],device='cuda')
        curmin[2] = coordmin[2]
        curmax[2] = coordmax[2]

        curchoice = torch.all((scene_data[:, :3] >= (curmin - 0.2)) & (scene_data[:, :3] <= (curmax + 0.2)), dim=1)
        cur_point_set = scene_data[curchoice]
        cur_semantic_seg = semantic[curchoice]

        if cur_semantic_seg.shape[0] == 0:
            continue

        mask = torch.all((cur_point_set[:, :3] >= (curmin - 0.01)) & (cur_point_set[:, :3] <= (curmax + 0.01)), dim=1)
        vidx = torch.unique(torch.ceil((cur_point_set[mask, :3] - curmin) / (curmax - curmin) * torch.tensor([31.0, 31.0, 62.0],device='cuda')))
        isvalid = torch.sum(cur_semantic_seg > 0).item() / cur_semantic_seg.shape[0] >= 0.7 and len(vidx) / 31.0 / 31.0 / 62.0 >= 0.02

        if isvalid:
            break


    chunk = cur_point_set

    choices = torch.randint(0, chunk.shape[0], (npoints,),device='cuda')
    chunk = chunk[choices]

    return chunk