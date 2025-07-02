
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

import os
import torch
from random import randint
from gaussian_renderer import render, network_gui, prefilter_voxel, render4attach
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from os import makedirs
import torchvision
import numpy as np
import torch.nn.functional as F
from utils.instancegs_utlis import mask_feature_mean, pair_mask_feature_mean,mask_feature_mean_2, \
    get_SAM_mask_and_feat,  \
    calculate_iou,  calculate_pairwise_distances

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import json
# 300 种配色, 仅用于 SAM mask 可视化
np.random.seed(9)
colors_defined = np.random.randint(100, 256, size=(5000, 3))
colors_defined[0] = np.array([0, 0, 0]) # mask ID 为 -1 的是空，设置为黑色
colors_defined = torch.from_numpy(colors_defined)
colors_defined_cuda = colors_defined.to('cuda')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sam_level = 0

def cohesion_loss(gt_image, feat_map, gt_mask, feat_mean_stack, mode=None):
    """
    让 mask 内的特征尽量相同（与平均值的差异尽量小）
    """
    N, H, W = gt_mask.shape
    C = feat_map.shape[0]
    # 扩展 feat_map 以匹配 gt_mask 的形状 [N, 3, H, W]
    feat_map_expanded = feat_map.unsqueeze(0).expand(N, C, H, W)
    # 调整 feat_mean_stack 的形状以进行广播
    feat_mean_stack_expanded = feat_mean_stack.unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)
    
    # 应用掩码并计算距离    [N, 3, H, W]
    masked_feat = feat_map_expanded * gt_mask.unsqueeze(1)  # 应用掩码
    dist = (masked_feat - feat_mean_stack_expanded).norm(p=2, dim=1)  # 计算距离 [N, H, W]
    
    masked_dist = dist * gt_mask    # [N, H, W]
    loss_per_mask = masked_dist.sum(dim=[1, 2]) / gt_mask.sum(dim=[1, 2]).clamp(min=1)

    # # 计算每个掩码有效区域的损失
    # mask_areas = gt_mask.view(N, -1).sum(dim=1).clamp(min=1)  # 避免除以0
    # loss_per_mask = dist.view(N, -1).sum(dim=1) / mask_areas
    
    # # smooth loss
    # if mode == 'cluster':
    #     sm_loss = smooth_loss(feat_map, gt_image)
    #     return loss_per_mask.mean() + sm_loss.mean()

    # 返回平均损失
    return loss_per_mask.mean()

def separation_loss(feat_mean_stack,mode=None):
    """
    让 mask 间的特征尽量大
    mask_bool: [N, H, W] N 个 mask
    """
    N, _ = feat_mean_stack.shape

    # feat_mean_stack = F.normalize(feat_mean_stack,dim=1)


    # 扩展feat_mean_stack以形成形状为[N, N, C]的张量，进行两两比较
    feat_expanded = feat_mean_stack.unsqueeze(1).expand(-1, N, -1)
    feat_transposed = feat_mean_stack.unsqueeze(0).expand(N, -1, -1)
    
    # 计算所有掩码对之间的差异平方和    [N, N]
    diff_squared = (feat_expanded - feat_transposed).pow(2).sum(2)
    
    # (1) 计算距离的倒数，以鼓励距离的增大
    # 为避免除以0，加上一个小的常数epsilon
    epsilon = 1     # 1e-6
    inverse_distance = 1.0 / (diff_squared + epsilon)
    # # (2) 1 - diff_squared
    # inverse_distance = 1 - diff_squared
    
    # 排除对角线元素（自己与自己的距离），并计算平均倒数距离
    mask = torch.eye(N, device=feat_mean_stack.device).bool()
    inverse_distance.masked_fill_(mask, 0)  

    # NOTE weight
    # (1) 按 loss 大小排序
    sorted_indices = inverse_distance.argsort().argsort()     # 权重排序（按行），[N, N]
    loss_weight = (sorted_indices.float() / (N - 1)) * (1.0 - 0.1) + 0.1    # 缩放到 0.1 - 1.0, [N, N]
    if mode == 'diff':
        low_weight_ind = diff_squared > 0.2
        loss_weight[low_weight_ind == True] = 0.0
        loss_weight[low_weight_ind == False] = 2.0
        inverse_distance *= loss_weight
        CNT = torch.sum(low_weight_ind)
        CNT = max(CNT,1)
        loss = inverse_distance.sum() / CNT
        return loss
    inverse_distance *= loss_weight     # [N, N]
    # final loss
    loss = inverse_distance.sum() / ((N * (N - 1)))

    return loss

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    
    first_iter = 0
    dataset.n_offsets = 5
    gaussians = GaussianModel(dataset.feat_dim,dataset.ins_feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                            dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians,load_iteration=30000)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] 
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter += 1

    thread_hold = 0.15#scannet
    # thread_hold = 0.1#others
    culster_result = torch.load(os.path.join(scene.model_path,'cluster_max_indices_'+str(thread_hold)+'.pt'))
    unique_indices_list,unique_indices = torch.unique(culster_result, return_inverse=True)

    # 如果需要紧凑排列的下标，可以使用:
    culster_result = torch.arange(unique_indices.max().item() + 1)[unique_indices]

    attach_lang_feature(scene, render, (pipe, background, first_iter), culster_result)

def attach_lang_feature(scene : Scene, renderFunc, renderArgs, culster_result):  
    cluster_id_list = torch.unique(culster_result)
    cluster_number = cluster_id_list.shape[0]
    sorted_train_cameras = sorted(scene.getTrainCameras(), key=lambda Camera: int(Camera.image_name))
    match_info = torch.zeros(cluster_number, len(sorted_train_cameras), 4).cuda()  # [640, 261, 3]
    rendered_image_path = os.path.join(scene.model_path, "test_image", "render")
    makedirs(rendered_image_path, exist_ok=True)
    gt_path = os.path.join(scene.model_path, "test_image", "gt")
    makedirs(gt_path, exist_ok=True)
    sam_path = os.path.join(scene.model_path, "test_image", "SAM_mask")
    makedirs(sam_path, exist_ok=True)
    for v_id, view in enumerate(tqdm(sorted_train_cameras, desc="Processing Views")):
        voxel_visible_mask = prefilter_voxel(view, scene.gaussians, renderArgs[0], renderArgs[1])
        render_pkg = renderFunc(view, scene.gaussians, *renderArgs, voxel_visible_mask = voxel_visible_mask)
        image = render_pkg["render"]
        rendered_ins_feat = render_pkg["ins_feat"]
        gt_image = view.original_image.cuda()
        gt_sam_mask = view.original_sam_mask.cuda()
        mask_id, mask_bool, invalid_pix, saved_idx = get_SAM_mask_and_feat(gt_sam_mask, level=sam_level, filter_th=10,\
                sample_mask=False)

        rendered_image_path = os.path.join(scene.model_path, "test_process", "render")
        makedirs(rendered_image_path, exist_ok=True)
        torchvision.utils.save_image(image.detach().cpu(), os.path.join(rendered_image_path, str(v_id) + ".png"))
        gt_image_path = os.path.join(scene.model_path, "test_process", "gt")
        makedirs(gt_image_path, exist_ok=True)
        torchvision.utils.save_image(gt_image.detach().cpu(), os.path.join(gt_image_path, str(v_id) + ".png"))
        ins_feat_path = os.path.join(scene.model_path, "test_process", "ins_feat")
        makedirs(ins_feat_path, exist_ok=True)
        torchvision.utils.save_image(rendered_ins_feat.detach().cpu()[:3, :, :], os.path.join(ins_feat_path, str(v_id) + ".png"))
        mask_color_rand = colors_defined[mask_id.detach().cpu()].type(torch.float64)
        mask_color_rand = mask_color_rand.permute(2, 0, 1)
        gt_sam_path = os.path.join(scene.model_path, "test_process", "gt_sam_mask")
        makedirs(gt_sam_path, exist_ok=True)
        torchvision.utils.save_image(mask_color_rand/255.0, os.path.join(gt_sam_path, str(v_id) + ".png"))

    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--pretrain_path", type=str, default='')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
