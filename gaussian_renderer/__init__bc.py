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
import math
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.instancegs_utlis import *
from sklearn.neighbors import NearestNeighbors
import pytorch3d.ops
from einops import repeat
# 100 种随机配色
import numpy as np
import matplotlib.colors as mcolors
# (1) 随机生成 100 种配色
np.random.seed(42)
colors_defined = np.random.randint(100, 255, size=(512, 3))
# colors_defined[0] = np.array([0, 0, 0])
colors_defined = torch.from_numpy(colors_defined)


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    # language_feature = pc._ins_feat[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]
    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity > 0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's language_feature
    # if pc.add_language_feature_dist:
    #     language_feature = pc.get_language_mlp(cat_local_view)
    # else:
    #     language_feature = pc.get_language_mlp(cat_local_view_wodist)
    
    # language_feature = language_feature.reshape([anchor.shape[0]*pc.n_offsets, 6]) # [mask]

    # 第一步：扩展张量
    # expanded_tensor = language_feature.unsqueeze(1).expand(-1, pc.n_offsets, -1)

    # 第二步：调整形状
    # language_feature = expanded_tensor.reshape(-1, pc.ins_feat_dim)

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    # concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets, language_feature], dim=-1)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    # scaling_repeat, repeat_anchor, color, scale_rot, offsets, language_feature = masked.split([6, 3, 3, 7, 3, 6], dim=-1)
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, visible_mask, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, visible_mask
def smooth_colors(pos, color, k=5, weight=0.5):
    """
    在图像平面对颜色做平滑
    """
    # 创建 NearestNeighbors 实例并拟合数据
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(pos)
    distances, indices = nbrs.kneighbors(pos)
    
    # 计算每个点的邻居颜色平均值
    neighbor_colors = np.array([color[idx[1:]].mean(axis=0) for idx in indices])
    
    # 计算平滑后的颜色
    smoothed_color = weight * neighbor_colors + (1 - weight) * color
    
    return smoothed_color
def create_ply_file_from_tensor(filename, points, colors):
    """
    从 PyTorch tensors 创建一个 .ply 文件。
    
    参数:
    - filename: 保存的 .ply 文件名
    - points: 包含点坐标的 tensor，形状为 (N, 3)
    - colors: 包含颜色的 tensor，形状为 (N, 3)
    """
    # 确保 points 和 colors 的大小一致
    assert points.shape[0] == colors.shape[0], "Points and colors tensors must have the same length"
    
    # 转换为 NumPy 数组
    points_np = points.cpu().numpy()
    colors_np = colors.cpu().numpy().astype(int)
    
    # 打开文件进行写入
    with open(filename, 'w') as file:
        # 写入头部
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write(f'element vertex {points_np.shape[0]}\n')
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')
        file.write('property uchar red\n')
        file.write('property uchar green\n')
        file.write('property uchar blue\n')
        file.write('end_header\n')
        
        # 写入点云数据
        for point, color in zip(points_np, colors_np):
            file.write(f'{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n')
def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,iteration,
            scaling_modifier = 1.0, 
            voxel_visible_mask = None,
            rescale = False,
            render_feat_map=True,   # 是否渲染整张图的特征图
            render_color=True,      # 是否渲染 rgb 图像
            retain_grad=None,
            pred_lang_feature = None): 
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    is_training = pc.get_color_mlp.training
    if is_training:
        means3D, colors_precomp, opacity, scales, rotations, visible_mask, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, voxel_visible_mask, is_training=is_training)
    else:
        means3D, colors_precomp, opacity, scales, rotations, visible_mask = generate_neural_gaussians(viewpoint_camera, pc, voxel_visible_mask, is_training=is_training)
    # visible_mask：所有的点->可见的点  mask:可见的点对应的高斯点->可见且不透明的高斯点
    # reverse的过程：可见且不透明的高斯点->可见的点对应的高斯点->可见的点->所有的点

    cov3D_precomp = None

    shs = None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    means2D = screenspace_points
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    if render_color:
        rendered_image, radii, rendered_depth, rendered_alpha, out_idx, _ = rasterizer(
            # means3D = means3D.deatch(),#[N,3]
            means3D = means3D,
            means2D = means2D,#[N,3]
            shs = None,
            colors_precomp = colors_precomp,#[N,3] RGB
            opacities = opacity,#[N,1]
            scales = scales,#[N,3]
            rotations = rotations,#[N,4]
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha, out_idx = None, None, None, None, None
    # else:
    #         rendered_image, radii, rendered_depth, rendered_alpha, out_idx, _ = rasterizer(
    #         means3D = means3D.detach(),#[N,3]
    #         # means3D = means3D,
    #         means2D = means2D,#[N,3]
    #         shs = None,
    #         colors_precomp = colors_precomp,#[N,3] RGB
    #         opacities = opacity,#[N,1]
    #         scales = scales,#[N,3]
    #         rotations = rotations,#[N,4]
    #         cov3D_precomp = cov3D_precomp)
    # RGB img -------------------------------------------------------------

    # (2) 渲染特征图. -------------------------------------------------------
    # 以一定的概率对尺度做 scale
    prob = torch.rand(1)
    # prob2 = torch.rand(1)
    rescale_factor = torch.tensor(1).cuda()
    opacity_rescale_factor = torch.tensor(1).cuda()
    # opacity_rescale_factor = torch.tensor(1).cuda()
    if prob > 0.5 and rescale:  # rescale 为 True 时才对尺度做缩放
        rescale_factor = torch.max(torch.tensor(0.7),torch.rand(1)).cuda()
        opacity_rescale_factor = torch.max(torch.tensor(0.5),torch.rand(1)).cuda()
        # rescale_factor = torch.rand(1).cuda()
    # if prob2 > 0.5 and rescale:   # 对不透明度也做 re-scale
    #     opa_prob = torch.rand_like(opacity) > 0.5
    #     opacity_rescale_factor = opa_prob.to(torch.float32)
    # rescale_factor = torch.rand(1).cuda()
    feature_count = pc.ins_feat_dim
    if render_feat_map:
        if pred_lang_feature is None:
            ins_feat = (pc.get_ins_feat(visible_mask) + 1) / 2   # pseudo -> norm, else -> raw\
        else:
            ins_feat = pred_lang_feature
            ins_feat = torch.nn.functional.normalize(ins_feat, dim=1)[visible_mask]
            ins_feat = ins_feat.unsqueeze(1).expand(-1, pc.n_offsets, -1) #N,5,6 
            ins_feat = ins_feat.reshape(-1,pred_lang_feature.shape[1])
            ins_feat = (ins_feat + 1)/2
            # ins_feat = (ins_feat + 1)/2
        # ins_feat = ins_feat[visible_mask]
        ins_feat = ins_feat[mask]
        # ins_feat = (language_feature + 1)/2
        rendered_ins_feat, _, _, _, _, idx_contribute = rasterizer(#如果维度只有三位那就渲染一次 shs=None?
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = ins_feat[:, :3],
            opacities = opacity.detach() * opacity_rescale_factor,    # 
            scales = scales.detach() * rescale_factor,
            # scales = scales*0+0.005,   # *0.1
            rotations = rotations.detach(),
            cov3D_precomp = None)
        feature_count -= 3
        while feature_count > 0:
            rendered_ins_feat2, _, _, _, _, _ = rasterizer(#否则还要接着渲染，然后拼接起来
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = ins_feat[:, pc.ins_feat_dim-feature_count:pc.ins_feat_dim-feature_count+3],
                opacities = opacity.detach() * opacity_rescale_factor,    # 
                scales = scales.detach() * rescale_factor,
                # scales = scales*0+0.005,   # *0.1
                rotations = rotations.detach(),
                cov3D_precomp = cov3D_precomp)
            rendered_ins_feat = torch.cat((rendered_ins_feat, rendered_ins_feat2), dim=0)
            feature_count -= 3
        _, _, _, silhouette, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity * opacity_rescale_factor,    # 
            scales = scales * rescale_factor,
            # opacities = opacity*0+1.0,    # 
            # scales = scales*0+0.001,   # *0.1
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
    else:
        rendered_ins_feat, silhouette = None, None
        idx_contribute = None
    # ins feat. ---------------------------------------------------------------------
    # opacity_pc = torch.ones_like(opacity, device='cuda')
    # _, _, gt_depth, _, _, _ = rasterizer(
    #     means3D = repeat_anchor,#[N,3]
    #     means2D = means2D.detach(),#[N,3]
    #     shs = None,
    #     colors_precomp = colors_precomp.detach(),#[N,3] RGB
    #     opacities = opacity_pc,#[N,1]
    #     scales = scales.detach(),#[N,3]
    #     rotations = rotations.detach(),#[N,4]
    #     cov3D_precomp = None)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    gt_depth = None
    # create_ply_file_from_tensor('wtf.ply',means3D.detach(),ins_feat[:, :3].detach()*255)



    return {"render": rendered_image,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "silhouette": silhouette,
            "ins_feat": rendered_ins_feat,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "selection_mask": mask,
            "neural_opacity": neural_opacity,
            "scaling":scales,
            "out_idx":out_idx,
            "idx_contribute":idx_contribute,
            "gt_depth":gt_depth}

def visualize_cluster(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, cluster_indices, cluster_ins_feat,
            scaling_modifier = 1.0,  voxel_visible_mask = None, 
            retain_grad=None, invalid_mask = None): 
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    with torch.no_grad():

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D, colors_precomp, opacity, scales, rotations, visible_mask, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, voxel_visible_mask, is_training=True)


        cov3D_precomp = None
        screenspace_points = torch.zeros_like(means3D, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
        means2D = screenspace_points
        if retain_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass
        # RGB img -------------------------------------------------------------
        # corrected_center = torch.cat((corrected_center,pc._ins_feat),dim=1)
        # neighbor_inds, start_len, proposals_idx, proposals_offset = forward_grouping(corrected_center,cluster_flag=True)
        # neighbor_inds, start_len, proposals_idx, proposals_offset = forward_grouping(pc._ins_feat,cluster_flag=True)
        # neighbor_inds : [M] 所有点的邻居的集合
        # start_len ： [N,2] 第一个维度为起始点 第二个维度为区间长度 ，表示邻居点区间在neighbor_inds的下标
        # proposals_idx： [N,2] 每一个点到聚类中心的映射，[0]为聚类中心，[1]为下标，按照聚类中心排序
        # proposals_offset： [K] 每一个聚类中心的起始索引

        # proposals_idx = proposals_idx.long()
        # N = proposals_idx.shape[0]
        # K = proposals_offset.shape[0]

        # 生成 [K, 3] 的随机特征
        # cluster_features = torch.rand(cluster_ins_feat.shape[0], 3, device='cuda')

        # 根据 proposals_idx 赋予每个点相应的特征
        # features = cluster_ins_feat[cluster_indices,:3]
        # features = torch.nn.functional.normalize(features, dim=1)[visible_mask]

        # features = F.normalize(cluster_ins_feat,dim=1)
        features = cluster_ins_feat[cluster_indices,:3][visible_mask]

        features = features.unsqueeze(1).expand(-1, pc.n_offsets, -1) #N,5,6 
        features = features.reshape(-1,3).to('cuda')


        # ins_feat = (features + 1) / 2   # pseudo -> norm, else -> raw\
        ins_feat = features[mask]
        rendered_ins_feat, _, _, _, _, _ = rasterizer(#如果维度只有三位那就渲染一次 shs=None?
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = ins_feat,
            opacities = opacity,    # 
            scales = scales,
            # scales = scales*0+0.005,   # *0.1
            rotations = rotations,
            cov3D_precomp = None)
        # if ins_feat.shape[-1] > 3:
        #     rendered_ins_feat2, _, _, _, _, _ = rasterizer(#否则还要接着渲染，然后拼接起来
        #         means3D = means3D,
        #         means2D = means2D,
        #         shs = None,
        #         colors_precomp = ins_feat[:, 3:6],
        #         opacities = opacity,    # 
        #         scales = scales,
        #         # scales = scales*0+0.005,   # *0.1
        #         rotations = rotations,
        #         cov3D_precomp = cov3D_precomp)
        #     rendered_ins_feat = torch.cat((rendered_ins_feat, rendered_ins_feat2), dim=0)
    return rendered_ins_feat
    
def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0

def render4attach(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,iteration,
            voxel_visible_mask = None,
            render_feat_map=True,   # 是否渲染整张图的特征图
            retain_grad=None,
            pred_lang_feature = None): 
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=True,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    is_training = pc.get_color_mlp.training
    if is_training:
        means3D, colors_precomp, opacity, scales, rotations, visible_mask, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, voxel_visible_mask, is_training=is_training)
    else:
        means3D, colors_precomp, opacity, scales, rotations, visible_mask = generate_neural_gaussians(viewpoint_camera, pc, voxel_visible_mask, is_training=is_training)
    # visible_mask：所有的点->可见的点  mask:可见的点对应的高斯点->可见且不透明的高斯点
    # reverse的过程：可见且不透明的高斯点->可见的点对应的高斯点->可见的点->所有的点

    cov3D_precomp = None

    shs = None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    means2D = screenspace_points
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

    feature_count = pc.ins_feat_dim
    if pred_lang_feature is None:
        ins_feat = (pc.get_ins_feat(visible_mask) + 1) / 2   # pseudo -> norm, else -> raw\
    else:
        ins_feat = pred_lang_feature
        ins_feat = torch.nn.functional.normalize(ins_feat, dim=1)[visible_mask]
        ins_feat = ins_feat.unsqueeze(1).expand(-1, pc.n_offsets, -1) #N,5,6 
        ins_feat = ins_feat.reshape(-1,pred_lang_feature.shape[1])
        ins_feat = (ins_feat + 1)/2
        # ins_feat = (ins_feat + 1)/2
    # ins_feat = ins_feat[visible_mask]
    ins_feat = ins_feat[mask]
    # ins_feat = (language_feature + 1)/2
    rendered_ins_feat, _, _, _, _, idx_contribute = rasterizer(#如果维度只有三位那就渲染一次 shs=None?
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = ins_feat[:, :3],
        opacities = opacity.detach(),
        scales = scales.detach(),
        # scales = scales*0+0.005,   # *0.1
        rotations = rotations.detach(),
        cov3D_precomp = None)
    feature_count -= 3
    while feature_count > 0:
        rendered_ins_feat2, _, _, _, _, _ = rasterizer(#否则还要接着渲染，然后拼接起来
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = ins_feat[:, pc.ins_feat_dim-feature_count:pc.ins_feat_dim-feature_count+3],
            opacities = opacity.detach() ,    # 
            scales = scales.detach() ,
            # scales = scales*0+0.005,   # *0.1
            rotations = rotations.detach(),
            cov3D_precomp = cov3D_precomp)
        rendered_ins_feat = torch.cat((rendered_ins_feat, rendered_ins_feat2), dim=0)
        feature_count -= 3
    _, _, _, silhouette, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,    # 
        scales = scales,
        # opacities = opacity*0+1.0,    # 
        # scales = scales*0+0.001,   # *0.1
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)





    return {"silhouette": silhouette,
            "ins_feat": rendered_ins_feat}