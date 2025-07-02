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
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.instancegs_utlis import *
from sklearn.neighbors import NearestNeighbors
from einops import repeat
import numpy as np
import matplotlib.colors as mcolors


def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

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
        appearance = pc.get_appearance(camera_indicies)


    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

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
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    if is_training:
        return xyz, color, opacity, scaling, rot, visible_mask, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, visible_mask

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor,iteration,
            scaling_modifier = 1.0, 
            voxel_visible_mask = None,
            rescale = False,
            render_feat_map=True,  
            render_color=True, 
            retain_grad=None,
            pred_lang_feature = None,
            mode = None): 
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
            cov3D_precomp = None)
    else:
        rendered_image, radii, rendered_depth, rendered_alpha, out_idx = None, None, None, None, None



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
        ins_feat = ins_feat[mask]
        if mode == 'cotrain':
            rendered_ins_feat, _, _, _, _, idx_contribute = rasterizer(
                means3D = means3D,
                means2D = means2D,
                shs = None,
                colors_precomp = ins_feat[:, :3],
                opacities = opacity ,    
                scales = scales ,
                rotations = rotations,
                cov3D_precomp = None)
            feature_count -= 3
            while feature_count > 0:
                rendered_ins_feat2, _, _, _, _, _ = rasterizer(
                    means3D = means3D,
                    means2D = means2D,
                    shs = None,
                    colors_precomp = ins_feat[:, pc.ins_feat_dim-feature_count:pc.ins_feat_dim-feature_count+3],
                    opacities = opacity,
                    scales = scales,
                    rotations = rotations,
                    cov3D_precomp = None)
                rendered_ins_feat = torch.cat((rendered_ins_feat, rendered_ins_feat2), dim=0)
                feature_count -= 3
        else:
            rendered_ins_feat, _, _, _, _, idx_contribute = rasterizer(
                means3D = means3D.detach(),
                means2D = means2D,
                shs = None,
                colors_precomp = ins_feat[:, :3],
                opacities = opacity.detach(),    # 
                scales = scales.detach(),
                # scales = scales*0+0.005,   # *0.1
                rotations = rotations.detach(),
                cov3D_precomp = None)
            feature_count -= 3
            while feature_count > 0:
                rendered_ins_feat2, _, _, _, _, _ = rasterizer(
                    means3D = means3D.detach(),
                    means2D = means2D,
                    shs = None,
                    colors_precomp = ins_feat[:, pc.ins_feat_dim-feature_count:pc.ins_feat_dim-feature_count+3],
                    opacities = opacity.detach(),    
                    scales = scales.detach(),
                    rotations = rotations.detach(),
                    cov3D_precomp = None)
                rendered_ins_feat = torch.cat((rendered_ins_feat, rendered_ins_feat2), dim=0)
                feature_count -= 3
        _, _, _, silhouette, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,   
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
    else:
        rendered_ins_feat, silhouette = None, None
        idx_contribute = None


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
            "idx_contribute":idx_contribute}
    
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
            render_feat_map=True,   # 
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

    ins_feat = ins_feat[mask]
    rendered_ins_feat, _, _, _, _, idx_contribute = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = None,
        colors_precomp = ins_feat[:, :3],
        opacities = opacity.detach(),
        scales = scales.detach(),
        rotations = rotations.detach(),
        cov3D_precomp = None)
    feature_count -= 3
    while feature_count > 0:
        rendered_ins_feat2, _, _, _, _, _ = rasterizer(
            means3D = means3D,
            means2D = means2D,
            shs = None,
            colors_precomp = ins_feat[:, pc.ins_feat_dim-feature_count:pc.ins_feat_dim-feature_count+3],
            opacities = opacity.detach() , 
            scales = scales.detach() ,
            rotations = rotations.detach(),
            cov3D_precomp = cov3D_precomp)
        rendered_ins_feat = torch.cat((rendered_ins_feat, rendered_ins_feat2), dim=0)
        feature_count -= 3
    _, _, _, silhouette, _, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,    
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)





    return {"silhouette": silhouette,
            "ins_feat": rendered_ins_feat}
