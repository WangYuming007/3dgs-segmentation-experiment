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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import cv2
import torch

WARNED = False

def preprocess_depth(depth: np.ndarray, desired_size, png_depth_scale=1000.0):
        r"""Preprocesses the depth image by resizing, adding channel dimension, and scaling values to meters. Optionally
        converts depth from channels last :math:`(H, W, 1)` to channels first :math:`(1, H, W)` representation.

        Args:
            depth (np.ndarray): Raw depth image

        Returns:
            np.ndarray: Preprocessed depth

        Shape:
            - depth: :math:`(H_\text{old}, W_\text{old})`
            - Output: :math:`(H, W, 1)` if `self.channels_first == False`, else :math:`(1, H, W)`.
        """
        depth = cv2.resize(
            depth.astype(float),
            (desired_size[0], desired_size[1]),
            interpolation=cv2.INTER_NEAREST,
        )
        depth = np.expand_dims(depth, 0) 
        return depth / png_depth_scale

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution) 


    if cam_info.depth is not None:
        gt_depth = preprocess_depth(cam_info.depth, resolution, png_depth_scale=1000.0)
        gt_depth = torch.from_numpy(gt_depth)
    else:
        gt_depth = None

    if cam_info.sam_mask is not None:

        step = int(max(args.resolution, 1))
        gt_sam_mask = cam_info.sam_mask[:, ::step, ::step] 
        gt_sam_mask = torch.from_numpy(gt_sam_mask)

        if resized_image_rgb.shape[1] != gt_sam_mask.shape[1]:
            resolution = (gt_sam_mask.shape[2], gt_sam_mask.shape[1])
            resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    else:
        gt_sam_mask = None
    if cam_info.mask_feat is not None:
        mask_feat = torch.from_numpy(cam_info.mask_feat)
    else:
        mask_feat = None


    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[0] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  cx=cam_info.cx/args.resolution, cy=cam_info.cy/args.resolution,
                  image=gt_image, depth=gt_depth, gt_alpha_mask=loaded_mask,
                  gt_sam_mask=gt_sam_mask, gt_mask_feat=mask_feat,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
