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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
import imageio
import random
from tqdm import tqdm
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.virtual_camera import *

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    depth: np.array     # modify -----
    sam_mask: np.array  # modify -----
    mask_feat: np.array  # modify -----
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly_scannet(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    estimated_normals = np.random.random((len(vertices['x']), 3)) / 255.0
    print("using given point clouds! total "+str(len(vertices['x']))+" points!")
    return BasicPointCloud(points=positions, colors=colors, normals=estimated_normals)

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []

    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        if 'frame_00162' in image_name and 'figurines' in image_path :
            continue
        # print(image_name)
        # print(R,T)
        # print('===================')
        # if 'figurines' in image_name and '162' in image_name:
        #     continue


        # load SAM mask and CLIP feat. modify -----
        mask_seg_path = os.path.join(images_folder[:-6], "language_features/" + extr.name.split('/')[-1][:-4] + "_s.npy")
        mask_feat_path = os.path.join(images_folder[:-6], "language_features/" + extr.name.split('/')[-1][:-4] + "_f.npy")
        # mask_feat_path = None
        if os.path.exists(mask_seg_path):
            sam_mask = np.load(mask_seg_path)    # [level=4, H, W]
        else:
            sam_mask = None
        if mask_feat_path is not None and os.path.exists(mask_feat_path):
            mask_feat = np.load(mask_feat_path)    # [level=4, H, W]
        else:
            mask_feat = None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, cx=width/2, cy=height/2, image=image, 
                              depth=None, sam_mask=sam_mask, mask_feat=mask_feat,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    normals = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    extension=".jpg"
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)

        if "camera_angle_x" not in contents.keys():
            fovx = None
        else:
            fovx = contents["camera_angle_x"] 
        if "cx" not in contents.keys():
            cx = contents["w"] / 2
            cy = contents["h"] / 2
        else:
            cx = contents["cx"]
            cy = contents["cy"]


        frames = contents["frames"]
        for idx, frame in tqdm(enumerate(frames), total=len(frames), desc="load images"):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            depth_path = os.path.join(path, "depth/" + frame["file_path"].split('/')[-1] + ".png")
            if os.path.exists(depth_path):
                depth_image = np.asarray(imageio.imread(depth_path), dtype=np.int64) 
            else:
                depth_image = None
            mask_seg_path = os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_s.npy")
            mask_feat_path = os.path.join(path, "language_features/" + frame["file_path"].split('/')[-1] + "_f.npy")
            if os.path.exists(mask_seg_path):
                sam_mask = np.load(mask_seg_path)    # [level=4, H, W]
            else:
                sam_mask = None
            if os.path.exists(mask_feat_path):
                mask_feat = np.load(mask_feat_path)  # [num_mask, dim=512]
            else:
                mask_feat = None

            if fovx == None:
                focal_length = contents["fl_x"]
                FovY = focal2fov(focal_length, image.size[1])
                FovX = focal2fov(focal_length, image.size[0])
            else:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovx 
                FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, cx=cx, cy=cy, image=image, 
                            depth=depth_image, sam_mask=sam_mask, mask_feat=mask_feat,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

import glob
def read_files_with_specific_extension(folder_path, extension):
    pattern = folder_path + '/*' + extension
    files = glob.glob(pattern)
    return files

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms.json", white_background, extension)
    llffhold = 8
    
    if not eval:
        test_cam_infos = []
    else:
        test_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % llffhold == 0]
        train_cam_infos = [c for idx, c in enumerate(train_cam_infos) if idx % llffhold != 0]
        

    nerf_normalization = getNerfppNorm(train_cam_infos)
    ply_path = path

    file_name = read_files_with_specific_extension(ply_path,'_vh_clean_2.ply')
    ply_path = os.path.join(ply_path, file_name[0])

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly_scannet(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}