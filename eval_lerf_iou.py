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
import torch.nn.functional as F
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, prefilter_voxel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import json
import cv2


def load_images_and_filenames(gt_images_path):
    filenames = [f for f in os.listdir(gt_images_path) if f.endswith('.jpg')]
    images = []
    file_name_without_extensions = []
    for filename in filenames:
        image_path = os.path.join(gt_images_path, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            images.append(image)
        file_name_without_extension = os.path.splitext(filename)[0]
        file_name_without_extensions.append(file_name_without_extension)
    
    return file_name_without_extensions, images

def render_set(model_path, iteration, views, gaussians, pipeline, background):
    # render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    # gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    # render_ins_feat_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_ins_feat")
    # render_silhouette_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_silhouette")
    # makedirs(render_path, exist_ok=True)
    # makedirs(gts_path, exist_ok=True)
    # makedirs(render_ins_feat_path, exist_ok=True)
    # makedirs(render_silhouette_path, exist_ok=True)

    mapping_file = os.path.join(model_path, "cluster_lang.npz")
    saved_data = np.load(mapping_file)
    leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()
    leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()
    leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()
    with open('assets/text_features.json', 'r') as f:
        data_loaded = json.load(f)
    all_texts = list(data_loaded.keys())
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]     # 512
    view_list = {
        "figurines":["frame_00041","frame_00105","frame_00152","frame_00195"],
        "ramen":["frame_00006","frame_00024","frame_00060","frame_00065","frame_00081","frame_00119","frame_00128"],
        "teatime":["frame_00002","frame_00025","frame_00043","frame_00107","frame_00129","frame_00140"],
        "waldo_kitchen":["frame_00053","frame_00066","frame_00089","frame_00140","frame_00154"],
    }
    
    for view_name in view_list.keys():
        if view_name in model_path:
            break
    if view_name != 'ramen':
        leaf_lang_feat[leaf_occu_count < 5] *= 0.0
    else:
        leaf_lang_feat[leaf_occu_count < 2] *= 0.0
    gt_path = os.path.join('/gdata/cold1/lerf_ovs/label',view_name,'gt')
    frame_name = view_list[view_name]
    ious = []
    for frame in frame_name:
        gt_images_path = os.path.join(gt_path, frame)
        file_names, images = load_images_and_filenames(gt_images_path)
        target_text = file_names
        
        query_text_feats = torch.zeros(len(target_text), 512).cuda()
        for i, text in enumerate(target_text):
            feat = text_features[all_texts.index(text)].unsqueeze(0)
            query_text_feats[i] = feat

        for t_i, text_feat in enumerate(query_text_feats):
            text_feat = F.normalize(text_feat.unsqueeze(0), dim=1, p=2)  
            leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)  
            cosine_similarity = torch.matmul(text_feat, leaf_lang_feat.transpose(0, 1))
            max_value, max_id = torch.max(cosine_similarity, dim=-1) # [cluster_num]

            render_mask = leaf_ind == max_id
            render_mask = render_mask.to('cuda')
            for idx, view in enumerate(views):
                if view.image_name != frame:
                    continue
                voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
                mask = voxel_visible_mask * render_mask
                render_pkg = render(view, gaussians, pipeline, background, iteration, \
                        pred_lang_feature = None, voxel_visible_mask = mask)

                rendered_silhouette = render_pkg["silhouette"].squeeze()

                rendered_silhouette = (rendered_silhouette > 0.7) * 255.0

                gt_image = np.array(images[t_i])

                rendered_silhouette = rendered_silhouette.cpu().numpy()

                iou = calculate_iou(rendered_silhouette,gt_image)
                ious.append(iou)
    print('==============================================')
    ious = np.array(ious)
    m50 = np.sum(ious > 0.5)/ious.shape[0]
    m25 = np.sum(ious > 0.25)/ious.shape[0]
    print('scene_name:', view_name, 'mious:', np.mean(ious), 'm50:', m50, 'm25:', m25)
    print('==============================================')

    # # RGB
    # torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + f"_{target_text[t_i]}" + f"_{max_id}.png"))
    # torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + "_.png"))

    # # ins_feat
    # torchvision.utils.save_image(rendered_ins_feat[:3,:,:], os.path.join(render_ins_feat_path, \
    #         view.image_name + f"_{target_text[t_i]}" + f"_{max_id}.png"))
    # # torchvision.utils.save_image(rendered_ins_feat[3:6,:,:], os.path.join(render_ins_feat_path, '{0:05d}'.format(idx) + "_2.png"))

    # torchvision.utils.save_image(rendered_silhouette, os.path.join(render_silhouette_path, \
    #         view.image_name + f"_{target_text[t_i]}" + f"_{max_id}.png"))
            

def calculate_iou(pred, gt):
    intersection = np.sum((pred > 0) & (gt > 0))
    union = np.sum((pred > 0) | (gt > 0))
    iou = intersection / union if union > 0 else 0
    return iou        
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    # print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)