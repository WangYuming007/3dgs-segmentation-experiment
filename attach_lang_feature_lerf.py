
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
from utils.instancegs_utlis import mask_feature_mean, pair_mask_feature_mean, \
    get_SAM_mask_and_feat,  \
    calculate_iou,  calculate_pairwise_distances,mask_feature_mean_2

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


np.random.seed(9)
colors_defined = np.random.randint(100, 256, size=(5000, 3))
colors_defined[0] = np.array([0, 0, 0])
colors_defined = torch.from_numpy(colors_defined)
colors_defined_cuda = colors_defined.to('cuda')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sam_level = 3

def training(dataset, pipe):
    with torch.no_grad():
        first_iter = 0
        gaussians = GaussianModel(dataset.feat_dim,dataset.ins_feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                                dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
        scene = Scene(dataset, gaussians,load_iteration=30000)
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        first_iter += 1
    
        culster_result = torch.load(os.path.join(scene.model_path,'cluster_result.pt'))
        cluster_feature = torch.load(os.path.join(scene.model_path,'cluster_feature.pt'))
        view_list = {
            "figurines":["frame_00041","frame_00105","frame_00152","frame_00195"],
            "ramen":["frame_00006","frame_00024","frame_00060","frame_00065","frame_00081","frame_00119","frame_00128"],
            "teatime":["frame_00002","frame_00025","frame_00043","frame_00107","frame_00129","frame_00140"],
            "waldo_kitchen":["frame_00053","frame_00066","frame_00089","frame_00140","frame_00154"],
        }
        min_score_list = {
            "figurines":0.5,
            "ramen":0.2,#0.2
            "teatime":0.5,
            "waldo_kitchen":0.2,
        }
        for view_name in view_list.keys():
            if view_name in dataset.source_path:
                break
        frame_name = view_list[view_name]
        construct_pseudo_ins_feat(scene, render, (pipe, background, first_iter),cluster_feature,culster_result)
        torch.cuda.empty_cache()
        min_score = min_score_list[view_name]
        attach_lang_feature(scene, render4attach, (pipe, background, first_iter),cluster_feature, culster_result, culster_result, min_score)

def construct_pseudo_ins_feat(scene : Scene, renderFunc, renderArgs, new_codebook,cluster_result, filter=True):   
    torch.cuda.empty_cache()

    sorted_train_cameras = sorted(scene.getTrainCameras(), key=lambda Camera: Camera.image_name)
    for idx, view in enumerate(tqdm(sorted_train_cameras, desc="construt pseudo feat")):
        voxel_visible_mask = prefilter_voxel(view, scene.gaussians, renderArgs[0], renderArgs[1])
        render_pkg = renderFunc(view, scene.gaussians, *renderArgs, rescale=False, pred_lang_feature=new_codebook[cluster_result], voxel_visible_mask=voxel_visible_mask)
        with torch.cuda.amp.autocast():
            rendered_ins_feat = render_pkg["ins_feat"]
            render_image = render_pkg["render"]
            mask_id, mask_bool, invalid_pix, saved_idx = \
                get_SAM_mask_and_feat(view.original_sam_mask.cuda(), level=sam_level,filter_th=0)

            pseudo_mask_ins_feat_, mask_var, pix_count = mask_feature_mean_2(rendered_ins_feat, mask_bool, return_var=True)   # [num_mask, 6]
            pseudo_mask_ins_feat = torch.cat((torch.zeros((1, 6)).cuda(), pseudo_mask_ins_feat_), dim=0)

            filter_mask = mask_var > 0.006
            filter_mask = torch.cat((torch.tensor([False]).cuda(), filter_mask), dim=0)

            ignored_mask_ind = torch.nonzero(pix_count > pix_count.max() * 0.8).squeeze()
            filter_mask[ignored_mask_ind + 1] = False
            filtered_mask_pseudo_ins_feat = pseudo_mask_ins_feat.clone()
            filtered_mask_pseudo_ins_feat[filter_mask] *= 0

            pseudo_ins_feat = pseudo_mask_ins_feat[mask_id]
            pseudo_ins_feat = pseudo_ins_feat.permute(2, 0, 1)

            filter_pseudo_ins_feat = filtered_mask_pseudo_ins_feat[mask_id]
            filter_pseudo_ins_feat = filter_pseudo_ins_feat.permute(2, 0, 1)

            mask_bool_filtered = torch.cat((torch.zeros_like(mask_bool[0].unsqueeze(0)), mask_bool), dim=0)
            mask_bool_filtered[filter_mask] *= 0

            view.pesudo_ins_feat = filter_pseudo_ins_feat if filter else pseudo_ins_feat
            view.pesudo_mask_bool = mask_bool_filtered.to(torch.bool)
            view.save_idx = saved_idx

            torch.cuda.empty_cache()
def attach_lang_feature(scene : Scene, renderFunc, renderArgs,new_codebook, culster_result, indices4feature, min_score):  
    import json
    target_names = ['nori', 'sake cup', 'kamaboko', 'corn', 'spoon', 'egg', 'onion segments', 'plate', \
                'napkin', 'bowl', 'glass of water', 'hand', 'chopsticks', 'wavy noodles']
    with open('assets/text_features.json', 'r') as f:
        data_loaded = json.load(f)
    all_texts = list(data_loaded.keys())
    text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]
    query_text_feats = torch.zeros(len(target_names), 512).cuda()
    for i, text in enumerate(target_names):
        feat = text_features[all_texts.index(text)].unsqueeze(0)
        query_text_feats[i] = feat
    
    cluster_id_list = torch.unique(culster_result)
    cluster_number = cluster_id_list.shape[0]
    sorted_train_cameras = sorted(scene.getTrainCameras(), key=lambda Camera: Camera.image_name)
    match_info = torch.zeros(cluster_number, len(sorted_train_cameras), 4).cuda()  # [640, 261, 3]
    for v_id, view in enumerate(tqdm(sorted_train_cameras, desc="Processing Views")):
        pesudo_mask_feat_mean = mask_feature_mean(view.pesudo_ins_feat, view.pesudo_mask_bool)
        voxel_visible_mask = prefilter_voxel(view, scene.gaussians, renderArgs[0], renderArgs[1])
        if sam_level == 0:
            strat_id = 0
            end_id = view.original_sam_mask[sam_level].max().to(torch.int64) + 1
        else:
            strat_id = view.original_sam_mask[sam_level-1].max().to(torch.int64) + 1
            end_id = view.original_sam_mask[sam_level].max().to(torch.int64) + 1
        curr_view_lang_feat = view.original_mask_feat[strat_id:end_id, :]   # [num_mask, 512]
        curr_view_lang_feat = torch.cat((torch.zeros_like(curr_view_lang_feat[0]).unsqueeze(0), \
            curr_view_lang_feat))
        for root_id in cluster_id_list:
            render_mask = culster_result == root_id
            render_mask = render_mask.to('cuda')
            mask = voxel_visible_mask * render_mask
            if torch.sum(mask) < 3:
                continue
            render_pkg = renderFunc(view, scene.gaussians, *renderArgs, \
                        pred_lang_feature=new_codebook[indices4feature], voxel_visible_mask = mask)
            with torch.cuda.amp.autocast():
                rendered_ins_feat = render_pkg["ins_feat"]
                rendered_leaf_cluster_silhouettes = render_pkg["silhouette"]

                image_area = torch.sum(rendered_ins_feat,dim = 0) > 0
                if torch.sum(image_area) / image_area.shape[0] / image_area.shape[1] > 0.95:
                    continue
                rendered_leaf_cluster_silhouettes = (rendered_leaf_cluster_silhouettes > 0.7) * 1.0
                ious = calculate_iou(view.pesudo_mask_bool, rendered_leaf_cluster_silhouettes)

                pred_mask_feat_mean = pair_mask_feature_mean(rendered_ins_feat.unsqueeze(0), rendered_leaf_cluster_silhouettes) 
                l1_dis, _ = calculate_pairwise_distances(pred_mask_feat_mean, pesudo_mask_feat_mean, metric="l1")   # method="l1"

                scores = ious * (1-l1_dis)
                max_score, max_ind = torch.max(scores, dim=-1)
                b_matched = max_score > min_score
                max_score[~b_matched] *= 0
                max_ind[~b_matched] *= 0
                pix_cnt = torch.sum(view.pesudo_mask_bool[max_ind]).unsqueeze(0)
                match_info[root_id, v_id] = torch.stack((max_ind, max_score, b_matched, pix_cnt), dim=1).squeeze(0)
                

    leaf_per_view_matched_mask = match_info[:, :, 0].to(torch.int64)
    match_info_sum = match_info.sum(dim=1)
    leaf_ave_score = match_info_sum[:, 1] / (match_info_sum[:, 2]+ 1e-6)
    leaf_occu_count = match_info_sum[:, 2]
    per_leaf_feat = torch.zeros(cluster_number, 512).cuda()
    per_leaf_feat_sum = torch.zeros(cluster_number, 512).cuda()
    
    leaf_feat_codebook = torch.zeros(cluster_number,len(sorted_train_cameras), 512).cuda()

    for v_id, view in enumerate(sorted_train_cameras):

        if sam_level == 0:
            strat_id = 0
            end_id = view.original_sam_mask[sam_level].max().to(torch.int64) + 1
        else:
            strat_id = view.original_sam_mask[sam_level-1].max().to(torch.int64) + 1
            end_id = view.original_sam_mask[sam_level].max().to(torch.int64) + 1
        curr_view_lang_feat = view.original_mask_feat[strat_id:end_id, :]
        curr_view_lang_feat = torch.cat((torch.zeros_like(curr_view_lang_feat[0]).unsqueeze(0), \
            curr_view_lang_feat))
        single_view_leaf_feat = curr_view_lang_feat[leaf_per_view_matched_mask[:, v_id]]
        leaf_feat_codebook[:,v_id] = single_view_leaf_feat
    
    for cluster_center in range(cluster_number):
        cluster_lang_feature = leaf_feat_codebook[cluster_center]
        non_zero_mask = ~torch.all(cluster_lang_feature == 0, dim=1)
        if torch.sum(non_zero_mask) == 0:
            continue
        non_zero_vectors = cluster_lang_feature[non_zero_mask]
        final_feature = torch.mean(non_zero_vectors,dim=0)
        per_leaf_feat[cluster_center] = final_feature
    np.savez(f'{scene.model_path}/cluster_lang.npz',leaf_feat=per_leaf_feat.cpu().numpy(), \
                                leaf_score=leaf_ave_score.cpu().numpy(), \
                                occu_count=leaf_occu_count.cpu().numpy(),
                                leaf_ind=culster_result.cpu().numpy())
    

    
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
    training(lp.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
