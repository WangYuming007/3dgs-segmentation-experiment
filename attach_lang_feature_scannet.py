
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
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from os import makedirs
import numpy as np
import torch.nn.functional as F
from utils.instancegs_utlis import mask_feature_mean, pair_mask_feature_mean,mask_feature_mean_2, \
    get_SAM_mask_and_feat,  \
    calculate_iou,  calculate_pairwise_distances

import json
np.random.seed(9)
colors_defined = np.random.randint(100, 256, size=(5000, 3))
colors_defined[0] = np.array([0, 0, 0]) 
colors_defined = torch.from_numpy(colors_defined)
colors_defined_cuda = colors_defined.to('cuda')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
sam_level = 3

def cohesion_loss(gt_image, feat_map, gt_mask, feat_mean_stack, mode=None):
    """
    让 mask 内的特征尽量相同（与平均值的差异尽量小）
    """
    N, H, W = gt_mask.shape
    C = feat_map.shape[0]
    feat_map_expanded = feat_map.unsqueeze(0).expand(N, C, H, W)
    feat_mean_stack_expanded = feat_mean_stack.unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)
    
    masked_feat = feat_map_expanded * gt_mask.unsqueeze(1) 
    dist = (masked_feat - feat_mean_stack_expanded).norm(p=2, dim=1)
    
    masked_dist = dist * gt_mask
    loss_per_mask = masked_dist.sum(dim=[1, 2]) / gt_mask.sum(dim=[1, 2]).clamp(min=1)
    return loss_per_mask.mean()

def separation_loss(feat_mean_stack,mode=None):
    N, _ = feat_mean_stack.shape

    feat_expanded = feat_mean_stack.unsqueeze(1).expand(-1, N, -1)
    feat_transposed = feat_mean_stack.unsqueeze(0).expand(N, -1, -1)
    
    diff_squared = (feat_expanded - feat_transposed).pow(2).sum(2)
    
    epsilon = 1
    inverse_distance = 1.0 / (diff_squared + epsilon)
    mask = torch.eye(N, device=feat_mean_stack.device).bool()
    inverse_distance.masked_fill_(mask, 0)  

    sorted_indices = inverse_distance.argsort().argsort()     
    loss_weight = (sorted_indices.float() / (N - 1)) * (1.0 - 0.1) + 0.1    
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
    loss = inverse_distance.sum() / ((N * (N - 1)))

    return loss

def training(dataset, opt, pipe):
    
    first_iter = 0
    gaussians = GaussianModel(dataset.feat_dim,dataset.ins_feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank, 
                            dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist)
    scene = Scene(dataset, gaussians,load_iteration=30000)
    gaussians.training_setup(opt)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    first_iter += 1

    culster_result = torch.load(os.path.join(scene.model_path,'cluster_result.pt'))
    new_codebook = torch.load(os.path.join(scene.model_path,'cluster_feature.pt'))



    with torch.no_grad():
        construct_pseudo_ins_feat(scene, render, (pipe, background, first_iter), new_codebook,culster_result)
        torch.cuda.empty_cache()
        min_score = 0.1
        attach_lang_feature(scene, render4attach, (pipe, background, first_iter), new_codebook, culster_result, culster_result, min_score)

def construct_pseudo_ins_feat(scene : Scene, renderFunc, renderArgs, new_codebook, cluster_result, filter=True):   
    torch.cuda.empty_cache()
    sorted_train_cameras = sorted(scene.getTrainCameras(), key=lambda Camera: int(Camera.image_name))
    for idx, view in enumerate(tqdm(sorted_train_cameras, desc="construt pseudo feat")):
        voxel_visible_mask = prefilter_voxel(view, scene.gaussians, renderArgs[0], renderArgs[1])
        render_pkg = renderFunc(view, scene.gaussians, *renderArgs, rescale=False, pred_lang_feature=new_codebook[cluster_result], voxel_visible_mask=voxel_visible_mask)
        gt_image = view.original_image.cuda()
        with torch.cuda.amp.autocast():
            rendered_ins_feat = render_pkg["ins_feat"]
            render_image = render_pkg["render"]
            mask_id, mask_bool, invalid_pix, saved_idx = \
                get_SAM_mask_and_feat(view.original_sam_mask.cuda(), level=sam_level, filter_th=0)#mask_id:min 0 max 27 mask_bool:[num,H,W]

            pseudo_mask_ins_feat_, mask_var, pix_count = mask_feature_mean_2(rendered_ins_feat, mask_bool, return_var=True)   # [num_mask, 6]
            pseudo_mask_ins_feat = torch.cat((torch.zeros((1, 6)).cuda(), pseudo_mask_ins_feat_), dim=0)# [num_mask+1, 6]
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
            view.saved_idx = saved_idx
            torch.cuda.empty_cache()


def attach_lang_feature(scene : Scene, renderFunc, renderArgs,new_codebook, culster_result, indices4feature, min_score):  
    target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]
    nyu40_dict = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 13: "blinds", 14: "desk", 15: "shelves",
    16: "curtain", 17: "dresser", 18: "pillow", 19: "mirror", 20: "floormat",
    21: "clothes", 22: "ceiling", 23: "books", 24: "refrigerator", 25: "television",
    26: "paper", 27: "towel", 28: "showercurtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "nightstand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture", 40: "otherprop"
    }
    target_dict = {key: nyu40_dict[key] for key in target_id}
    target_names = list(target_dict.values())
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
    sorted_train_cameras = sorted(scene.getTrainCameras(), key=lambda Camera: int(Camera.image_name))
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
        curr_view_lang_feat = view.original_mask_feat[strat_id:end_id, :]   
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
                
                l1_dis, _ = calculate_pairwise_distances(pred_mask_feat_mean, pesudo_mask_feat_mean, metric="l1") 

                scores = ious * (1-l1_dis)
                max_score, max_ind = torch.max(scores, dim=-1)
                b_matched = max_score > min_score
                max_score[~b_matched] *= 0
                max_ind[~b_matched] *= 0
                pix_cnt = torch.sum(view.pesudo_mask_bool[max_ind]).unsqueeze(0) #排序还是有问题
                match_info[root_id, v_id] = torch.stack((max_ind, max_score, b_matched, pix_cnt), dim=1).squeeze(0)
                torch.cuda.empty_cache()

    for root_id in cluster_id_list:
        root_info = match_info[root_id]
        last_row = root_info[:, -1]
        top_indices = torch.topk(last_row, 5).indices
        root_matched = torch.zeros_like(root_info[:,2])
        root_matched[top_indices] = 1
        match_info[root_id, : , 2] = match_info[root_id, : , 2] * root_matched
        match_info[root_id, : , 1] = match_info[root_id, : , 2] * match_info[root_id, : , 1]
        match_info[root_id, : , 0] = match_info[root_id, : , 2] * match_info[root_id, : , 0]

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
        curr_view_lang_feat = view.original_mask_feat[strat_id:end_id, :]   # [num_mask, 512]
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
        final_feature = torch.median(non_zero_vectors,dim=0)[0]

        query_text_feats = F.normalize(query_text_feats, dim=1, p=2)  
        cosine_similarity = torch.matmul(query_text_feats, final_feature.unsqueeze(1))
        max_id = torch.argmax(cosine_similarity, dim=0)
        target_name = target_names[max_id]

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
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args))

    # All done
    print("\nTraining complete.")
