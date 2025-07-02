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
import uuid
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, reverse_index, generate_chunks
from utils.image_utils import psnr
from utils.sh_utils import RGB2SH
from utils.instancegs_utlis import (
    mask_feature_mean,
    get_SAM_mask_and_feat,
    positional_encoding,
    merge_clusters_graph,
    farthest_point_sampling
)

from gaussian_renderer import render, network_gui, prefilter_voxel, generate_neural_gaussians
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from MLP import PointCloudBackbone

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# Constants
np.random.seed(9)
colors_defined = np.random.randint(100, 256, size=(5000, 3))
colors_defined[0] = np.array([0, 0, 0])
colors_defined = torch.from_numpy(colors_defined)
colors_defined_cuda = colors_defined.to('cuda')
sam_level = 3


class LossFunctions:
    @staticmethod
    def cohesion_loss(gt_image, feat_map, gt_mask, feat_mean_stack, mode=None):
        """Minimize intra-mask feature differences."""
        N, H, W = gt_mask.shape
        C = feat_map.shape[0]
        
        feat_map_expanded = feat_map.unsqueeze(0).expand(N, C, H, W)
        feat_mean_stack_expanded = feat_mean_stack.unsqueeze(-1).unsqueeze(-1).expand(N, C, H, W)
        
        masked_feat = feat_map_expanded * gt_mask.unsqueeze(1)
        dist = (masked_feat - feat_mean_stack_expanded).norm(p=2, dim=1)
        masked_dist = dist * gt_mask
        loss_per_mask = masked_dist.sum(dim=[1, 2]) / gt_mask.sum(dim=[1, 2]).clamp(min=1)
        
        return loss_per_mask.mean()

    @staticmethod
    def separation_loss(feat_mean_stack, mode=None):
        """Maximize inter-mask feature differences."""
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
        
        if mode == 'cotrain':
            low_weight_ind = diff_squared > 0.4
            loss_weight[low_weight_ind == True] = 0.0
            
        inverse_distance *= loss_weight
        loss = inverse_distance.sum() / (N * (N - 1))
        
        return loss

class Trainer:
    def __init__(self, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
        self.dataset = dataset
        self.opt = opt
        self.pipe = pipe
        self.testing_iterations = testing_iterations
        self.saving_iterations = saving_iterations
        self.checkpoint_iterations = checkpoint_iterations
        self.checkpoint = checkpoint
        self.debug_from = debug_from
        
        self.tb_writer = self.prepare_output_and_logger()
        self.gaussians = GaussianModel(
            dataset.feat_dim, dataset.ins_feat_dim, dataset.n_offsets, 
            dataset.voxel_size, dataset.update_depth, dataset.update_init_factor,
            dataset.update_hierachy_factor, dataset.use_feat_bank, 
            dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
            dataset.add_cov_dist, dataset.add_color_dist
        )
        self.scene = Scene(dataset, self.gaussians, shuffle=False)
        self.gaussians.training_setup(opt)
        
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        
        self.viewpoint_stack = None
        self.ema_loss_for_log = 0.0
        self.progress_bar = None

    def prepare_output_and_logger(self):
        if not self.dataset.model_path:
            unique_str = os.getenv('OAR_JOB_ID', str(uuid.uuid4()))
            self.dataset.model_path = os.path.join("./output/", unique_str[0:10])
        
        print(f"Output folder: {self.dataset.model_path}")
        os.makedirs(self.dataset.model_path, exist_ok=True)
        with open(os.path.join(self.dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(self.dataset))))

        if TENSORBOARD_FOUND:
            return SummaryWriter(self.dataset.model_path)
        print("Tensorboard not available: not logging progress")
        return None

    def training_loop(self):
        first_iter = 0
        self.progress_bar = tqdm(range(first_iter, self.opt.iterations), desc="Training progress")
        
        for iteration in range(first_iter, self.opt.iterations + 1):
            self.iter_start.record()
            
            self.gaussians.update_learning_rate(iteration, self.opt.start_root_cb_iter, self.opt.iterations)
            viewpoint_cam = self.get_random_camera()
            
            bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
            voxel_visible_mask = prefilter_voxel(viewpoint_cam, self.gaussians, self.pipe, self.background)
            
            render_pkg = self.render_view(viewpoint_cam, bg, voxel_visible_mask, iteration)
            loss, loss_img, loss_feat = self.compute_losses(render_pkg, viewpoint_cam, iteration)
            
            loss.backward()
            self.iter_end.record()
            
            self.update_progress(iteration, loss, loss_img, loss_feat)
            self.handle_checkpoints(iteration, render_pkg, voxel_visible_mask)
            
            if iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()

        self.progress_bar.close()
        self.post_training_processing(iteration)

    def get_random_camera(self):
        if not self.viewpoint_stack:
            self.viewpoint_stack = self.scene.getTrainCameras().copy()
        return self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))

    def render_view(self, viewpoint_cam, bg, voxel_visible_mask, iteration):
        if (iteration - 1) == self.debug_from:
            self.pipe.debug = True
            
        retain_grad = (iteration < self.opt.update_until and iteration >= 0)
        mode = 'cotrain' if iteration > 20000 else None
        
        return render(
            viewpoint_cam, self.gaussians, self.pipe, bg, iteration,
            rescale=False,
            render_feat_map=True,
            voxel_visible_mask=voxel_visible_mask,
            retain_grad=retain_grad,
            mode=mode
        )

    def compute_losses(self, render_pkg, viewpoint_cam, iteration):
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss_img = loss.clone()
        loss_feat = 0
        
        if iteration > self.opt.start_ins_feat_iter:
            gt_sam_mask = viewpoint_cam.original_sam_mask.cuda()
            mask_id, mask_bool, invalid_pix, saved_idx = get_SAM_mask_and_feat(
                gt_sam_mask, level=sam_level, filter_th=10, sample_mask=False
            )
            
            rendered_ins_feat = render_pkg["ins_feat"]
            rendered_silhouette = (render_pkg["silhouette"] if render_pkg["silhouette"] is not None else render_pkg["alpha"]) > 0.7
            
            feat_mean_stack = mask_feature_mean(
                rendered_ins_feat, mask_bool, 
                image_mask=rendered_silhouette, return_var=False
            )
            
            loss_cohesion = LossFunctions.cohesion_loss(gt_image, rendered_ins_feat, mask_bool, feat_mean_stack)
            loss_separation = LossFunctions.separation_loss(
                feat_mean_stack, 'cotrain' if iteration > 20000 else None
            )
            loss_feat = loss_separation + 0.1 * loss_cohesion
            loss += loss_feat
        
        return loss, loss_img, loss_feat

    def update_progress(self, iteration, loss, loss_img, loss_feat):
        if not torch.isnan(loss):
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            
        if iteration % 10 == 0:
            self.progress_bar.set_postfix({
                "L_total": f"{self.ema_loss_for_log:.{4}f}",
                "L_img": f"{loss_img:.{4}f}",
                "L_feat": f"{loss_feat:.{4}f}",
                "Total_points": f"{self.gaussians.get_anchor.shape[0]}"
            })
            self.progress_bar.update(10)
            
        if iteration == self.opt.iterations:
            self.scene.save(iteration)

    def handle_checkpoints(self, iteration, render_pkg, voxel_visible_mask):
        if iteration in self.testing_iterations:
            self.run_test_iteration(iteration)
            
        if iteration in self.saving_iterations:
            print(f"\n[ITER {iteration}] Saving Gaussians")
            self.scene.save(iteration)
            
        if iteration in self.checkpoint_iterations:
            print(f"\n[ITER {iteration}] Saving Checkpoint")
            torch.save(
                (self.gaussians.capture(), iteration), 
                self.scene.model_path + "/chkpnt" + str(iteration) + ".pth"
            )

    def run_test_iteration(self, iteration):
        viewpoint_stack_test = self.scene.getTestCameras().copy() if self.dataset.eval else self.scene.getTrainCameras().copy()
        with torch.no_grad():
            l1_test = 0.0
            psnr_test = 0.0
            for view in viewpoint_stack_test:
                bg = torch.zeros((3), device="cuda")
                gt_image = view.original_image.cuda()
                voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.pipe, self.background)
                render_pkg = render(
                    view, self.gaussians, self.pipe, bg, iteration,
                    voxel_visible_mask=voxel_visible_mask,
                    render_feat_map=False
                )
                image = render_pkg["render"]
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            
            psnr_test /= len(viewpoint_stack_test)
            l1_test /= len(viewpoint_stack_test)
            print(f"\n[ITER {iteration}] Evaluating: L1 {l1_test} PSNR {psnr_test}")
            torch.cuda.empty_cache()

    def post_training_processing(self, iteration):
        self.gaussians.Voxelizer.voxelize(self.gaussians._anchor)
        self.get_well_optimized_pc(iteration)
        
        viewpoint_stacks = sorted(self.scene.getTrainCameras(), key=lambda Camera: int(Camera.image_name))
        xyz_encoding = F.normalize(positional_encoding(self.gaussians._anchor, d=8), dim=1).cuda()
        cnt = torch.zeros((self.gaussians._anchor.shape[0]), device='cuda')
        
        for view in viewpoint_stacks:
            for idx_label in view.idx_labels:
                cnt[idx_label] += 1
                
        invalid_mask = cnt == 0
        pc_ins_feat = F.normalize(self.gaussians._ins_feat, dim=1).detach()
        points = torch.concat((pc_ins_feat, xyz_encoding), dim=1)
        
        centers, centroids = farthest_point_sampling(
            points[:200000] if points.shape[0] > 200000 else points,
            npoints=1000
        )
        
        distances = torch.cdist(points, centers)
        cluster_distances_f, cluster_indices_f = torch.min(distances, dim=1)
        
        code_book = self.train_cluster_centers(cluster_indices_f, centers.shape[0])
        self.graph_based_aggregation(cluster_indices_f, invalid_mask, code_book)
        
    def train_cluster_centers(self, cluster_indices_f, centers_num):
        with torch.no_grad():
            torch.cuda.empty_cache()
            self.scene.gaussians._anchor = self.scene.gaussians._anchor.detach()
            self.scene.gaussians._offset = self.scene.gaussians._offset.detach()
            self.scene.gaussians._opacity = self.scene.gaussians._opacity.detach()
            self.scene.gaussians._scaling = self.scene.gaussians._scaling.detach()
            self.scene.gaussians._rotation = self.scene.gaussians._rotation.detach()
            self.scene.gaussians._anchor_feat = self.scene.gaussians._anchor_feat.detach()
            
            if self.scene.gaussians.use_feat_bank:
                for param in self.scene.gaussians.mlp_feature_bank.parameters():
                    param.requires_grad = False

            for param in self.scene.gaussians.mlp_opacity.parameters():
                param.requires_grad = False

            for param in self.scene.gaussians.mlp_cov.parameters():
                param.requires_grad = False

            for param in self.scene.gaussians.mlp_color.parameters():
                param.requires_grad = False

            for param in self.scene.gaussians.embedding_appearance.parameters():
                param.requires_grad = False

        new_ins_feat = torch.rand((centers_num, self.gaussians.ins_feat_dim), dtype=torch.float, device="cuda")
        code_book = torch.nn.Parameter(new_ins_feat.requires_grad_(True))
        code_book_optim = torch.optim.Adam([code_book], lr=0.001)

        self.viewpoint_stack = None
        progress_bar = tqdm(range(1, 10001), desc="Training cluster progress")
        
        for iteration in range(1, 10001):
            bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
            view = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
            
            voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.pipe, self.background)
            render_pkg = render(
                view, self.gaussians, self.pipe, bg, iteration,
                voxel_visible_mask=voxel_visible_mask,
                pred_lang_feature=code_book[cluster_indices_f]
            )
            
            image = render_pkg["render"]
            rendered_ins_feat = render_pkg["ins_feat"]
            gt_image = view.original_image.cuda()
            gt_sam_mask = view.original_sam_mask.cuda()
            
            mask_id, mask_bool, invalid_pix, saved_idx = get_SAM_mask_and_feat(
                gt_sam_mask, level=sam_level, filter_th=10, sample_mask=False
            )
            alpha = render_pkg["alpha"]
            rendered_silhouette = (render_pkg["silhouette"] if render_pkg["silhouette"] is not None else alpha) > 0.7
            feat_mean_stack = mask_feature_mean(
                rendered_ins_feat, mask_bool, 
                image_mask=rendered_silhouette, return_var=False
            )
            
            loss_cohesion = LossFunctions.cohesion_loss(gt_image, rendered_ins_feat, mask_bool, feat_mean_stack)
            loss_separation = LossFunctions.separation_loss(feat_mean_stack)
            loss = loss_separation + 0.1 * loss_cohesion
            Ll1 = l1_loss(image, gt_image)
            loss += (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            loss.backward()
            code_book_optim.step()
            code_book_optim.zero_grad(set_to_none=True)
            # torch.cuda.empty_cache()
            
            if not torch.isnan(loss):
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{self.ema_loss_for_log:.{7}f}",
                    "Total_points": f"{self.gaussians.get_anchor.shape[0]}"
                })
                progress_bar.update(10)
                
        progress_bar.close()
        self.scene.save(iteration)
        return code_book

    def graph_based_aggregation(self, cluster_indices_f, invalid_mask, code_book):
        thread_hold = 0.15
        new_code_book = code_book.clone()
        new_code_book = F.normalize(new_code_book, dim=1)
        new_code_book = (new_code_book + 1) / 2
        new_cluster_indices = cluster_indices_f.clone()

        inverse_mapping = {
            i.item(): torch.where(new_cluster_indices == i)[0] 
            for i in torch.unique(new_cluster_indices[~invalid_mask])
        }
        
        new_code_book, merged_pair = merge_clusters_graph(
            new_code_book, inverse_mapping, self.gaussians, thread_hold
        )
        new_cluster_indices = merged_pair[new_cluster_indices]

        self.mlp_smoothing(new_cluster_indices, invalid_mask)

    def mlp_smoothing(self, new_cluster_indices, invalid_mask):
        inverse_mapping = {
            i.item(): torch.where(new_cluster_indices == i)[0] 
            for i in torch.unique(new_cluster_indices[~invalid_mask])
        }
        
        for points in inverse_mapping.values():
            if points.shape[0] < 5:
                invalid_mask[points] = 1
                
        color = torch.tensor(self.gaussians.ori_color, dtype=torch.float, device='cuda')[~invalid_mask]
        xyz = self.gaussians._anchor.clone().detach().requires_grad_(False)[~invalid_mask]
        xyz_encoding = positional_encoding(xyz, d=12).to('cuda')
        new_cluster_indices = new_cluster_indices[~invalid_mask]
        unique_labels, continuous_gt = torch.unique(new_cluster_indices, return_inverse=True)

        print(f"cluster into {unique_labels.shape[0]} centers")
        
        progress_bar = tqdm(range(10000), desc="Training progress")
        
        INPUT_DIM = xyz_encoding.shape[1] + (0 if color is None else color.shape[1])
        feature_pn = torch.concat((color,xyz_encoding, continuous_gt.unsqueeze(-1)), dim=1).detach()

        MLP_net = PointCloudBackbone(
            input_dim=INPUT_DIM,
            output_dim=unique_labels.shape[0],
            hidden_dim=128,
            num_layers=4,
            skip_layer=2
        ).to('cuda')
        
        criterion = torch.nn.CrossEntropyLoss()
        MLP_optimizer = torch.optim.Adam(list(MLP_net.parameters()), lr=0.001)

        for epoch in range(10000):
            chunk = generate_chunks(feature_pn, new_cluster_indices, npoints=1024)
            MLP_input_feature = chunk[:, :-1]
            label = chunk[:, -1].long()
            output2 = MLP_net(MLP_input_feature)
            loss2 = criterion(output2, label)
            
            MLP_optimizer.zero_grad()
            loss2.backward()
            MLP_optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                progress_bar.set_postfix({"MLP_loss": f"{loss2:.{3}f}"})
                progress_bar.update(10)

        self.evaluate_mlp(MLP_net, invalid_mask)

    def evaluate_mlp(self, MLP_net, invalid_mask):
        MLP_net.eval()
        batch_size = 8192
        
        xyz = torch.tensor(self.gaussians.ori_xyz, dtype=torch.float, device='cuda')
              
        xyz_encoding = positional_encoding(xyz, d=12).to('cuda')
        color = torch.tensor(self.gaussians.ori_color, dtype=torch.float, device='cuda')
        feature_pn = torch.concat(
            (color, 
             xyz_encoding),
            dim=1
        ).detach()
        
        new_cluster_indices = torch.zeros((xyz.shape[0]), device='cuda')
        dataset = TensorDataset(feature_pn, new_cluster_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        test_output_list = []
        with torch.no_grad():
            for batch_input, _ in dataloader:
                batch_input = batch_input.to('cuda').unsqueeze(0)
                output = MLP_net(batch_input).squeeze()
                test_output_list.append(output.cpu())

        test_outputs = torch.cat(test_output_list, dim=0)
        max_values, max_indices = torch.max(test_outputs, dim=1)
        unique_indices_list,unique_indices = torch.unique(max_indices, return_inverse=True)

        culster_result = torch.arange(unique_indices.max().item() + 1)[unique_indices]
        torch.save(culster_result, os.path.join(self.scene.model_path, 'cluster_result.pt'))

        self.final_cluster_training(culster_result, unique_indices_list)

    def final_cluster_training(self, culster_result, unique_indices_list):
        # unique_indices_list, unique_indices = torch.unique(culster_result, return_inverse=True)
        # culster_result = torch.arange(unique_indices.max().item() + 1)[unique_indices]
        new_codebook = torch.rand((unique_indices_list.shape[0], self.gaussians.ins_feat_dim), dtype=torch.float, device="cuda")
        new_codebook = torch.nn.Parameter(new_codebook.requires_grad_(True))
        code_book_optim = torch.optim.Adam([new_codebook], lr=0.001)
        self.viewpoint_stack = None
        progress_bar = tqdm(range(1, 5001), desc="Training cluster progress")
        
        for iteration in range(1, 5001):
            bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
            if not self.viewpoint_stack:
                self.viewpoint_stack = self.scene.getTrainCameras().copy()
            view = self.viewpoint_stack.pop(randint(0, len(self.viewpoint_stack)-1))
            
            voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.pipe, self.background)
            render_pkg = render(
                view, self.gaussians, self.pipe, bg, iteration,
                voxel_visible_mask=voxel_visible_mask,
                pred_lang_feature=new_codebook[culster_result]
            )
            
            image = render_pkg["render"]
            rendered_ins_feat = render_pkg["ins_feat"]
            gt_image = view.original_image.cuda()
            gt_sam_mask = view.original_sam_mask.cuda()
            
            mask_id, mask_bool, invalid_pix, saved_idx = get_SAM_mask_and_feat(
                gt_sam_mask, level=sam_level, filter_th=0, sample_mask=False
            )
            alpha = render_pkg["alpha"]
            rendered_silhouette = (render_pkg["silhouette"] if render_pkg["silhouette"] is not None else alpha) > 0.7
            feat_mean_stack = mask_feature_mean(
                rendered_ins_feat, mask_bool, 
                image_mask=rendered_silhouette, return_var=False
            )
            
            loss_cohesion = LossFunctions.cohesion_loss(gt_image, rendered_ins_feat, mask_bool, feat_mean_stack)
            loss_separation = LossFunctions.separation_loss(
                feat_mean_stack, 'diff' if iteration >= 3000 else None
            )
            loss = loss_separation + 0.1 * loss_cohesion
            loss.backward()

            code_book_optim.step()
            code_book_optim.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            
            if not torch.isnan(loss):
                self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{self.ema_loss_for_log:.{7}f}",
                    "Total_points": f"{self.gaussians.get_anchor.shape[0]}"
                })
                progress_bar.update(10)
                
        torch.save(new_codebook, os.path.join(self.scene.model_path, 'cluster_feature.pt'))

    def get_well_optimized_pc(self, iteration):
        with torch.no_grad():
            torch.cuda.empty_cache()
            sorted_train_cameras = sorted(self.scene.getTrainCameras(), key=lambda Camera: int(Camera.image_name))
            
            for idx, view in enumerate(tqdm(sorted_train_cameras, desc="getting well optimized pc")):
                voxel_visible_mask = prefilter_voxel(view, self.gaussians, self.pipe, self.background)
                render_pkg = render(
                    view, self.gaussians, self.pipe, self.background, iteration,
                    rescale=False, pred_lang_feature=None, voxel_visible_mask=voxel_visible_mask
                )
                
                rendered_ins_feat = render_pkg["ins_feat"]
                out_idx = render_pkg["out_idx"]
                idx_contribute = render_pkg["idx_contribute"]
                offset_selection_mask = render_pkg["selection_mask"]

                _, max_idx = torch.max(idx_contribute, dim=0)
                max_idx = max_idx.unsqueeze(0)
                out_idx = out_idx.gather(0, max_idx)
                
                rev_idx = reverse_index(offset_selection_mask, out_idx)
                rev_idx = rev_idx // self.gaussians.n_offsets
                rev_idx = reverse_index(voxel_visible_mask, rev_idx)
                
                idx_labels = []
                mask_id, mask_bool, invalid_pix, saved_idx = get_SAM_mask_and_feat(
                    view.original_sam_mask.cuda(), level=sam_level
                )
                
                for idx_mask in range(mask_bool.shape[0]):
                    idx_bool = mask_bool[idx_mask] == 1
                    idx_label = rev_idx[:, idx_bool].view(-1)
                    idx_labels.append(torch.unique(idx_label))

                pseudo_mask_ins_feat_, mask_var, pix_count = mask_feature_mean(
                    rendered_ins_feat, mask_bool, return_var=True
                )
                
                filter_mask = mask_var > 0.006
                filter_mask = torch.cat((torch.tensor([False]).cuda(), filter_mask), dim=0)
                
                if len(pix_count) == 0:
                    view.idx_labels = []
                    return
                    
                ignored_mask_ind = torch.nonzero(pix_count > pix_count.max() * 0.8).squeeze()
                filter_mask[ignored_mask_ind + 1] = False
                
                filter_mask_4id = ~filter_mask[1:]
                view.idx_labels = [data for data, m in zip(idx_labels, filter_mask_4id) if m]

            torch.cuda.empty_cache()


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations,
                   start_root_cb_iter, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            f"{config['name']}_view_{viewpoint.image_name}/render",
                            image[None], global_step=iteration
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                f"{config['name']}_view_{viewpoint.image_name}/ground_truth",
                                gt_image[None], global_step=iteration
                            )
                    
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test} PSNR {psnr_test}")
                sys.stdout.flush()
                
                if tb_writer:
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{config['name']}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print(f"Optimizing {args.model_path}")
    safe_state(args.quiet)
    
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    trainer = Trainer(
        lp.extract(args), op.extract(args), pp.extract(args),
        args.test_iterations, args.save_iterations,
        args.checkpoint_iterations, args.start_checkpoint,
        args.debug_from
    )
    trainer.training_loop()
    
    print("\nTraining complete.")