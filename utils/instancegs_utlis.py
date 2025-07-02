import torch
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn.functional as F
def find_connected_components_sparse(adj_matrix):
    # 将邻接矩阵转换为稀疏矩阵格式，并转移到 GPU
    adj_matrix = adj_matrix.to_sparse().to('cuda')
    N = adj_matrix.size(0)
    selected = torch.zeros(N, dtype=torch.bool, device='cuda')
    components = []

    for start_node in range(N):
        if not selected[start_node]:
            # 初始化组件和队列，使用 GPU 张量
            component = []
            selected[start_node] = True
            component.append(start_node)
            queue = [start_node]

            while queue:
                # 取出队列的第一个节点
                u = queue.pop(0)

                # 使用稀疏矩阵的索引来找到邻居
                neighbors = adj_matrix.indices()[1][adj_matrix.indices()[0] == u]

                # 筛选未被访问的邻居
                new_neighbors = neighbors[~selected[neighbors]]
                if len(new_neighbors) > 0:
                    selected[new_neighbors] = True
                    component.extend(new_neighbors.tolist())
                    queue.extend(new_neighbors.tolist())

            components.append(component)

    return components

def calculate_pairwise_distances(tensor1, tensor2, metric=None):
    """
    Calculate L1 (Manhattan) and L2 (Euclidean) distances between every pair of vectors
    in two tensors of shape [m, 6] and [n, 6].
    Args:
        tensor1 (torch.Tensor): A tensor of shape [m, 6].
        tensor2 (torch.Tensor): Another tensor of shape [n, 6].
    Returns:
        torch.Tensor: L1 distances of shape [m, n].
        torch.Tensor: L2 distances of shape [m, n].
    """
    # Reshape tensors to allow broadcasting
    # tensor1 shape becomes [m, 1, 6] and tensor2 shape becomes [1, n, 6]
    tensor1 = tensor1.unsqueeze(1)  # Now tensor1 is [m, 1, 6]
    tensor2 = tensor2.unsqueeze(0)  # Now tensor2 is [1, n, 6]

    # Compute the L1 distance
    if metric == "l1":
        return torch.abs(tensor1 - tensor2).sum(dim=2), None  # Result is [m, n]

    # Compute the L2 distance
    if metric == "l2":
        return None, torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))  # Result is [m, n]

    l1_distances = torch.abs(tensor1 - tensor2).sum(dim=2)
    l2_distances = torch.sqrt((tensor1 - tensor2).pow(2).sum(dim=2))
    return l1_distances, l2_distances

def calculate_iou(masks1, masks2, base=None):
    """
    Calculate the Intersection over Union (IoU) between two sets of masks.
    Args:
        masks1: PyTorch tensor of shape [n, H, W], torch.int32.
        masks2: PyTorch tensor of shape [m, H, W], torch.int32.
    Returns:
        iou_matrix: PyTorch tensor of shape [m, n], containing IoU values.
    """
    # Ensure the masks are of type torch.int32
    if masks1.dtype != torch.bool:
        masks1 = masks1.to(torch.bool)
    if masks2.dtype != torch.bool:
        masks2 = masks2.to(torch.bool)
    
    # Expand masks to broadcastable shapes
    masks1_expanded = masks1.unsqueeze(0)  # [1, n, H, W]
    masks2_expanded = masks2.unsqueeze(1)  # [m, 1, H, W]
    
    # Compute intersection
    intersection = (masks1_expanded & masks2_expanded).float().sum(dim=(2, 3))  # [m, n]
    
    # Compute union
    if base == "former":
        union = (masks1_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    elif base == "later":
        union = (masks2_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    else:
        union = (masks1_expanded | masks2_expanded).float().sum(dim=(2, 3)) + 1e-6  # [m, n]
    
    # Compute IoU
    iou_matrix = intersection / union
    
    return iou_matrix

def get_SAM_mask_and_feat(gt_sam_mask, level=3, filter_th=50, original_mask_feat=None, sample_mask=False):
    """
    input: 
        gt_sam_mask[4, H, W]: mask id
    output:
        mask_id[H, W]: 每个 mask 的 id (0 包含了无效点)
        mask_bool[num_mask+1, H, W]: bool 型，注意返回是去掉第 0 个 mask (无效点)
        invalid_pix[H, W]: bool, 无效点的像素索引
    """

    mask_id = gt_sam_mask[level].clone()
    if level > 0:

        mask_id = mask_id - (gt_sam_mask[level-1].max().detach().cpu()+1)

    if mask_id.min() < 0:
        mask_id = mask_id.clamp_min(-1)   
    mask_id += 1   
    invalid_pix = mask_id==0  


    instance_num = mask_id.max()
    one_hot = F.one_hot(mask_id.type(torch.int64), num_classes=int(instance_num.item() + 1))

    mask_bool = one_hot.permute(2, 0, 1)


    saved_idx = mask_bool.sum(dim=(1,2)) >= filter_th 

    if sample_mask:
        prob = torch.rand(saved_idx.shape[0])
        sample_ind = prob > 0.5
        saved_idx = saved_idx & sample_ind.cuda()
    saved_idx[0] = True
    mask_bool = mask_bool[saved_idx] 

    mask_id = torch.argmax(mask_bool, dim=0) 
    invalid_pix = mask_id==0

    if original_mask_feat is not None:
        mask_feat = original_mask_feat.clone()      
        max_ind = int(gt_sam_mask[level].max())+1
        min_ind = int(gt_sam_mask[level-1].max())+1 if level > 0 else 0
        mask_feat = mask_feat[min_ind:max_ind, :]
        mask_feat = mask_feat[saved_idx[1:]]   
        return mask_id, mask_bool[1:, :, :], mask_feat, invalid_pix
    return mask_id, mask_bool[1:, :, :], invalid_pix, saved_idx

def pair_mask_feature_mean(feat_map, masks):

    N, C, H, W = feat_map.shape


    expanded_masks = masks.unsqueeze(1).expand(-1, C, -1, -1)

    masked_features = feat_map * expanded_masks.float()

    mask_counts = expanded_masks.sum(dim=[2, 3]) + 1e-6

    mean_values = masked_features.sum(dim=[2, 3]) / mask_counts

    return mean_values

def mask_feature_mean(feat_map, gt_masks, image_mask=None, return_var=False):
    """拿 N 个 mask 取同一张特征图中的均值
    feat_map: [C, H, W] 一张特征图
    gt_masks: [N, H, W] N 个 mask
    """
    N, H, W = gt_masks.shape


    feat_expanded = feat_map.unsqueeze(0).expand(N, *feat_map.shape)  # [N, C, H, W]
    masks_expanded = gt_masks.unsqueeze(1).expand(-1, feat_map.shape[0], -1, -1)  # [N, C, H, W]
    if image_mask is not None:
        image_mask_expanded = image_mask.unsqueeze(0).expand(N, feat_map.shape[0], -1, -1)

    if image_mask is not None:
        masked_feats = feat_expanded * masks_expanded.float() * image_mask_expanded.float()
        mask_counts = (masks_expanded * image_mask_expanded.float()).sum(dim=(2, 3))
    else:
        masked_feats = feat_expanded * masks_expanded.float()  # [N, C, H, W]
        mask_counts = masks_expanded.sum(dim=(2, 3))  # [N, C]

    mask_counts = mask_counts.clamp(min=1)

    sum_per_channel = masked_feats.sum(dim=[2, 3]) 
    mean_per_channel = sum_per_channel / mask_counts  

    if not return_var:
        return mean_per_channel  
    else:

        masked_for_variance = torch.where(masks_expanded.bool(), masked_feats - mean_per_channel.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(masked_feats))
        variance_per_channel = (masked_for_variance ** 2).sum(dim=[2, 3]) / mask_counts    # [num_mask, 6]

        mean = mean_per_channel.mean(dim=1)         
        variance = variance_per_channel.mean(dim=1)  

        return mean_per_channel, variance, mask_counts[:, 0]  

def mask_feature_mean_2(feat_map, gt_masks, image_mask=None, return_var=False, batch_size=32):

    N, H, W = gt_masks.shape
    C = feat_map.shape[0]


    feat_expanded = feat_map.unsqueeze(0).expand(N, *feat_map.shape) 
    masks_expanded = gt_masks.unsqueeze(1).expand(-1, C, -1, -1) 
    
    if image_mask is not None:
        image_mask_expanded = image_mask.unsqueeze(0).expand(N, C, -1, -1)

    mean_per_channel_list = []
    variance_per_channel_list = []
    mask_counts_list = []

    for i in range(0, N, batch_size):
        batch_masks = masks_expanded[i:i + batch_size]
        batch_feats = feat_expanded[i:i + batch_size]
        
        if image_mask is not None:
            batch_image_mask = image_mask_expanded[i:i + batch_size]
            masked_feats = batch_feats * batch_masks.float() * batch_image_mask.float()
            mask_counts = (batch_masks * batch_image_mask.float()).sum(dim=(2, 3))
        else:
            masked_feats = batch_feats * batch_masks.float()
            mask_counts = batch_masks.sum(dim=(2, 3))


        mask_counts = mask_counts.clamp(min=1)


        sum_per_channel = masked_feats.sum(dim=[2, 3])  
        mean_per_channel = sum_per_channel / mask_counts 

        mean_per_channel_list.append(mean_per_channel)

        if return_var:
            masked_for_variance = torch.where(batch_masks.bool(), masked_feats - mean_per_channel.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(masked_feats))
            variance_per_channel = (masked_for_variance ** 2).sum(dim=[2, 3]) / mask_counts  # [num_mask, C]

            variance_per_channel_list.append(variance_per_channel)
        
        mask_counts_list.append(mask_counts[:, 0])  

    mean_per_channel = torch.cat(mean_per_channel_list, dim=0)  # [N, C]
    if return_var:
        variance_per_channel = torch.cat(variance_per_channel_list, dim=0)  # [N, C]
        mask_counts = torch.cat(mask_counts_list, dim=0)  # [N]
        variance_per_channel = torch.mean(variance_per_channel,dim=1) # [N]

        return mean_per_channel, variance_per_channel, mask_counts  # [N, C], [N], [N]
    
    return mean_per_channel  # [N, C]

def positional_encoding(positions, d=8):
    N = positions.size(0)
    pe = torch.zeros(N, d)
    if d == 8:
        for i in range(d // 4):
            pe[:, 4*i] = torch.sin(positions[:, 0] / (25 ** (2 * i / d)))  # x
            pe[:, 4*i + 1] = torch.cos(positions[:, 0] / (25 ** (2 * i / d)))  # x
            pe[:, 4*i + 2] = torch.sin(positions[:, 1] / (25 ** (2 * i / d)))  # y
            pe[:, 4*i + 3] = torch.cos(positions[:, 1] / (25 ** (2 * i / d)))  # y
    elif d == 12:
        for i in range(d // 6):
            pe[:, 6*i] = torch.sin(positions[:, 0] / (25 ** (2 * i / d)))  # x
            pe[:, 6*i + 1] = torch.cos(positions[:, 0] / (25 ** (2 * i / d)))  # x
            pe[:, 6*i + 2] = torch.sin(positions[:, 1] / (25 ** (2 * i / d)))  # y
            pe[:, 6*i + 3] = torch.cos(positions[:, 1] / (25 ** (2 * i / d)))  # y
            pe[:, 6*i + 4] = torch.sin(positions[:, 2] / (25 ** (2 * i / d)))  # z
            pe[:, 6*i + 5] = torch.cos(positions[:, 2] / (25 ** (2 * i / d)))  # z
    elif d == 6:
        for i in range(d // 6):
            pe[:, 6*i] = torch.sin(positions[:, 0] / (25 ** (2 * i / d)))  # x
            pe[:, 6*i + 1] = torch.cos(positions[:, 0] / (25 ** (2 * i / d)))  # x
            pe[:, 6*i + 2] = torch.sin(positions[:, 1] / (25 ** (2 * i / d)))  # y
            pe[:, 6*i + 3] = torch.cos(positions[:, 1] / (25 ** (2 * i / d)))  # y
            pe[:, 6*i + 4] = torch.sin(positions[:, 2] / (25 ** (2 * i / d)))  # y
            pe[:, 6*i + 5] = torch.cos(positions[:, 2] / (25 ** (2 * i / d)))  # y
    return pe

def farthest_point_sampling(data, npoints):
    """Farthest point sampling implementation."""
    N, D = data.shape
    centroids = torch.zeros(npoints, dtype=torch.long, device='cuda')
    distance = torch.ones(N, device='cuda') * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long, device='cuda')
    
    for i in range(npoints):
        centroids[i] = farthest
        centroid = data[farthest, :].view(1, -1)
        dist = torch.sum((data - centroid) ** 2, dim=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance)
    
    return data[centroids], centroids

def merge_clusters_graph(codebook, inverse_mapping, gaussians, threshold=0.9):
    """Merge clusters based on feature similarity and spatial proximity."""
    feature_distance = torch.cdist(codebook, codebook)
    feature_distance.fill_diagonal_(1)
    neighbor_flag = torch.zeros((codebook.shape[0], codebook.shape[0]), device='cuda')

    def get_voxel_indices(inverse_mapping, gaussians, idx):
        if idx in inverse_mapping:
            points_indice = inverse_mapping[idx]
            if points_indice.shape[0] < 10:
                return None
            pos = gaussians._anchor[points_indice]
            voxel_indices = torch.floor(pos / 0.1).int()
            return torch.unique(voxel_indices, dim=0)
        return None

    unique_voxels = {i: get_voxel_indices(inverse_mapping, gaussians, i) for i in range(codebook.shape[0])}

    for i in range(codebook.shape[0]):
        unique_voxels_i = unique_voxels[i]
        if unique_voxels_i is None:
            continue
        for j in range(codebook.shape[0]):
            unique_voxels_j = unique_voxels[j]
            if unique_voxels_j is None:
                continue
            i_expanded = unique_voxels_i.unsqueeze(1)
            j_expanded = unique_voxels_j.unsqueeze(0)
            manhattan_distances = torch.abs(i_expanded - j_expanded).sum(dim=2)
            min_distances = manhattan_distances.min().item()
            if min_distances <= 1:
                neighbor_flag[i, j] = 1

    high_similarity_mask = (feature_distance < threshold) * neighbor_flag
    components = find_connected_components_sparse(high_similarity_mask)
    new_node_feature = torch.zeros((len(components), codebook.shape[1]), device='cuda')
    query = torch.zeros(codebook.shape[0], device='cuda', dtype=torch.long)
    
    for idx, comp in enumerate(components):
        query[comp] = idx
        indices = torch.tensor(comp)
        new_node_feature[idx] = torch.mean(codebook[indices], dim=0)

    return new_node_feature, query