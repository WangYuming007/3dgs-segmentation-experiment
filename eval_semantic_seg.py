import os
from plyfile import PlyData, PlyElement
import torch.nn.functional as F
import numpy as np
import torch
import json
from dataclasses import dataclass, field
from typing import Tuple, Type
import torch
    
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

# ScanNet 的 20 个类别标签（不考虑 otherfurniture）
scannet19_dict = {
    1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
    6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
    11: "picture", 12: "counter", 14: "desk", 16: "curtain",
    24: "refrigerator", 28: "shower curtain", 33: "toilet", 34: "sink",
    36: "bathtub", # 39: "otherfurniture"
}

import numpy as np  

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

def calculate_metrics(gt, pred, total_classes):
    pred[gt == 0] = 0

    ious = torch.zeros(total_classes)

    intersection = torch.zeros(total_classes)
    union = torch.zeros(total_classes)
    correct = torch.zeros(total_classes)
    total = torch.zeros(total_classes)

    for cls in range(1, total_classes):
        intersection[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        union[cls] = torch.sum((gt == cls) | (pred == cls)).item()
        correct[cls] = torch.sum((gt == cls) & (pred == cls)).item()
        total[cls] = torch.sum(gt == cls).item()

    valid_union = union != 0
    ious[valid_union] = intersection[valid_union] / union[valid_union]

    gt_classes = torch.unique(gt)
    valid_gt_classes = gt_classes[gt_classes != 0]

    ious = ious.cuda()

    mean_iou = ious[valid_gt_classes].mean().item()

    valid_mask = gt != 0
    correct_predictions = torch.sum((gt == pred) & valid_mask).item()
    total_valid_points = torch.sum(valid_mask).item()
    accuracy = correct_predictions / total_valid_points if total_valid_points > 0 else float('nan')

    class_accuracy = correct / total

    class_accuracy = class_accuracy.cuda()
    mean_class_accuracy = class_accuracy[valid_gt_classes].mean().item()

    return ious, mean_iou, accuracy, mean_class_accuracy, valid_gt_classes

if __name__ == "__main__":
    # scene_list = [  'scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00', 'scene0140_00', 
    #                 'scene0200_00', 'scene0347_00', 'scene0400_00', 'scene0590_00', 'scene0645_00']
    scene_list = ['scene0000_00']
    miousss = []
    maccsss = []
    for scan_name in scene_list:
        file_path = f"/gdata/cold1/ScannetV2/data/scans/{scan_name}/{scan_name}_vh_clean_2.labels.ply"# Modify if you need
        mapping_file = os.path.join('./output/scannet/' + scan_name, "cluster_lang.npz")# Modify if you need

        points, labels = read_labels_from_ply(file_path)

        target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36]   # 19
        # target_id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 33, 34]   # 15
        # target_id = [1, 2, 4, 5, 6, 7, 8, 9, 10, 33] # 10

        target_dict = {key: nyu40_dict[key] for key in target_id}
        target_names = list(target_dict.values())


        target_id_mapping = {value: index + 1 for index, value in enumerate(target_id)}
        updated_labels = np.zeros_like(labels)
        for original_value, new_value in target_id_mapping.items():
            updated_labels[labels == original_value] = new_value
        updated_gt_labels = torch.from_numpy(updated_labels.astype(np.int64)).cuda()
        

        
        
        saved_data = np.load(mapping_file)
        leaf_lang_feat = torch.from_numpy(saved_data["leaf_feat.npy"]).cuda()
        leaf_score = torch.from_numpy(saved_data["leaf_score.npy"]).cuda()
        leaf_occu_count = torch.from_numpy(saved_data["occu_count.npy"]).cuda()
        leaf_ind = torch.from_numpy(saved_data["leaf_ind.npy"]).cuda()
        leaf_lang_feat[leaf_occu_count < 2] *= 0.0

        with open('assets/text_features.json', 'r') as f:
            data_loaded = json.load(f)
        all_texts = list(data_loaded.keys())
        text_features = torch.from_numpy(np.array(list(data_loaded.values()))).to(torch.float32)  # [num_text, 512]
        
        query_text_feats = torch.zeros(len(target_names), 512).cuda()
        for i, text in enumerate(target_names):
            feat = text_features[all_texts.index(text)].unsqueeze(0)
            query_text_feats[i] = feat

        query_text_feats = F.normalize(query_text_feats, dim=1, p=2)  
        leaf_lang_feat = F.normalize(leaf_lang_feat, dim=1, p=2)
        cosine_similarity = torch.matmul(query_text_feats, leaf_lang_feat.transpose(0, 1))
        max_id = torch.argmax(cosine_similarity, dim=0)
        pred_pts_cls_id = max_id[leaf_ind] + 1

        ious, mean_iou, accuracy, mean_acc, valid_class = calculate_metrics(updated_gt_labels, pred_pts_cls_id, total_classes=len(target_names)+1)
        print(f"Scene: {scan_name}, mIoU: {mean_iou:.4f}, mAcc.: {mean_acc:.4f}")
        miousss.append(mean_iou * 100)
        maccsss.append(mean_acc * 100)
print(f'mIoU: {np.mean(np.array(miousss)):.4f}', f'm25: {np.mean(np.array(maccsss)):.4f}')
