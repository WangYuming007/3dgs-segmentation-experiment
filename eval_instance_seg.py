from plyfile import PlyData
import numpy as np
import torch
import json
import os
def calculate_iou(pred, gt, pred_cnt, gt_cnt):
    iou = np.zeros((gt_cnt, pred_cnt))
    for cls in range(gt_cnt):
        for pred_cls in range(pred_cnt):
            intersection = np.sum((pred == pred_cls) & (gt == cls))
            union = np.sum((pred == pred_cls) | (gt == cls))
            iou[cls, pred_cls] = intersection / union if union > 0 else 0
    return iou

def get_max_iou_mapping(iou):
    max_iou_indices = np.argmax(iou, axis=1)
    max_iou_values = np.max(iou, axis=1)
    return max_iou_indices, max_iou_values

def calculate_miou_and_macc(pred, gt):
    pred_cnt = np.unique(pred).shape[0]
    gt_cnt = np.unique(gt).shape[0]
    iou = calculate_iou(pred, gt, pred_cnt, gt_cnt)
    
    max_iou_indices, max_iou_values = get_max_iou_mapping(iou)

    miou = np.mean(max_iou_values)
   
    return miou, max_iou_values

def read_labels_from_ply(file_path):
    ply_data = PlyData.read(file_path)
    vertex_data = ply_data['vertex'].data
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T
    labels = vertex_data['label']
    return points, labels

dataset_names = [  'scene0000_00', 'scene0062_00', 'scene0070_00', 'scene0097_00', 'scene0140_00', 
                    'scene0200_00', 'scene0347_00', 'scene0400_00', 'scene0590_00', 'scene0645_00']
# dataset_names = ["scene0000_00"]# Modify if you need
miousss = []
maccsss = []
maccsss50 = []
for pt, dataset_name in enumerate(dataset_names):
    GT_pcd_path = f"/gdata/cold1/ScannetV2/data/scans/{dataset_name}"# Modify if you need
    cluster_result = 'ckpt/'+dataset_name+'/cluster_result.pt'

    file_path = os.path.join(GT_pcd_path, f"{dataset_name}_vh_clean_2.labels.ply")
    points, labels = read_labels_from_ply(file_path)
    seg_path = os.path.join(GT_pcd_path, f"{dataset_name}_vh_clean_2.0.010000.segs.json")



    with open(seg_path, 'r') as file:
        data = json.load(file)
    seg_indices = data['segIndices']

    agg_path = os.path.join(GT_pcd_path,f"{dataset_name}_vh_clean.aggregation.json")
    with open(agg_path, 'r') as file:
        data = json.load(file)
    segGroups = data['segGroups']
    label2points = []
    for object in segGroups:
        label2points.append(object['segments'])
    points2label = {}
    for idx,f in enumerate(label2points):
        for point_id in f:
            points2label[point_id] = idx
    GT_instance_label = []
    for indices in seg_indices:
        if indices in points2label.keys():
            GT_instance_label.append(points2label[indices])
        else:
            GT_instance_label.append(-1)

    GT_instance_label = torch.tensor(GT_instance_label)
    mask = GT_instance_label >= 0
    invalid_cnt = torch.sum(mask)

    GT_instance_label = GT_instance_label.numpy()

    
    pred_indices = torch.load(cluster_result)

    pred_indices = pred_indices.cpu().numpy()

    unique_labels, continuous_gt = torch.unique(torch.tensor(pred_indices), return_inverse=True)
    continuous_gt = continuous_gt.numpy() 
    pred_indices = continuous_gt[mask]
    gt_classes = np.unique(GT_instance_label)
    pred_classes = np.unique(pred_indices)

    miou, ious = calculate_miou_and_macc(pred_indices, GT_instance_label[mask])
    m50 = np.sum(ious > 0.5) / ious.shape[0] 
    m25 = np.sum(ious > 0.25) / ious.shape[0] 
    print(dataset_name, f'mIoU: {miou * 100:.2f}', f'm50: {m50 * 100:.2f}', f'm25: {m25 * 100:.2f}')
    miousss.append(miou * 100)
    maccsss.append(m25 * 100)
    maccsss50.append(m50 * 100)

print(f'mIoU: {np.mean(np.array(miousss)):.2f}', f'm50: {np.mean(np.array(maccsss50)):.2f}', f'm25: {np.mean(np.array(maccsss)):.2f}')
