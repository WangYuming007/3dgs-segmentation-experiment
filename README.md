<div align="center">

# [CVPR2025🔥] InstanceGaussian: Appearance-Semantic Joint Gaussian Representation  for 3D Instance-Level Perception

[![arXiv](https://img.shields.io/badge/arXiv-<Paper>-<COLOR>.svg)](https://arxiv.org/abs/2411.19235)
[![Project Page](https://img.shields.io/badge/Project_Page-<Website>-blue.svg)](https://lhj-git.github.io/InstanceGaussian/)

[Haijie Li](https://scholar.google.com/citations?hl=zh-CN&user=QjNgc4MAAAAJ)<sup>1</sup>, [Yanmin Wu](https://yanmin-wu.github.io/)<sup>1</sup>, [Jiarui Meng](https://scholar.google.com/citations?user=N_pRAVAAAAAJ&hl=en&oi=ao)<sup>1</sup>,  [Qiankun Gao](https://gaoqiankun.com/)<sup>1</sup>, Zhiyao Zhang<sup>2</sup>
[Ronggang Wang](https://www.ece.pku.edu.cn/info/1046/2147.htm)<sup>1</sup>, [Jian Zhang](https://jianzhang.tech/cn/)<sup>1</sup>

<sup>1</sup>Peking University, <sup>2</sup>Northeastern University
</div>

## 0. Installation

The installation of OpenGaussian is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).
```shell
git clone https://github.com/yanmin-wu/OpenGaussian.git
```
Then install the dependencies:
```shell
conda env create --file environment.yml
conda activate gaussian_splatting

# the rasterization lib is modified from Scaffold-GS.
cd InstanceGaussian/submodules
pip install diff-gaussian-rasterization
```


## 1. Data preparation
The files are as follows:
```
[DATA_ROOT]
├── [1] scannet/
│	│	├── scene0000_00
|   |   |   |── color/
|   |   |   |── language_features/
|   |   |   |── points3d.ply
|   |   |   |── transforms_train/test.json
|   |   |   |── *_vh_clean_2.ply
|   |   |   |── *_vh_clean_2.0.010000.segs.json (Optional, evaluate IOU)
|   |   |   |── *_vh_clean_2.labels.ply (Optional, evaluate IOU)
│	│	├── scene0062_00
│	│	└── ...
├── [2] lerf_ovs
│	│	├── figurines & ramen & teatime & waldo_kitchen
|   |   |   |── images/
|   |   |   |── language_features/
|   |   |   |── sparse/
│	│	├── label/ (Optional, evaluate IOU)
```
+ **[1] Prepare ScanNet Data**
    + You can directly download OpenGaussian pre-processed data: [**OneDrive**](https://onedrive.live.com/?authkey=%21AIgsXZy3gl%5FuKmM&id=744D3E86422BE3C9%2139813&cid=744D3E86422BE3C9). Please unzip the `color.zip` and `language_features.zip` files. Your need to modify the SAM_level in train_scannet.py or train_lerf.py from 3 to 0.
    + The ScanNet dataset requires permission for use, following the [ScanNet instructions](https://github.com/ScanNet/ScanNet) to apply for dataset permission.
    + You can follow [**Langsplat**](https://github.com/minghanqin/LangSplat) to extract the language feature. 
+ **[2] Prepare lerf_ovs Data**
    + You can directly download our pre-processed data: [**OneDrive**](https://onedrive.live.com/?authkey=%21AIgsXZy3gl%5FuKmM&id=744D3E86422BE3C9%2139815&cid=744D3E86422BE3C9) (re-annotated by LangSplat). Your need to modify the SAM_level in train_scannet.py or train_lerf.py from 3 to 0.
+ **Mask and Language Feature Extraction Details**
    + We use the tools provided by LangSplat to extract the SAM mask and CLIP features, but we only use the large-level mask.



## 3. Training
### 3.1 ScanNet
```shell
# Instance segmentation.
python train_scannet.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to outputpath> -p <path to pretrained_model>
# Attach the semantic feature into instance. Semantic segmentation.
python attach_lang_feature_scannet.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to outputpath> -p <path to pretrained_model>
```

### 3.2 LeRF_ovs
```shell
# Instance segmentation.
python train_lerf.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to outputpath> -p <path to pretrained_model>
# Attach the semantic feature into instance. Semantic segmentation.
python attach_lang_feature_lerf.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to outputpath> -p <path to pretrained_model>
```
## 4. Render & Eval & Downstream Tasks

### 4.1 ScanNet Evalution (Open-Vocabulary Point Cloud Understanding)
+ Evaluate Category-Agnostic 3D Instance Segmentation performance on ScanNet.
    ```shell
    # 1. please check the `GT_pcd_path` and `cluster_result` are correct
    python eval_instance_seg.py
    ```

+ Evaluate text-guided segmentation performance on ScanNet for 19, 15, and 10 categories.
    ```shell
    # 1. please check the `file_path` and `mapping_file` are correct
    # 2. specify `target_id` as 19, 15, or 10 categories.
    python eval_semantic_seg.py
    ```

### 4.2 LeRF Evalution (Open-Vocabulary Object Selection in 3D Space)
+ (1) First, render text-selected 3D Gaussians into multi-view images.
    ```shell
    # 1. specify the model path using -m
    # 2. specify the scene name: figurines, teatime, ramen, waldo_kitchen
    python eval_lerf_iou.py -s <path to COLMAP or NeRF Synthetic dataset> -m <path to outputpath> -p <path to pretrained_model>

    ex: python eval_lerf_iou.py -s data/lerf/ramen -m output/lerf/ramen -p output/lerf/ramen
    ```
    > The metrics may be unstable due to the limited evaluation samples of LeRF.

## 5. Acknowledgements
We are quite grateful for [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [LangSplat](https://github.com/minghanqin/LangSplat), [SAM](https://segment-anything.com/) , and [OpenGaussian](https://github.com/yanmin-wu/OpenGaussian).


## 6. Citation

```
@inproceedings{li2025instancegaussian,
  title={Instancegaussian: Appearance-semantic joint gaussian representation for 3d instance-level perception},
  author={Li, Haijie and Wu, Yanmin and Meng, Jiarui and Gao, Qiankun and Zhang, Zhiyao and Wang, Ronggang and Zhang, Jian},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  year={2025}
}
```



## 7. Contact
If you have any question about this project, please feel free to contact [Haijie Li]: lhj69314[AT]163.com