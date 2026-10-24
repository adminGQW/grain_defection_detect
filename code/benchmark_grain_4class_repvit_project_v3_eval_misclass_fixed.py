# 【离线阅读导航】本文件同时包含训练、整套数据评估和多个备用模型。
# 推荐顺序：parse_args → main → build_dataloaders/build_transforms → build_model → RepViTProjectNet → train_one_run → evaluate_and_export_split。
# 想调超参数先改命令行；想改网络先找到 build_model 对应分支；单图展示在 ../detect_ui/app.py。
# 本轮只补充中文注释，保留已有算法、默认值及旧版函数；离线修改清单见 ../readme/离线改代码指南.md。
# I'll create an updated benchmark script file for the user that integrates:
# - 3 seeds default
# - no augmentation transforms
# - task-specific dataset statistics table export
# - benchmark result table export
# - best architecture selection by mean Macro-F1
# - best seed selection by highest Test Macro-F1 for that architecture
# - best-model confusion matrices (raw & normalized) in Times New Roman 14 pt
# - misclassified sample CSV + copied image files + grid figure
#
# This is a user-visible code/file generation step.

import os
import json
import time
import copy
import math
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import timm


# ============================================================
# 1. Global configuration
# ============================================================

# 默认模型列表：省略 --models 时可能运行多个模型，初学者建议命令明确写 --models repvit_v4。
DEFAULT_MODELS = [
    "resnet18",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_tiny",
    "repvit",
    "repvit_v2",
    "repvit_v3",
    "repvit_v4"
]

# Default changed to 3 random seeds
# 默认多次独立初始化的种子；--seeds 2026,2027,2028 会分别训练三次，不是三轮。
DEFAULT_SEEDS = [2026, 2027, 2028]

# Figure style for paper-ready plots
# 论文图字体/字号入口；更改只影响绘图，不会改变模型。
PAPER_FONT_FAMILY = "Times New Roman"
PAPER_FONT_SIZE = 14

plt.rcParams["font.family"] = PAPER_FONT_FAMILY
plt.rcParams["font.size"] = PAPER_FONT_SIZE
plt.rcParams["axes.titlesize"] = PAPER_FONT_SIZE
plt.rcParams["axes.labelsize"] = PAPER_FONT_SIZE
plt.rcParams["xtick.labelsize"] = PAPER_FONT_SIZE
plt.rcParams["ytick.labelsize"] = PAPER_FONT_SIZE
plt.rcParams["legend.fontsize"] = PAPER_FONT_SIZE


# ============================================================
# 2. Utility functions
# ============================================================

# 固定 Python、NumPy、PyTorch 的随机数种子，使初始化和打乱顺序尽量可重复。
# 同一 seed（随机过程的起点编号）不保证跨显卡、跨库版本得到逐位相同的结果；数据划分的 seed 另在 MATLAB 中设置。
def set_seed(seed: int):
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 创建输出文件夹，父目录缺失时一起创建；目录已存在不报错。
# 这里不会清空目录，因此重复评估可能留下旧的错图副本；正式比较请使用新的输出目录。
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# 把字典保存为 JSON（可读的键值记录文件）；同路径文件会被覆盖。
# 需要新增实验元信息时可在调用处的字典中加字段，读取端也应相应处理。
def save_json(data, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# 读取已有 JSON 结果。注意：读到旧结果不等于按本次参数重新训练过。
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# 统计 requires_grad=True 的参数数量，即参与学习的数值个数。
# 参数少不直接等于实际推理更快，速度还受算子、输入尺寸和硬件影响。
def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 把秒数转换成便于阅读的时长；仅影响打印，不改变训练计时。
def format_seconds(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


# 把均值和标准差格式化为“均值 ± 标准差”。
# 这是旧论文表格辅助函数；当前主流程按每个 seed 单独汇总，并未计算跨 seed 均值。
def num_to_mean_std_str(mean_value: float, std_value: float, digits: int = 4) -> str:
    if pd.isna(std_value):
        std_value = 0.0
    return f"{mean_value:.{digits}f} ± {std_value:.{digits}f}"


# ============================================================
# 3. Dataset & transforms
# ============================================================

# 带原图路径的图片数据集。ImageFolder 根据子文件夹名确定类别，并按类别名排序编号。
# 例如一个样本会返回 (image, label, path)，路径用于追查错分图片；路径本身不送入模型。
class ImageFolderWithPaths(datasets.ImageFolder):
    """
    ImageFolder that also returns image file path.
    Used for final test-set analysis and misclassification export.
    """
    # 读取第 index 张图：沿用父类的读图和预处理，再补上文件路径。
    # 修改返回值数量时，要一起检查 evaluate 中对二元组/三元组 batch 的解包。
    def __getitem__(self, index: int):
        image, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, label, path


# 统一定义图像预处理：Resize（缩放到固定尺寸）→ ToTensor（图片变数字数组）→ Normalize（按均值/标准差调整数值）。
# 输出通常为 [3, img_size, img_size]，3 表示 RGB 三个颜色通道；直接缩放会改变非方形图片的比例。
# 训练与评估在这里使用相同处理，没有在线随机增强；本工程的翻转增强在 MATLAB 划分脚本中完成。
# 改尺寸、裁剪或归一化时，评估和 detect_ui/app.py 必须使用同一约定；界面目前固定 224。
def build_transforms(img_size: int = 224):
    """
    No data augmentation.
    Training / validation / testing all use:
      Resize -> ToTensor -> Normalize
    """
    common_transform = transforms.Compose([
        # 固定高/宽；想保持长宽比不能只改一个数字，还要设计一致的裁剪/填充方案并重新比较。
        transforms.Resize((img_size, img_size)),
        # 把 PIL 图像转换为浮点张量（多维数字数组），通常将 0~255 转为 0~1，并由 HWC 排列成 CHW。
        transforms.ToTensor(),
        # 逐通道执行 (像素值-mean)/std；mean/std 要与预训练模型及后续推理保持一致。
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_transform = common_transform
    eval_transform = common_transform

    return train_transform, eval_transform


# 建立 train/val/test 的 DataLoader（把图片分批送给模型的读取器）。
# data_root 必须直接含 train、val、test，而不是再套一层同名目录；每个集合下再放类别文件夹。
# 训练集 shuffle=True 表示每轮打乱；验证/测试不打乱，便于路径与预测对应。
# 类别编号来自 ImageFolder.class_to_idx，三个集合必须完全一致；文件重命名不会直接改变类别。
def build_dataloaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    test_with_paths: bool = False
):
    train_transform, eval_transform = build_transforms(img_size)

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")
    test_dir  = os.path.join(data_root, "test")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"train directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"val directory not found: {val_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"test directory not found: {test_dir}")

    # 标签由“类别文件夹”给出，不靠图片文件名；增加类别需要三个集合的子目录与权重输出维度一起调整。
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)

    if test_with_paths:
        test_dataset = ImageFolderWithPaths(test_dir, transform=eval_transform)
    else:
        test_dataset = datasets.ImageFolder(test_dir, transform=eval_transform)

    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Class mappings between train and val are inconsistent.")
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Class mappings between train and test are inconsistent.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        # 将 CPU 数据放在便于向 GPU 传输的内存中；不会把整个数据集永久放入显存。
        pin_memory=True,
        # 保留最后不足一批的图片；改 True 会少用部分图片，且当前损失分母仍用完整数据集大小，需一起审查。
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # 将 CPU 数据放在便于向 GPU 传输的内存中；不会把整个数据集永久放入显存。
        pin_memory=True,
        # 保留最后不足一批的图片；改 True 会少用部分图片，且当前损失分母仍用完整数据集大小，需一起审查。
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # 将 CPU 数据放在便于向 GPU 传输的内存中；不会把整个数据集永久放入显存。
        pin_memory=True,
        # 保留最后不足一批的图片；改 True 会少用部分图片，且当前损失分母仍用完整数据集大小，需一起审查。
        drop_last=False
    )

    info = {
        "num_classes": len(train_dataset.classes),
        "class_names": train_dataset.classes,
        "class_to_idx": train_dataset.class_to_idx,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset)
    }

    return train_loader, val_loader, test_loader, info


# 只为指定集合建立评估读取器，可返回原图路径。
# expected_class_names 来自权重文件，用于核对当前目录类别顺序，避免“编号一样、含义不同”。
# 评估集也需要类别子目录；这个入口不能直接用于一张没有标签的散图。单图展示使用 app.py。
def build_eval_loader_for_split(
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    with_paths: bool = True,
    expected_class_names: Optional[List[str]] = None
):
    """
    Build a dataloader for a specific split (train / val / test).
    This is mainly used for evaluation-only mode with existing best checkpoints.
    """
    _, eval_transform = build_transforms(img_size)
    split_dir = os.path.join(data_root, split)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"{split} directory not found: {split_dir}")

    if with_paths:
        dataset = ImageFolderWithPaths(split_dir, transform=eval_transform)
    else:
        dataset = datasets.ImageFolder(split_dir, transform=eval_transform)

    if expected_class_names is not None:
        if list(dataset.classes) != list(expected_class_names):
            raise ValueError(
                "Class names in dataset split are inconsistent with checkpoint.\n"
                f"Dataset classes   : {dataset.classes}\n"
                f"Checkpoint classes: {expected_class_names}"
            )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        # 将 CPU 数据放在便于向 GPU 传输的内存中；不会把整个数据集永久放入显存。
        pin_memory=True,
        # 保留最后不足一批的图片；改 True 会少用部分图片，且当前损失分母仍用完整数据集大小，需一起审查。
        drop_last=False
    )

    info = {
        "split": split,
        "num_classes": len(dataset.classes),
        "class_names": dataset.classes,
        "class_to_idx": dataset.class_to_idx,
        "split_size": len(dataset)
    }

    return loader, info


# ============================================================
# 4. Dataset statistics table: Paper Table 1 (task-specific)
# ============================================================

# 按允许扩展名统计当前类别目录的直接子文件，不递归子目录。
# 若自行增加更深目录，ImageFolder 的读取数量与本统计可能不一致，应核对 data_info.json。
def count_images_in_class_dir(class_dir: str) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    count = 0
    for file_name in os.listdir(class_dir):
        p = os.path.join(class_dir, file_name)
        if os.path.isfile(p) and Path(file_name).suffix.lower() in exts:
            count += 1
    return count


# 导出数据数量的 CSV 和 LaTeX（论文排版文本）表。
# train 中包含离线增强图时，统计的是增强后的文件数，不能把它当成独立原始样本数量。
def export_dataset_statistics_table(data_root: str, task_name: str, output_dir: str) -> pd.DataFrame:
    """
    Export task-specific dataset statistics table:
      Class | Train | Val | Test | Total
    """
    split_names = ["train", "val", "test"]

    class_names = []
    train_dir = os.path.join(data_root, "train")
    for item in sorted(os.listdir(train_dir)):
        p = os.path.join(train_dir, item)
        if os.path.isdir(p):
            class_names.append(item)

    rows = []
    for cls in class_names:
        row = {"Task": task_name, "Category": cls}
        total = 0
        for split in split_names:
            class_dir = os.path.join(data_root, split, cls)
            n = count_images_in_class_dir(class_dir)
            row[split.capitalize()] = n
            total += n
        row["Total"] = total
        rows.append(row)

    df = pd.DataFrame(rows)
    save_path_csv = os.path.join(output_dir, f"table1_dataset_statistics_{task_name}.csv")
    df.to_csv(save_path_csv, index=False, encoding="utf-8-sig")

    # Also export a LaTeX-friendly table (without full table environment)
    save_path_tex = os.path.join(output_dir, f"table1_dataset_statistics_{task_name}.tex")
    with open(save_path_tex, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))

    return df


# ============================================================
# 5. Model builder
# ============================================================

# 兼容 torchvision 新旧加载接口：优先 weights=...，失败后尝试旧 pretrained 参数。
# pretrained=True 通常需要本地已有预训练权重缓存或联网下载；回退不保证解决所有版本/网络错误。
def load_torchvision_model_safely(model_name: str, pretrained: bool = True):
    """
    Compatible with old/new torchvision APIs:
    - new: weights=xxx.DEFAULT
    - old: pretrained=True
    """
    weight_enum_map = {
        "resnet18": "ResNet18_Weights",
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "efficientnet_b0": "EfficientNet_B0_Weights",
        "convnext_tiny": "ConvNeXt_Tiny_Weights",
    }

    model_fn = getattr(models, model_name)

    if pretrained:
        try:
            weight_enum = getattr(models, weight_enum_map[model_name]).DEFAULT
            model = model_fn(weights=weight_enum)
        except Exception:
            model = model_fn(pretrained=True)
    else:
        try:
            model = model_fn(weights=None)
        except Exception:
            model = model_fn(pretrained=False)

    return model



# ---------------- MambaVision based project models ----------------
# MambaVision is not part of torchvision. The two project models below use
# HuggingFace remote code from NVIDIA's official MambaVision release.
# Install before use:
#   pip install transformers huggingface_hub mambavision
# If your machine cannot access HuggingFace, download/cache the model first or
# set environment variable MAMBAVISION_MODEL_ID to a local model directory.


# 读取 MAMBAVISION_MODEL_ID 环境变量；未设置时用默认模型标识。
# 这只影响 MambaVision 备用分支，不影响目前的 RepViT 训练。
def get_mambavision_model_id() -> str:
    """Return HF model id or local directory for MambaVision-T."""
    return os.environ.get("MAMBAVISION_MODEL_ID", "nvidia/MambaVision-T-1K")


# 按需导入 MambaVision 所需的 transformers；只在选择该模型时调用。
# 缺失依赖会提示报错；不要为了当前 RepViT 流程额外安装不需要的模型依赖。
def _import_transformers_for_mambavision():
    try:
        from transformers import AutoModel, AutoModelForImageClassification  # type: ignore
        return AutoModel, AutoModelForImageClassification
    except Exception as exc:
        raise ImportError(
            "MambaVision models require transformers/HuggingFace support. "
            "Please install dependencies first: pip install transformers huggingface_hub mambavision"
        ) from exc


# 备用模型：把 MambaVision 的分类输出接入统一训练接口；当前 repvit_v4 不经过这里。
# 注意其构造函数实际使用 from_pretrained，传入 pretrained=False 也没有切换成从零初始化。
class MambaVisionTClassifier(nn.Module):
    """
    MambaVision-T baseline classifier.

    This wrapper keeps the rest of the benchmark code unchanged by returning
    logits directly. The classification head is resized to num_classes.
    """
    # 加载外部预训练模型并指定类别数；trust_remote_code=True 允许执行模型仓库代码，应只使用可信来源。
    # 这里的 pretrained 形参未控制 from_pretrained 调用；不要据此声称该分支完全离线或从零训练。
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        _, AutoModelForImageClassification = _import_transformers_for_mambavision()
        model_id = get_mambavision_model_id()

        # ignore_mismatched_sizes lets HF replace the original 1000-class head.
        self.model = AutoModelForImageClassification.from_pretrained(
            model_id,
            trust_remote_code=True,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    # forward（前向计算：图片数字经过网络变成分类分数），兼容不同模型返回格式。
    # 统一返回 logits，即尚未经过 Softmax 的各类别原始分数；训练损失直接接收它。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if hasattr(out, "logits"):
            return out.logits
        if isinstance(out, dict) and "logits" in out:
            return out["logits"]
        if isinstance(out, (tuple, list)):
            return out[0]
        raise RuntimeError("Unsupported MambaVision classifier output format.")


# 常用特征提取积木：卷积 Conv2d（学习局部图案）→ BatchNorm（调整各通道数值分布）→ SiLU（非线性变换）。
# 卷积核、步幅、通道数会影响特征尺寸与计算量，改动后要核对后续模块的输入形状。
class ConvBNAct(nn.Module):
    # 组装卷积、归一化和激活层；卷积后有 BatchNorm，因此卷积不额外设置偏置。
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    # 顺序执行 self.block，返回新的特征图（保留空间位置的一组数字图层）。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# MambaVision v2 备用分支的多尺度融合：把不同层、不同分辨率的特征组合到一起。
# 横向 1×1 卷积先统一通道数，随后从低分辨率向高分辨率传递信息，最后加权融合。
class DefectAwarePyramidFusion(nn.Module):
    """
    Defect-aware weighted pyramid fusion module.

    It fuses four-stage hierarchical MambaVision features. Shallow stages retain
    fine defect texture; deep stages provide semantic context. Learnable fusion
    weights reduce manual scale weighting.
    """
    # in_channels 是各输入层的通道列表；out_channels 是融合后通道数。
    # 若替换骨干网络（主要负责提取图像特征的网络），必须重新确认层数与每层通道。
    def __init__(self, in_channels: List[int], out_channels: int = 192):
        super().__init__()
        self.out_channels = out_channels
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(c, out_channels, kernel_size=1, padding=0) for c in in_channels
        ])
        self.smooth_convs = nn.ModuleList([
            ConvBNAct(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels
        ])

        # Three top-down fusion steps: C5->C4, P4->C3, P3->C2.
        self.fusion_weights = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(len(in_channels) - 1)
        ])

        # Multi-level aggregation weights for final fused map.
        self.level_weights = nn.Parameter(torch.ones(len(in_channels), dtype=torch.float32))
        self.eps = 1e-4

    # 先把权重变为非负并按总和归一化，再把高层特征插值到低层的宽高后相加。
    # eps 是防止分母为零的小数；插值只调整空间尺寸，不会自动修正通道不匹配。
    def _weighted_add(self, x_low: torch.Tensor, x_high: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        w = torch.relu(weight)
        w = w / (w.sum() + self.eps)
        x_high = torch.nn.functional.interpolate(
            x_high,
            size=x_low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return w[0] * x_low + w[1] * x_high

    # 逐层映射→自顶向下融合→平滑→统一分辨率→加权求和。
    # 输入特征数量必须与 lateral 层数量一致，不能随意减少其中一项。
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.lateral_convs):
            raise ValueError(f"Expected {len(self.lateral_convs)} feature maps, got {len(features)}.")

        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # Top-down weighted pathway.
        results = [None] * len(laterals)
        results[-1] = laterals[-1]
        for i in range(len(laterals) - 2, -1, -1):
            weight_idx = len(laterals) - 2 - i
            results[i] = self._weighted_add(laterals[i], results[i + 1], self.fusion_weights[weight_idx])

        results = [conv(x) for conv, x in zip(self.smooth_convs, results)]

        # Fuse all pyramid levels to the highest spatial resolution stage.
        target_size = results[0].shape[-2:]
        level_w = torch.relu(self.level_weights)
        level_w = level_w / (level_w.sum() + self.eps)
        fused = 0.0
        for i, x in enumerate(results):
            if x.shape[-2:] != target_size:
                x = torch.nn.functional.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            fused = fused + level_w[i] * x
        return fused


# 轻量注意力：让网络学习“哪些通道、哪些空间位置更值得关注”的权重。
# 这里仍然输出特征图，并不输出缺陷坐标或边界框。
class LightMultiScaleAttention(nn.Module):
    """
    Lightweight multi-scale attention for fused defect features.

    The module combines channel attention and multi-kernel spatial attention.
    It is intentionally small, because the project target is classification
    accuracy with limited additional computation rather than heavy detection.
    """
    # 通道注意力先汇总全图，再生成通道权重；空间部分用 3/5/7 卷积核观察不同邻域。
    # reduction 控制中间通道压缩程度；减小会增加中间宽度和计算量。
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        # reduction 是通道压缩倍数；必须大于 0；中间至少保留 16 通道。
        hidden = max(channels // reduction, 16)
        self.channel_mlp = nn.Sequential(
            # 把任意允许的特征高/宽平均成 1×1；每个通道保留一个全图摘要数值。
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        branch_channels = max(channels // 4, 16)
        self.reduce = ConvBNAct(channels, branch_channels, kernel_size=1, padding=0)
        self.spatial_3 = nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1, groups=branch_channels, bias=False)
        self.spatial_5 = nn.Conv2d(branch_channels, branch_channels, kernel_size=5, padding=2, groups=branch_channels, bias=False)
        self.spatial_7 = nn.Conv2d(branch_channels, branch_channels, kernel_size=7, padding=3, groups=branch_channels, bias=False)
        self.spatial_proj = nn.Sequential(
            nn.Conv2d(branch_channels * 3, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    # 先按通道加权，再按空间加权，并加回原输入（残差连接，便于保留原信息）。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_attn = self.channel_mlp(x)
        x_c = x * channel_attn

        z = self.reduce(x_c)
        s3 = self.spatial_3(z)
        s5 = self.spatial_5(z)
        s7 = self.spatial_7(z)
        spatial_attn = self.spatial_proj(torch.cat([s3, s5, s7], dim=1))
        return x_c * spatial_attn + x


# 备用改进模型：MambaVision 特征→多尺度融合→注意力→分类头。
# 目前 RepViT 路线无需先读懂它；此处仍补注释，便于以后选择模型对照。
class MambaVisionTV2DefectNet(nn.Module):
    """
    MambaVision-T_v2 project model:
      MambaVision-T backbone
      + Defect-aware weighted pyramid fusion
      + Lightweight multi-scale attention
      + 4-class classification head

    The official MambaVision feature extractor returns four hierarchical stage
    features. For MambaVision-T, their documented channels are [80, 160, 320, 640].
    """
    # 加载外部特征模型并创建融合、注意力和分类头；feature_channels 是该骨干的假定通道数。
    # 换模型必须核对这些通道；这里同样直接调用 from_pretrained，未按 pretrained 形参切换初始化。
    def __init__(self, num_classes: int, pretrained: bool = True, pyramid_channels: int = 192, dropout: float = 0.20):
        super().__init__()
        AutoModel, _ = _import_transformers_for_mambavision()
        model_id = get_mambavision_model_id()
        self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)

        self.feature_channels = [80, 160, 320, 640]
        self.pyramid = DefectAwarePyramidFusion(self.feature_channels, out_channels=pyramid_channels)
        self.attention = LightMultiScaleAttention(pyramid_channels, reduction=8)
        self.classifier = nn.Sequential(
            # 把任意允许的特征高/宽平均成 1×1；每个通道保留一个全图摘要数值。
            nn.AdaptiveAvgPool2d(1),
            # 从 [B,C,1,1] 展开为 [B,C]，B 批维度保持不变。
            nn.Flatten(),
            nn.LayerNorm(pyramid_channels),
            # 分类头随机置零比例：0.20 表示训练时约 20% 特征被置零；若 factory 传了值，修改这里默认值未必生效。
            nn.Dropout(dropout),
            nn.Linear(pyramid_channels, num_classes),
        )

    # 兼容不同外部模型输出结构并取末尾四层；必要时把 BHWC 转成 BCHW。
    # B=一批图片数量，C=通道数，H/W=高/宽；布局判断是针对现有模型的启发式规则，不能保证适用所有新模型。
    def _extract_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = self.backbone(x)

        # Official example: out_avg_pool, features = model(inputs)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            features = out[1]
        elif isinstance(out, dict):
            features = out.get("features", None) or out.get("hidden_states", None)
        elif hasattr(out, "features"):
            features = out.features
        elif hasattr(out, "hidden_states"):
            features = out.hidden_states
        else:
            raise RuntimeError("Unsupported MambaVision feature output format.")

        if features is None:
            raise RuntimeError("MambaVision backbone did not return hierarchical features.")

        features = list(features)
        if len(features) < 4:
            raise RuntimeError(f"Expected at least 4 stage features from MambaVision, got {len(features)}.")

        # Use the last four stage features. Ensure BCHW layout.
        selected = features[-4:]
        formatted = []
        for f in selected:
            if f.dim() != 4:
                raise RuntimeError(f"Expected 4D feature map, got shape {tuple(f.shape)}.")
            # Some backbones may return BHWC. Convert to BCHW when needed.
            if f.shape[1] not in self.feature_channels and f.shape[-1] in self.feature_channels:
                f = f.permute(0, 3, 1, 2).contiguous()
            formatted.append(f)
        return formatted

    # 输入图片依次经过特征提取、融合、注意力和分类头，输出每张图片的类别分数。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._extract_features(x)
        fused = self.pyramid(features)
        enhanced = self.attention(fused)
        logits = self.classifier(enhanced)
        return logits



# ---------------- RepViT based project models ----------------
# RepViT is used through timm. Recommended dependency:
#   pip install -U timm huggingface_hub safetensors
# Default backbone can be changed without editing code:
#   Windows: set REPVIT_MODEL_NAME=repvit_m0_9.dist_300e_in1k
#   Linux  : export REPVIT_MODEL_NAME=repvit_m0_9.dist_300e_in1k
# If local timm does not include RepViT, you can use the HuggingFace-hub name:
#   set REPVIT_MODEL_NAME=hf_hub:timm/repvit_m0_9.dist_300e_in1k


# RepViT 骨干名称入口：环境变量 REPVIT_MODEL_NAME 优先，否则使用默认 repvit_m0_9.dist_300e_in1k。
# 权重中的 model_name=repvit_v4 并不能完整记录这个环境变量；更换骨干后评估与展示也必须保持一致。
def get_repvit_model_name() -> str:
    """Return timm model name or hf_hub identifier for RepViT."""
    return os.environ.get("REPVIT_MODEL_NAME", "repvit_m0_9.dist_300e_in1k")


# timm 是提供现成视觉模型的库：先按本地注册名称创建，再尝试 Hugging Face 仓库名称。
# 创建失败时第二次尝试仍可能下载配置/权重；离线使用需要相应依赖和缓存已就绪。
def create_timm_model_with_fallback(model_name: str, pretrained: bool, **kwargs):
    """
    Create a timm model. If the local timm registry does not contain the model,
    retry with hf_hub:timm/<model_name>. This keeps the script usable across
    different timm versions.
    """
    try:
        return timm.create_model(model_name, pretrained=pretrained, **kwargs)
    except Exception as first_exc:
        if model_name.startswith("hf_hub:"):
            raise first_exc
        hf_name = f"hf_hub:timm/{model_name}"
        try:
            return timm.create_model(hf_name, pretrained=pretrained, **kwargs)
        except Exception as second_exc:
            raise RuntimeError(
                "Failed to create RepViT model with timm. Tried both:\n"
                f"  1) {model_name}\n"
                f"  2) {hf_name}\n"
                "Please install/update timm, or set REPVIT_MODEL_NAME to a valid local/HF model name.\n"
                "Example: set REPVIT_MODEL_NAME=hf_hub:timm/repvit_m0_9.dist_300e_in1k"
            ) from second_exc


# 原始 RepViT 对照模型，直接使用 timm 的分类网络，不加 DSK 或 DSG。
# 与 repvit_v4 的权重结构不同，不能只改名称就混用训练好的权重。
class RepViTClassifier(nn.Module):
    """
    Baseline RepViT classifier.

    Default backbone: repvit_m0_9.dist_300e_in1k. You can switch to a different
    RepViT variant with REPVIT_MODEL_NAME, e.g. repvit_m1_0.dist_300e_in1k.
    """
    # 把输出类别数改为当前数据集类别数；pretrained 控制此分支是否请求预训练权重。
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        model_name = get_repvit_model_name()
        self.model = create_timm_model_with_fallback(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )

    # 直接调用完整 RepViT，输出 [B, num_classes] 的 logits。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# DSG（通过两路特征逐元素相乘，增强相互关联信息的模块）。
# 不是人工规则检测器，也不产生框；它是在训练中学习如何调整特征的网络层。
class DefectStarGate(nn.Module):
    """
    DSG: Defect Star-Gated feature interaction module.

    It uses two light 1x1 branches and element-wise multiplication to enhance
    high-order interactions between local defect edges, gray-level changes, and
    background pattern structures. It is designed for feature maps, not vectors.
    """
    # 两路 1×1 卷积先扩展通道再交互，proj 将通道恢复；expansion 越大通常计算与参数越多。
    # gamma 是可学习的残差缩放系数，初值 0，使模块开始时等同于直接传递输入。
    # 此处 dropout 是 DSG 内部的随机置零比例，与最终分类头的 dropout 是两个位置。
    def __init__(self, channels: int, expansion: float = 2.0, dropout: float = 0.0):
        super().__init__()
        # DSG 中间宽度至少为原通道数；改 expansion 需要在 RepViTProjectNet 的构造调用处改，那里显式写了 2.0。
        hidden = max(int(channels * expansion), channels)
        # BatchNorm2d 按通道处理一批特征；batch 太小/特征太小时某些训练形状可能不满足其统计要求。
        self.norm = nn.BatchNorm2d(channels)
        # 第一路投影；1×1 卷积主要混合通道，默认步幅 1 不改变特征宽高。
        self.branch_a = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        # 第二路独立学习一套参数；虽然结构相同，但不是第一路参数的复制共享。
        self.branch_b = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.proj = nn.Sequential(
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # A learnable residual scale makes the module stable at the beginning.
        self.gamma = nn.Parameter(torch.zeros(1))

    # 归一化后分成 a、b 两路，相同位置逐元素相乘，再投影并执行 x + gamma*z。
    # 乘法要求两路形状相同；残差相加还要求输出与原 x 的通道和宽高相同。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.branch_a(z) * self.branch_b(z)
        z = self.proj(z)
        return x + self.gamma * z


# DSK（让网络为不同大小邻域的特征学习选择权重的模块）。
# 三个分支分别提取不同感受野的信息；感受野就是某个特征点能“看到”的原图邻域范围。
class DefectSelectiveKernel(nn.Module):
    """
    DSK: Defect Selective Kernel context module.

    The local branch focuses on fine defect texture, while the large/dilated
    branches provide broader pattern context. Spatially adaptive soft selection
    learns how much each receptive field should contribute at every location.
    """
    # 3×3、5×5 和膨胀 7×7 深度卷积并行；深度卷积表示各通道分别做卷积。
    # 7×7 分支 dilation=3 时有效跨度为 19，padding=9 保持宽高；只改 kernel_size 不改 padding 可能让相加报错。
    # selector 输出三个选择分数，后续 Softmax 把它们变成和为 1 的权重；gamma 初始为 0。
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        # DSK 小邻域分支：kernel_size=3，padding=1，保持输入宽高。
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        # DSK 中邻域分支：kernel_size=5，padding=2；groups=channels 表示逐通道卷积。
        self.mid_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        # DSK 大邻域分支：dilation（卷积采样点间距）为 3，有效跨度=(7-1)*3+1=19。
        self.large_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=7, padding=9, dilation=3, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # reduction 是通道压缩倍数；必须大于 0；中间至少保留 16 通道。
        hidden = max(channels // reduction, 16)
        # Produce three spatial selection maps for local/mid/large contexts.
        self.selector = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            # 3 对应 local/mid/large 三条分支，不是 RGB 通道数，也不是类别数；增减分支要同步 selector 和 forward。
            nn.Conv2d(hidden, 3, kernel_size=1, bias=True),
        )
        self.proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    # 并行提取→汇总上下文→生成每个位置的三路权重→加权融合→残差输出。
    # weights[:, 0:1] 保留通道维度，便于自动广播到全部通道；不要随意去掉 :1。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f_local = self.local_branch(x)
        f_mid = self.mid_branch(x)
        f_large = self.large_branch(x)

        context = f_local + f_mid + f_large
        # 沿 dim=1 的三路分支做归一化：同一张图、同一个空间位置的三个权重加起来为 1。
        weights = torch.softmax(self.selector(context), dim=1)  # B, 3, H, W
        fused = (
            weights[:, 0:1, :, :] * f_local
            + weights[:, 1:2, :, :] * f_mid
            + weights[:, 2:3, :, :] * f_large
        )
        fused = self.proj(fused)
        return x + self.gamma * fused


# 当前改进模型的核心：骨干最后一层特征→可选 DSK→可选 DSG→分类头。
# 版本对应：v2 只有 DSK；v3 只有 DSG；v4 两者都有。模型仍然做整图分类。
class RepViTProjectNet(nn.Module):
    """
    Project model family based on RepViT.

    Variants:
      - repvit      : original RepViT classifier, implemented separately above.
      - repvit_v2   : RepViT + DSK.
      - repvit_v3   : RepViT + DSG.
      - repvit_v4   : RepViT + DSK + DSG.

    The added modules are applied to the final stage feature map to keep the
    model efficient and easy to train for image-level four-class classification.
    """
    # features_only=True 表示骨干返回特征而不是分类结果；分类头由本工程另建。
    # 从 feature_info 获取最后层通道数，避免把通道数写死；分类头使用全局平均池化→展开→LayerNorm→Dropout→Linear。
    # 全局平均池化把每个通道的空间值取平均；Dropout 在训练时随机置零部分特征，eval 模式下关闭。
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        use_dsk: bool = False,
        use_dsg: bool = True,
        dropout: float = 0.20,
    ):
        super().__init__()
        model_name = get_repvit_model_name()
        # 骨干提取图案特征；features_only 返回中间层列表，out_indices 选择返回哪些层。
        self.backbone = create_timm_model_with_fallback(
            model_name,
            pretrained=pretrained,
            features_only=True,
            # 当前请求四层，forward 实际只使用列表最后一层；删改索引前先查所换骨干支持的层数。
            out_indices=(0, 1, 2, 3),
        )
        # [-1] 取最后一层的通道数，与 forward 选最后一层对应，不能只改其中一处。
        channels = self.backbone.feature_info.channels()[-1]
        self.feature_channels = int(channels)

        # Identity（原样传递输入）用于关闭模块，方便消融对比，即每次关闭一个模块观察效果变化。
        self.dsk = DefectSelectiveKernel(self.feature_channels) if use_dsk else nn.Identity()
        # DSG 实际参数在这里显式指定；只改 DefectStarGate 的默认 expansion/dropout 不会覆盖本行传参。
        self.dsg = DefectStarGate(self.feature_channels, expansion=2.0, dropout=0.0) if use_dsg else nn.Identity()
        self.classifier = nn.Sequential(
            # 把任意允许的特征高/宽平均成 1×1；每个通道保留一个全图摘要数值。
            nn.AdaptiveAvgPool2d(1),
            # 从 [B,C,1,1] 展开为 [B,C]，B 批维度保持不变。
            nn.Flatten(),
            # LayerNorm（对单个样本的特征做归一化）与 BatchNorm 的计算维度不同，不宜互换名字。
            nn.LayerNorm(self.feature_channels),
            # 分类头随机置零比例：0.20 表示训练时约 20% 特征被置零；若 factory 传了值，修改这里默认值未必生效。
            nn.Dropout(dropout),
            # 最后全连接层把 C 个摘要特征映射成“类别数”个 logits；改类别需重新训练匹配的分类层。
            nn.Linear(self.feature_channels, num_classes),
        )

    # 从骨干返回的多层特征中取最后一层，依次通过 DSK、DSG、分类头。
    # 输入通常 [B,3,224,224]，输出 [B,类别数]；当前未把所有层融合，也没有定位输出。
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, torch.Tensor):
            feat = features
        else:
            feat = features[-1]
        feat = self.dsk(feat)
        feat = self.dsg(feat)
        logits = self.classifier(feat)
        return logits

# 模型工厂（根据名称挑选并创建网络）：大部分改结构的入口都从这里追踪。
# 命令行 --models 传到这里；先找到自己的分支，再进入对应类修改，避免改了备用模型却看不到变化。
# 新增模型还需注册新分支；DEFAULT_MODELS 只决定省略 --models 时的默认列表。改结构后通常需重新训练。
def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    # Normalize model aliases. Both "mambavision_t_v2" and "MambaVision-T_v2" work.
    model_key = model_name.lower().replace("-", "_")

    if model_key == "resnet18":
        model = load_torchvision_model_safely("resnet18", pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif model_key == "mobilenet_v3_large":
        model = load_torchvision_model_safely("mobilenet_v3_large", pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_key == "efficientnet_b0":
        model = load_torchvision_model_safely("efficientnet_b0", pretrained)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_key == "convnext_tiny":
        model = load_torchvision_model_safely("convnext_tiny", pretrained)
        replaced = False
        for i in range(len(model.classifier) - 1, -1, -1):
            if isinstance(model.classifier[i], nn.Linear):
                in_features = model.classifier[i].in_features
                model.classifier[i] = nn.Linear(in_features, num_classes)
                replaced = True
                break
        if not replaced:
            raise RuntimeError("Could not find Linear layer in ConvNeXt classifier.")

    elif model_key == "vit_tiny":
        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes
        )

    elif model_key in {"repvit", "repvit_base", "repvit_m0_9"}:
        model = RepViTClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_key in {"repvit_v2", "repvit_dsk", "repvit_m0_9_v2"}:
        model = RepViTProjectNet(
            num_classes=num_classes,
            pretrained=pretrained,
            use_dsk=True,
            use_dsg=False,
            dropout=0.20
        )

    elif model_key in {"repvit_v3", "repvit_dsg", "repvit_m0_9_v3"}:
        model = RepViTProjectNet(
            num_classes=num_classes,
            pretrained=pretrained,
            use_dsk=False,
            use_dsg=True,
            dropout=0.20
        )

    # 【当前模型分支】这里同时启用 DSK 与 DSG；分类头 dropout=0.20 是该分支显式传入的实际值。
    # 若只想比较无 DSK/无 DSG，先用现成 repvit_v3/repvit_v2 名称，无需改代码。
    elif model_key in {"repvit_v4", "repvit_dsk_dsg", "repvit_m0_9_v4"}:
        model = RepViTProjectNet(
            num_classes=num_classes,
            pretrained=pretrained,
            use_dsk=True,
            use_dsg=True,
            dropout=0.20
        )

    elif model_key in {"mambavision_t", "mamba_vision_t", "mambavision_tiny"}:
        model = MambaVisionTClassifier(
            num_classes=num_classes,
            pretrained=pretrained
        )

    elif model_key in {"mambavision_t_v2", "mamba_vision_t_v2", "mambavision_tiny_v2"}:
        model = MambaVisionTV2DefectNet(
            num_classes=num_classes,
            pretrained=pretrained,
            pyramid_channels=192,
            dropout=0.20
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


# ============================================================
# 6. Metrics & evaluation
# ============================================================

# 计算四分类评价。Accuracy=全部图片中判对的比例；Precision=判成某类的图片中有多少是真的；Recall=真实该类中找回多少。
# F1 是 Precision 与 Recall 的调和平均；Macro-F1 先逐类计算再等权平均，Weighted-F1 按各类样本数量加权。
# zero_division=0 把无法计算的项记为 0；本函数未显式传 labels，缺失类别时宏平均口径需另外核对。
def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(acc),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted)
    }


# 识别哪个类别表示正常，用于把多种缺陷合并成“有缺陷”来评价。
# 正常类改名时优先修改匹配规则；若找不到会警告并使用编号 0，这不保证编号 0 真的代表正常。
def find_normal_class_index(class_names: List[str]) -> int:
    """
    Locate the normal class for detection-level metrics.

    For the current four-class grain task, the normal class is expected to be
    named "Normal". If no exact match is found, class index 0 is used as a
    conservative fallback, because ImageFolder sorts folders alphabetically.
    """
    normalized_names = [str(c).strip().lower() for c in class_names]

    # Preferred exact match.
    for i, name in enumerate(normalized_names):
        if name == "normal":
            return i

    # Common variants.
    variants = {"normal", "ok", "good", "negative", "non_defect", "non-defect", "no_defect", "no-defect"}
    for i, name in enumerate(normalized_names):
        key = name.replace(" ", "_")
        if key in variants:
            return i

    # Loose match.
    for i, name in enumerate(normalized_names):
        if "normal" in name:
            return i

    print("[Warning] Normal class was not found by name. Index 0 is used as the normal class.")
    return 0


# 把所有非正常类别合并为缺陷，再统计“有无缺陷”的二分类指标。
# TP=缺陷判缺陷；TN=正常判正常；FP=正常误报缺陷；FN=缺陷漏判正常。
# Defect-DR=TP/(TP+FN)，越高越好；Detect-ER=(FP+FN)/总数，越低越好。
# 缺陷 A 错判为缺陷 B 会降低四分类准确率，却不算这里的漏检；不能只报二分类结果。
def compute_detection_metrics_from_multiclass(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict[str, float]:
    """
    Compute the two project-oriented detection metrics from four-class outputs.

    Four-class outputs:
        Normal + defect type 1 + defect type 2 + defect type 3

    Detection-level binary mapping:
        prediction == Normal          -> Normal
        prediction != Normal          -> Defect
        ground truth == Normal        -> Normal
        ground truth != Normal        -> Defect

    Metrics:
        defect_detection_rate = TP / (TP + FN)
        detection_error_rate  = (FP + FN) / (TP + TN + FP + FN)

    Here TP/TN/FP/FN are binary detection-level counts after merging all
    defect classes into one "Defect" class.
    """
    normal_idx = find_normal_class_index(class_names)

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    true_defect = y_true_arr != normal_idx
    pred_defect = y_pred_arr != normal_idx

    tp = int(np.logical_and(true_defect, pred_defect).sum())
    tn = int(np.logical_and(~true_defect, ~pred_defect).sum())
    fp = int(np.logical_and(~true_defect, pred_defect).sum())
    fn = int(np.logical_and(true_defect, ~pred_defect).sum())

    n_defect = tp + fn
    n_normal = tn + fp
    n_total = tp + tn + fp + fn

    defect_detection_rate = tp / n_defect if n_defect > 0 else float("nan")
    miss_rate = fn / n_defect if n_defect > 0 else float("nan")
    false_alarm_rate = fp / n_normal if n_normal > 0 else float("nan")
    detection_error_rate = (fp + fn) / n_total if n_total > 0 else float("nan")

    return {
        "normal_class_index": int(normal_idx),
        "normal_class_name": class_names[normal_idx],
        "binary_tp": tp,
        "binary_tn": tn,
        "binary_fp": fp,
        "binary_fn": fn,
        "num_defect_samples": int(n_defect),
        "num_normal_samples": int(n_normal),
        "num_detection_errors": int(fp + fn),
        "defect_detection_rate": float(defect_detection_rate),
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm_rate),
        "detection_error_rate": float(detection_error_rate)
    }


# 评估入口：model.eval() 关闭训练专用行为；no_grad 禁止记录用于反向传播的梯度，以节省内存。
# 不更新权重。按图片数加权计算平均 loss；返回 metrics、真实标签、预测标签，可选路径和 Softmax 分数。
# 修改 return_paths/return_probs 时要同步调用端解包顺序；空数据集会导致除零等错误，应先核查目录。
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_paths: bool = False,
    return_probs: bool = False
):
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_preds = []
    all_paths = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, paths = batch
            else:
                images, labels = batch
                paths = None

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # criterion 默认返回一批平均损失；乘批内图片数再累计，避免最后小批量权重过大。
            total_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            # dim=1 是类别维，得到 [批量,类别数] 的每类分数；argmax 可以直接对 logits 求，无需先 Softmax。
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            if return_paths and paths is not None:
                all_paths.extend(list(paths))

            if return_probs:
                all_probs.extend(probs.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = float(avg_loss)

    outputs = [metrics, all_labels, all_preds]
    if return_paths:
        outputs.append(all_paths)
    if return_probs:
        outputs.append(all_probs)

    return tuple(outputs)


# ============================================================
# 7. Plot functions
# ============================================================

# 读取训练历史并保存 loss、验证 Macro-F1、准确率三种曲线。
# 只影响展示；若更改历史字典的列名，需要同步改本函数的索引键。
def plot_training_curves(log_df: pd.DataFrame, save_dir: str):
    # Loss curve
    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_loss"], label="Train Loss")
    plt.plot(log_df["epoch"], log_df["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=240)
    plt.close()

    # Macro-F1 curve
    plt.figure()
    plt.plot(log_df["epoch"], log_df["val_macro_f1"], label="Val Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_macro_f1_curve.png"), dpi=240)
    plt.close()

    # Accuracy curve
    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_acc"], label="Train Acc")
    plt.plot(log_df["epoch"], log_df["val_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=240)
    plt.close()


# 绘制混淆矩阵（行是真实类别，列是预测类别，对角线表示判对）。
# normalize=True 表示每行除以该真实类总数，便于比较类别找回比例；False 显示图片数量。
# 此函数没有显式指定 labels；若数据/预测同时缺失某类，矩阵尺寸可能小于 class_names，需核对后再扩展用途。
def plot_confusion_matrix_paper(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str,
    normalize: bool = False,
    title: Optional[str] = None
):
    """
    Paper-ready confusion matrix:
    - Times New Roman
    - 14 pt
    - light background: dark text
    - dark background: white text
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        value_format = ".4f"
    else:
        cm_display = cm
        value_format = "d"

    fig, ax = plt.subplots(figsize=(7.8, 6.8))
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=PAPER_FONT_SIZE)
    for label in cbar.ax.get_yticklabels():
        label.set_fontname(PAPER_FONT_FAMILY)
        label.set_fontsize(PAPER_FONT_SIZE)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    if title:
        ax.set_title(title)

    plt.setp(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontname=PAPER_FONT_FAMILY,
        fontsize=PAPER_FONT_SIZE
    )
    plt.setp(
        ax.get_yticklabels(),
        fontname=PAPER_FONT_FAMILY,
        fontsize=PAPER_FONT_SIZE
    )

    ax.xaxis.label.set_fontname(PAPER_FONT_FAMILY)
    ax.yaxis.label.set_fontname(PAPER_FONT_FAMILY)
    ax.xaxis.label.set_fontsize(PAPER_FONT_SIZE)
    ax.yaxis.label.set_fontsize(PAPER_FONT_SIZE)

    # Threshold based on displayed values
    thresh = np.nanmax(cm_display) / 2.0 if np.nanmax(cm_display) > 0 else 0.0

    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            if normalize:
                text = format(cm_display[i, j], value_format)
            else:
                text = format(int(cm_display[i, j]), value_format)

            ax.text(
                j, i, text,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontname=PAPER_FONT_FAMILY,
                fontsize=PAPER_FONT_SIZE
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 8. Training for one run
# ============================================================

# 一次实验=一个模型名称+一个随机种子。输出目录为 output_root/task_name/model_name/seed_N。
# 流程：读数据→建模型→训练每一轮→按验证 Macro-F1 保存最好权重→用最好权重分别评估 val/test→导出。
# 调参优先用命令行；同一目录已有 training_final_metrics.json 会直接跳过，不会检查新参数是否相同。
def train_one_run(
    model_name: str,
    seed: int,
    task_name: str,
    data_root: str,
    output_root: str,
    epochs: int,
    batch_size: int,
    num_workers: int,
    img_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
    pretrained: int,
    device: torch.device
):
    set_seed(seed)

    run_dir = os.path.join(
        output_root,
        task_name,
        model_name,
        f"seed_{seed}"
    )
    ensure_dir(run_dir)

    # 【重复实验注意】该文件存在就立即返回旧结果；改 lr/epochs 后请换 output_root 或 task_name。
    # 这里不是断点续训：不会恢复优化器、学习率调度器或上次中断的轮次。
    metrics_path = os.path.join(run_dir, "training_final_metrics.json")
    if os.path.exists(metrics_path):
        print(f"[Skip] Existing final training result found: {task_name} | {model_name} | seed={seed}")
        return load_json(metrics_path)

    print("\n" + "=" * 90)
    print(f"Task       : {task_name}")
    print(f"Model      : {model_name}")
    print(f"Seed       : {seed}")
    print(f"Output Dir : {run_dir}")
    print("=" * 90)

    train_loader, val_loader, test_loader, data_info = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        test_with_paths=False
    )

    num_classes = data_info["num_classes"]
    class_names = data_info["class_names"]

    print(f"Classes    : {class_names}")
    print(f"Train/Val/Test: {data_info['train_size']} / {data_info['val_size']} / {data_info['test_size']}")

    save_json(data_info, os.path.join(run_dir, "data_info.json"))

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=bool(pretrained)
    ).to(device)

    param_count = count_trainable_params(model)

    # CrossEntropyLoss（交叉熵损失）衡量分类分数与真实类别差多远；越小通常表示拟合越好。
    # 它内部已处理 logits，不要在模型末尾先加 Softmax 再把概率送进来。
    criterion = nn.CrossEntropyLoss()

    # AdamW（根据梯度调整参数的更新方法）：lr 是更新步幅，weight_decay 是抑制权重过大的正则项。
    # 学习率更大不一定更快收敛，过大可能训练不稳定；调参时先只改一项并保留对照。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 余弦学习率调度：随训练轮次平滑降低学习率，T_max=epochs。
    # 每轮 scheduler.step() 后才记录 lr，因此日志里的 lr 对应更新后的值。
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    # AMP（自动混合精度）在 CUDA 上用部分低精度计算节省显存；GradScaler 避免梯度数值太小。
    # 这里使用旧 torch.cuda.amp 接口，FutureWarning 通常是接口迁移提醒；本轮注释不替换接口。
    use_amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    epochs_without_improvement = 0

    history = []
    total_start_time = time.time()

    # Epoch（一轮）表示训练集完整经过网络一次；epochs 是最多轮数，早停可能提前结束。
    for epoch in range(1, epochs + 1):
        # 进入训练模式，使 Dropout/BatchNorm 使用训练行为；评估 evaluate 内部会切为 eval，下一轮这里切回来。
        model.train()

        epoch_start = time.time()
        running_loss = 0.0
        all_train_labels = []
        all_train_preds = []

        pbar = tqdm(
            train_loader,
            desc=f"[{model_name}][seed={seed}] Epoch {epoch}/{epochs}",
            leave=False
        )

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 清空上一批的梯度（损失对参数的变化方向），避免意外累加到本批。
            optimizer.zero_grad(set_to_none=True)

            # 前向：图片→网络分数→损失；这时尚未修改网络参数。
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # 反向传播计算每个参数该往哪个方向调整；step 才真正更新，update 调整混合精度缩放。
            # 修改损失函数或加正则项，需在 backward 前组合成同一个 loss。
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 按该批实际图片数累计损失，最后除以总样本数；末批不足 batch_size 时仍能正确加权。
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_train_labels.extend(labels.cpu().numpy().tolist())
            all_train_preds.extend(preds.detach().cpu().numpy().tolist())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # 本工程每一轮结束更新一次学习率；移到每个 batch 会改变调度节奏，不能认为是相同实验。
        scheduler.step()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_train_labels, all_train_preds)

        val_metrics, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_time_sec": float(epoch_time)
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss {train_loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val Macro-F1 {val_metrics['macro_f1']:.4f}"
        )

        # Select best model by validation Macro-F1
        # 验证集用于决定保留哪一轮；严格大于才算提升，分数持平也会累计未提升轮数。
        # 最好权重来自这一轮，不一定来自训练的最后一轮。
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            # 深拷贝参数防止后续更新影响“最佳权重”的内存副本；不能随意去掉 deepcopy。
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            # checkpoint（权重存档）在这里保存模型名称、类别顺序和参数数值；同目录 best_model.pth 会被新最佳轮覆盖。
            # 当前未保存 img_size、优化器状态、完整命令或 REPVIT_MODEL_NAME；请另记实验配置。
            # 同一实验目录只保留当前最好 best_model.pth，不会自动为每一轮另存完整历史权重。
            torch.save(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "epoch": epoch,
                    "best_val_macro_f1": best_val_f1,
                    "class_names": class_names,
                    "state_dict": best_state
                },
                os.path.join(run_dir, "best_model.pth")
            )
        else:
            epochs_without_improvement += 1

        # 早停：连续 patience 轮验证 Macro-F1 没有严格提高就停止；patience 应设置为正整数。
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}.")
            break

    total_train_time = time.time() - total_start_time

    log_df = pd.DataFrame(history)
    log_df.to_csv(os.path.join(run_dir, "train_log.csv"), index=False, encoding="utf-8-sig")

    plot_training_curves(log_df, run_dir)

    # Best checkpoint evaluation on both validation and test sets.
    if best_state is None:
        raise RuntimeError("No best checkpoint was selected. Please check training/validation data.")

    # 恢复验证表现最好的参数，再分别算 val/test；最终报告不是最后一轮当前参数的成绩。
    model.load_state_dict(best_state)
    best_ckpt_path = os.path.join(run_dir, "best_model.pth")

    # Rebuild val/test loaders with image paths so that misclassified samples can be exported.
    val_eval_loader, _ = build_eval_loader_for_split(
        data_root=data_root,
        split="val",
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        with_paths=True,
        expected_class_names=class_names,
    )
    test_eval_loader, _ = build_eval_loader_for_split(
        data_root=data_root,
        split="test",
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        with_paths=True,
        expected_class_names=class_names,
    )

    final_eval_root = os.path.join(run_dir, "final_eval")
    val_result = evaluate_and_export_split(
        model=model,
        dataloader=val_eval_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        output_dir=os.path.join(final_eval_root, "val"),
        split="val",
        task_name=task_name,
        model_name=model_name,
        seed=seed,
        trainable_params=param_count,
        checkpoint_path=best_ckpt_path,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        dataset_sizes={
            "train_size": data_info["train_size"],
            "val_size": data_info["val_size"],
            "test_size": data_info["test_size"],
        },
    )
    test_result = evaluate_and_export_split(
        model=model,
        dataloader=test_eval_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        output_dir=os.path.join(final_eval_root, "test"),
        split="test",
        task_name=task_name,
        model_name=model_name,
        seed=seed,
        trainable_params=param_count,
        checkpoint_path=best_ckpt_path,
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_f1,
        dataset_sizes={
            "train_size": data_info["train_size"],
            "val_size": data_info["val_size"],
            "test_size": data_info["test_size"],
        },
    )

    # Combined validation/test metrics table in the run root.
    pd.DataFrame([val_result, test_result]).to_csv(
        os.path.join(run_dir, "final_eval_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Backward-compatible metrics JSON used by existing aggregation code.
    # 保存兼容旧表格的最终结果：带 val_ 前缀的是验证集；test_ 和下方检测级统计来自测试集。
    # 不要把不同集合的 FP/FN、错误总数混在同一结论中。
    final_metrics = {
        "task_name": task_name,
        "model_name": model_name,
        "seed": seed,
        "num_classes": num_classes,
        "class_names": class_names,
        "train_size": data_info["train_size"],
        "val_size": data_info["val_size"],
        "test_size": data_info["test_size"],
        "trainable_params": int(param_count),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "checkpoint_path": best_ckpt_path,
        "val_loss": val_result["loss"],
        "val_accuracy": val_result["accuracy"],
        "val_macro_precision": val_result["macro_precision"],
        "val_macro_recall": val_result["macro_recall"],
        "val_macro_f1": val_result["macro_f1"],
        "val_weighted_precision": val_result["weighted_precision"],
        "val_weighted_recall": val_result["weighted_recall"],
        "val_weighted_f1": val_result["weighted_f1"],
        "test_loss": test_result["loss"],
        "test_accuracy": test_result["accuracy"],
        "test_macro_precision": test_result["macro_precision"],
        "test_macro_recall": test_result["macro_recall"],
        "test_macro_f1": test_result["macro_f1"],
        "test_weighted_precision": test_result["weighted_precision"],
        "test_weighted_recall": test_result["weighted_recall"],
        "test_weighted_f1": test_result["weighted_f1"],
        # Detection-level metrics below are from TEST split, to preserve old table semantics.
        "normal_class_index": test_result["normal_class_index"],
        "normal_class_name": test_result["normal_class_name"],
        "binary_tp": test_result["binary_tp"],
        "binary_tn": test_result["binary_tn"],
        "binary_fp": test_result["binary_fp"],
        "binary_fn": test_result["binary_fn"],
        "num_defect_samples": test_result["defect_total"],
        "num_normal_samples": test_result["normal_total"],
        "num_detection_errors": test_result["num_detection_errors"],
        "defect_detection_rate": test_result["defect_detection_rate"],
        "detection_error_rate": test_result["detection_error_rate"],
        "miss_rate": test_result["miss_rate"],
        "false_alarm_rate": test_result["false_alarm_rate"],
        "four_class_error_num": test_result["four_class_error_num"],
        "total_train_time_sec": float(total_train_time),
        "total_train_time_readable": format_seconds(total_train_time),
    }

    save_json(final_metrics, metrics_path)
    # Keep this old name so that existing aggregation scripts still work.
    save_json(final_metrics, os.path.join(run_dir, "test_metrics.json"))
    save_json(val_result, os.path.join(run_dir, "val_metrics.json"))

    print("\n[Final Best-Checkpoint Result]")
    print(f"Task                : {task_name}")
    print(f"Model               : {model_name}")
    print(f"Seed                : {seed}")
    print(f"Best Epoch          : {best_epoch}")
    print(f"Val  Accuracy/F1    : {val_result['accuracy']:.4f} / {val_result['macro_f1']:.4f}")
    print(f"Test Accuracy/F1    : {test_result['accuracy']:.4f} / {test_result['macro_f1']:.4f}")
    print(f"Test Defect DR      : {test_result['defect_detection_rate']:.6f}")
    print(f"Test Detect ER      : {test_result['detection_error_rate']:.6f}")
    print(f"Test TP/TN/FP/FN    : {test_result['binary_tp']} / {test_result['binary_tn']} / {test_result['binary_fp']} / {test_result['binary_fn']}")
    print(f"Final eval outputs  : {final_eval_root}")
    print(f"Training Time       : {final_metrics['total_train_time_readable']}")

    return final_metrics


# ============================================================
# 9. Aggregation: Paper Table 2 and best-model selection
# ============================================================

# 收集各个 seed 的 test_metrics.json，每次实验一行；缺失的文件只警告并跳过。
# 现有目录中若混有旧参数的结果，会一起读取，因此每套配置应有独立 output_root 或 task_name。
def collect_all_seed_results(output_root: str, task_name: str, models_to_run: List[str], seeds: List[int]) -> pd.DataFrame:
    """
    Collect one row per seed. No mean/std aggregation is performed here.

    The table is designed for the current four-class task:
      - normal class is kept as Normal
      - all defect classes are merged only when calculating project metrics
    """
    all_rows = []

    for model_name in models_to_run:
        for seed in seeds:
            metrics_path = os.path.join(
                output_root,
                task_name,
                model_name,
                f"seed_{seed}",
                "test_metrics.json"
            )

            if not os.path.exists(metrics_path):
                print(f"[Warning] Missing result file: {metrics_path}")
                continue

            metrics = load_json(metrics_path)

            row = {
                "task_name": task_name,
                "model_name": model_name,
                "seed": seed,
                "trainable_params": metrics["trainable_params"],
                "best_epoch": metrics["best_epoch"],

                # Four-class recognition metrics
                "test_accuracy": metrics["test_accuracy"],
                "test_macro_precision": metrics["test_macro_precision"],
                "test_macro_recall": metrics["test_macro_recall"],
                "test_macro_f1": metrics["test_macro_f1"],
                "test_weighted_precision": metrics["test_weighted_precision"],
                "test_weighted_recall": metrics["test_weighted_recall"],
                "test_weighted_f1": metrics["test_weighted_f1"],

                # Project-oriented detection metrics after merging defect classes
                "defect_detection_rate": metrics.get("defect_detection_rate", np.nan),
                "detection_error_rate": metrics.get("detection_error_rate", np.nan),
                "miss_rate": metrics.get("miss_rate", np.nan),
                "false_alarm_rate": metrics.get("false_alarm_rate", np.nan),
                "num_defect_samples": metrics.get("num_defect_samples", np.nan),
                "num_normal_samples": metrics.get("num_normal_samples", np.nan),
                "num_detection_errors": metrics.get("num_detection_errors", np.nan),
                "binary_tp": metrics.get("binary_tp", np.nan),
                "binary_tn": metrics.get("binary_tn", np.nan),
                "binary_fp": metrics.get("binary_fp", np.nan),
                "binary_fn": metrics.get("binary_fn", np.nan),

                "total_train_time_sec": metrics["total_train_time_sec"]
            }
            all_rows.append(row)

    if len(all_rows) == 0:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values(by=["model_name", "seed"]).reset_index(drop=True)
    return df


# 当前主流程使用的汇总：每次实验一行、每模型一表，按测试 Macro-F1 排序；没有跨 seed 均值/标准差。
# 测试集排名仅用于描述已有结果；若反复据此选模型再报告同一测试集成绩，会使评估偏乐观，正式选型应依据验证集。
def aggregate_task_results(output_root: str, task_name: str, models_to_run: List[str], seeds: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Export per-seed result tables.

    Different from the previous version, this function does NOT output mean/std
    across seeds. It saves:
      1) one combined table for all models and all seeds;
      2) one per-model table after that model's seeds are finished;
      3) one single-run ranking table sorted by Test Macro-F1.
    """
    detail_df = collect_all_seed_results(output_root, task_name, models_to_run, seeds)

    if detail_df.empty:
        print("[Warning] No results available for aggregation.")
        return detail_df, pd.DataFrame()

    task_dir = os.path.join(output_root, task_name)
    ensure_dir(task_dir)

    detail_save_path = os.path.join(task_dir, f"summary_{task_name}_all_models_per_seed.csv")
    detail_df.to_csv(detail_save_path, index=False, encoding="utf-8-sig")

    metric_cols = [
        "task_name", "model_name", "seed", "trainable_params", "best_epoch",
        "test_accuracy", "test_macro_precision", "test_macro_recall", "test_macro_f1",
        "test_weighted_f1",
        "defect_detection_rate", "detection_error_rate",
        "miss_rate", "false_alarm_rate",
        "num_defect_samples", "num_normal_samples", "num_detection_errors",
        "binary_tp", "binary_tn", "binary_fp", "binary_fn",
        "total_train_time_sec"
    ]
    metric_cols = [c for c in metric_cols if c in detail_df.columns]

    # Save one table per model. This is the key output requested:
    # after one model has completed its seeds, the three seed results are summarized.
    for model_name, group in detail_df.groupby("model_name", sort=False):
        model_df = group[metric_cols].sort_values(by="seed").reset_index(drop=True)
        model_save_path = os.path.join(task_dir, f"summary_{task_name}_{model_name}_per_seed.csv")
        model_df.to_csv(model_save_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 90)
        print(f"[{task_name} | {model_name}] Per-seed result table saved:")
        print(model_save_path)
        display_cols = [
            "seed", "best_epoch",
            "test_accuracy", "test_macro_f1",
            "defect_detection_rate", "detection_error_rate",
            "binary_fp", "binary_fn", "num_detection_errors"
        ]
        display_cols = [c for c in display_cols if c in model_df.columns]
        print(model_df[display_cols].to_string(index=False))
        print("=" * 90)

    ranking_df = detail_df.sort_values(by="test_macro_f1", ascending=False).reset_index(drop=True)
    ranking_save_path = os.path.join(task_dir, f"summary_{task_name}_single_run_ranking.csv")
    ranking_df.to_csv(ranking_save_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"[{task_name}] Per-seed aggregation outputs saved:")
    print(detail_save_path)
    print(ranking_save_path)
    print("=" * 90)

    return detail_df, ranking_df


# 旧版论文表格导出，要求 accuracy_mean、accuracy_std、params 等旧汇总字段。
# 当前主流程不调用它，当前逐 seed 排名表也不具备这些字段；不要直接把新表传进来，否则会 KeyError。
def export_paper_table2_benchmark(summary_df: pd.DataFrame, task_name: str, output_dir: str) -> pd.DataFrame:
    """
    Export task-specific paper benchmark table.
    Best row is marked in CSV and bolded in LaTeX.
    """
    rows = []
    for idx, row in summary_df.iterrows():
        rows.append({
            "Model": row["model_name"],
            "Params": int(row["params"]),
            "Accuracy": num_to_mean_std_str(row["accuracy_mean"], row["accuracy_std"]),
            "Macro-P": num_to_mean_std_str(row["macro_precision_mean"], row["macro_precision_std"]),
            "Macro-R": num_to_mean_std_str(row["macro_recall_mean"], row["macro_recall_std"]),
            "Macro-F1": num_to_mean_std_str(row["macro_f1_mean"], row["macro_f1_std"]),
            "Best": "Yes" if idx == 0 else ""
        })

    df = pd.DataFrame(rows)

    csv_path = os.path.join(output_dir, f"table2_benchmark_{task_name}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # LaTeX version with bold best row
    tex_path = os.path.join(output_dir, f"table2_benchmark_{task_name}.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Params & Accuracy & Macro-P & Macro-R & Macro-F1 \\\\\n")
        f.write("\\midrule\n")
        for idx, row in df.iterrows():
            vals = [
                str(row["Model"]),
                str(row["Params"]),
                str(row["Accuracy"]),
                str(row["Macro-P"]),
                str(row["Macro-R"]),
                str(row["Macro-F1"]),
            ]
            if idx == 0:
                vals = [f"\\textbf{{{v}}}" for v in vals]
            f.write(" & ".join(vals) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    return df


# ============================================================
# 10. Best-model final analysis:
#     - confusion matrix
#     - misclassified samples
# ============================================================

# 旧版最优模型选择辅助函数；当前 main 没有调用。
# 原文档字符串称“跨 seed 均值选架构”，实际依赖传入 summary_df 的第一行，再按测试 Macro-F1 选 seed。
# 当前汇总返回逐次实验排名，不能把它误称为跨 seed 均值；测试参与选优也会污染最终独立评价。
def select_best_architecture_and_seed(detail_df: pd.DataFrame, summary_df: pd.DataFrame) -> Dict:
    """
    Rule:
    1) Best architecture = highest mean Macro-F1 across seeds.
    2) Representative/best checkpoint = highest single-run Test Macro-F1
       within that best architecture.
    """
    if summary_df.empty or detail_df.empty:
        raise RuntimeError("Cannot select best model from empty results.")

    best_arch = summary_df.iloc[0]["model_name"]
    arch_runs = detail_df[detail_df["model_name"] == best_arch].copy()
    arch_runs = arch_runs.sort_values(by="test_macro_f1", ascending=False)
    best_run = arch_runs.iloc[0]

    return {
        "best_architecture": best_arch,
        "best_seed": int(best_run["seed"]),
        "best_run_test_macro_f1": float(best_run["test_macro_f1"]),
        "best_run_test_accuracy": float(best_run["test_accuracy"]),
        "selection_rule": (
            "Best architecture selected by highest mean test Macro-F1 across seeds; "
            "best checkpoint selected by highest test Macro-F1 within that architecture."
        )
    }


# 旧版分析路径：按目录定位权重、创建网络并加载 state_dict（网络各参数的数值字典）。
# 本函数会把 pretrained 继续传入建模，可能先请求预训练权重再覆盖；当前 eval-only 分支使用 pretrained=False。
def load_best_checkpoint_for_analysis(
    output_root: str,
    task_name: str,
    model_name: str,
    seed: int,
    num_classes: int,
    pretrained: int,
    device: torch.device
):
    ckpt_path = os.path.join(
        output_root,
        task_name,
        model_name,
        f"seed_{seed}",
        "best_model.pth"
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Best checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=bool(pretrained)
    ).to(device)

    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    return model, ckpt_path


# 把不适合路径的字符替换为下划线，便于生成错分目录名；不改变训练标签编号。
def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)


# 把错图复制到“真实类→预测类”的子目录，并把原路径、复制路径和复制状态写回表格。
# 复制失败不会终止全部评估，必须查看 copy_status；同名目标可能覆盖，旧目录多余文件不会自动清空。
def copy_misclassified_images(
    records_df: pd.DataFrame,
    class_names: List[str],
    output_dir: str,
    split: Optional[str] = None
):
    """
    Copy misclassified images into pair-wise folders.

    The destination file names are intentionally kept short to avoid Windows
    path-length errors. The original absolute source path is still preserved in
    the exported CSV table.
    """
    mis_root = os.path.join(output_dir, "misclassified_samples")
    if split:
        # When output_dir is already split-specific, this only appears in file names.
        split_prefix = sanitize_filename(split)
    else:
        split_prefix = "eval"
    ensure_dir(mis_root)

    copied_paths = []
    copy_status = []

    for idx, row in records_df.iterrows():
        true_label = sanitize_filename(str(row["true_label"]))
        pred_label = sanitize_filename(str(row["pred_label"]))

        pair_dir = os.path.join(mis_root, f"true_{true_label}__pred_{pred_label}")
        ensure_dir(pair_dir)

        src_path = row.get("image_path", row.get("source_path", ""))
        base_name = os.path.basename(str(src_path))
        # Shorter than the old naming rule to avoid long Windows paths.
        dst_name = f"{idx:05d}__{split_prefix}__{base_name}"
        dst_path = os.path.join(pair_dir, dst_name)

        try:
            shutil.copy2(src_path, dst_path)
            copied_paths.append(dst_path)
            copy_status.append("copied")
        except Exception as exc:
            copied_paths.append("")
            copy_status.append(f"copy_failed: {type(exc).__name__}: {exc}")
            print(f"[Warning] Failed to copy misclassified image: {src_path} -> {dst_path}. Error: {exc}")

    records_df = records_df.copy()
    records_df["copied_path"] = copied_paths
    records_df["copy_status"] = copy_status
    return records_df

# 把错分图片拼成便于人工检查的缩略图。max_images 只限制展示数量，不限制错分统计。
# T 表示真实类别，P 表示预测类别；这里的缩放只用于拼图，不改变模型评估输入。
def create_misclassified_grid(
    mis_df: pd.DataFrame,
    save_path: str,
    max_images: int = 20,
    thumb_size: int = 180
):
    """
    Create a visual grid of misclassified examples for quick inspection / paper selection.
    """
    if mis_df.empty:
        return

    subset = mis_df.head(max_images).copy()
    n = len(subset)
    cols = 4
    rows = math.ceil(n / cols)

    pad = 18
    text_h = 48
    canvas_w = cols * thumb_size + (cols + 1) * pad
    canvas_h = rows * (thumb_size + text_h) + (rows + 1) * pad

    canvas = Image.new("RGB", (canvas_w, canvas_h), color="white")
    draw = ImageDraw.Draw(canvas)

    # PIL may not find Times New Roman by name on all systems; fallback safely.
    try:
        font = ImageFont.truetype("times.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for idx, (_, row) in enumerate(subset.iterrows()):
        r = idx // cols
        c = idx % cols

        x = pad + c * thumb_size
        y = pad + r * (thumb_size + text_h)

        img_path = row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize((thumb_size, thumb_size))
        except Exception:
            img = Image.new("RGB", (thumb_size, thumb_size), color=(230, 230, 230))

        canvas.paste(img, (x, y))

        text1 = f"T: {row['true_label']}"
        text2 = f"P: {row['pred_label']}"
        draw.text((x, y + thumb_size + 4), text1, fill="black", font=font)
        draw.text((x, y + thumb_size + 24), text2, fill="black", font=font)

    canvas.save(save_path)


# 旧版论文图表总入口，保留作历史兼容；当前 main 不会走到此函数。
# 它依赖旧版均值表字段与选优方式，不适合直接接当前 aggregate_task_results 返回值。
# 当前训练已通过 evaluate_and_export_split 为每个实验分别导出混淆矩阵与错图。
def run_best_model_final_analysis(
    data_root: str,
    output_root: str,
    task_name: str,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    img_size: int,
    pretrained: int,
    device: torch.device
):
    """
    After all models/seeds finish:
    - export best-model information
    - load best checkpoint
    - re-evaluate on test set with image paths and probabilities
    - export paper-ready confusion matrices
    - export misclassified CSV + copy wrong images + make error-grid image
    """
    paper_dir = os.path.join(output_root, task_name, "paper_outputs")
    ensure_dir(paper_dir)

    selection_info = select_best_architecture_and_seed(detail_df, summary_df)
    save_json(selection_info, os.path.join(paper_dir, "best_model_selection.json"))

    # Dataset statistics table for this task (Paper Table 1 section)
    export_dataset_statistics_table(
        data_root=data_root,
        task_name=task_name,
        output_dir=paper_dir
    )

    # Benchmark table for this task (Paper Table 2 section)
    export_paper_table2_benchmark(
        summary_df=summary_df,
        task_name=task_name,
        output_dir=paper_dir
    )

    # Build dataloaders with test paths
    _, _, test_loader, data_info = build_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        test_with_paths=True
    )

    num_classes = data_info["num_classes"]
    class_names = data_info["class_names"]

    model, ckpt_path = load_best_checkpoint_for_analysis(
        output_root=output_root,
        task_name=task_name,
        model_name=selection_info["best_architecture"],
        seed=selection_info["best_seed"],
        num_classes=num_classes,
        pretrained=pretrained,
        device=device
    )

    # CrossEntropyLoss（交叉熵损失）衡量分类分数与真实类别差多远；越小通常表示拟合越好。
    # 它内部已处理 logits，不要在模型末尾先加 Softmax 再把概率送进来。
    criterion = nn.CrossEntropyLoss()

    metrics, y_true, y_pred, paths, probs = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        return_paths=True,
        return_probs=True
    )

    final_analysis_info = {
        "checkpoint_path": ckpt_path,
        "class_names": class_names,
        "test_metrics_recomputed": metrics
    }
    save_json(final_analysis_info, os.path.join(paper_dir, "best_model_final_test_metrics.json"))

    # Paper confusion matrices
    plot_confusion_matrix_paper(
        y_true, y_pred, class_names,
        os.path.join(paper_dir, f"figure_confusion_matrix_{task_name}_raw.png"),
        normalize=False,
        title=None
    )

    plot_confusion_matrix_paper(
        y_true, y_pred, class_names,
        os.path.join(paper_dir, f"figure_confusion_matrix_{task_name}_normalized.png"),
        normalize=True,
        title=None
    )

    # Misclassification records
    rows = []
    for i, (true_idx, pred_idx, path, prob_vec) in enumerate(zip(y_true, y_pred, paths, probs)):
        if true_idx != pred_idx:
            prob_arr = np.asarray(prob_vec)
            rows.append({
                "index_in_test_evaluation": i,
                "image_path": path,
                "true_index": int(true_idx),
                "pred_index": int(pred_idx),
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "pred_confidence": float(prob_arr[pred_idx]),
                "true_class_probability": float(prob_arr[true_idx])
            })

    mis_df = pd.DataFrame(rows)

    mis_csv_path = os.path.join(paper_dir, "misclassified_samples.csv")

    if mis_df.empty:
        # Save empty CSV with meaningful columns
        pd.DataFrame(columns=[
            "index_in_test_evaluation",
            "image_path",
            "true_index",
            "pred_index",
            "true_label",
            "pred_label",
            "pred_confidence",
            "true_class_probability",
            "copied_path"
        ]).to_csv(mis_csv_path, index=False, encoding="utf-8-sig")
    else:
        mis_df = mis_df.sort_values(by="pred_confidence", ascending=False).reset_index(drop=True)
        mis_df = copy_misclassified_images(mis_df, class_names, paper_dir)
        mis_df.to_csv(mis_csv_path, index=False, encoding="utf-8-sig")

        create_misclassified_grid(
            mis_df=mis_df,
            save_path=os.path.join(paper_dir, f"figure_misclassified_grid_{task_name}.png"),
            max_images=20,
            thumb_size=180
        )

    print("\n" + "=" * 90)
    print(f"[{task_name}] Paper outputs generated in:")
    print(paper_dir)
    print("Included:")
    print("- Table 1 dataset statistics")
    print("- Table 2 benchmark summary")
    print("- Best-model confusion matrices (raw & normalized)")
    print("- Misclassified sample CSV and copied error images")
    print("=" * 90)



# ============================================================
# 11. Evaluation-only mode for existing best checkpoints
# ============================================================

# 把混淆矩阵导出为可用表格软件打开的 CSV；归一化按真实类的行计算。
# 与绘图函数一样未固定 labels，全类别样本缺失时需注意矩阵形状与类名数是否相符。
def save_confusion_matrix_csv(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_dir: str,
    prefix: str
):
    """Save raw and row-normalized confusion matrices as CSV files."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    raw_path = os.path.join(save_dir, f"{prefix}_confusion_matrix_raw.csv")
    norm_path = os.path.join(save_dir, f"{prefix}_confusion_matrix_normalized.csv")

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        raw_path, encoding="utf-8-sig"
    )
    pd.DataFrame(cm_norm, index=class_names, columns=class_names).to_csv(
        norm_path, encoding="utf-8-sig"
    )

    return cm, cm_norm



# 对单张错图分类：false_alarm=正常误报；missed_defect=缺陷漏检；defect_type_confusion=缺陷类型混淆。
# 例如真实为污染、预测为探针偏移：四分类错，但“是否缺陷”仍判对。
def get_binary_error_type(true_idx: int, pred_idx: int, normal_idx: int) -> str:
    if true_idx == pred_idx:
        return "correct"
    if true_idx == normal_idx and pred_idx != normal_idx:
        return "false_alarm"
    if true_idx != normal_idx and pred_idx == normal_idx:
        return "missed_defect"
    return "defect_type_confusion"


# 把每张图片的路径、真实/预测类别、各类得分及错误类型组成表格。
# 这张表是离线排查错例的入口；各列表必须按同一顺序且长度相同，zip 不会主动提醒较长列表被截断。
def build_prediction_dataframe(
    y_true: List[int],
    y_pred: List[int],
    paths: List[str],
    probs: List[List[float]],
    class_names: List[str]
) -> pd.DataFrame:
    normal_idx = find_normal_class_index(class_names)
    rows = []
    for i, (true_idx, pred_idx, path, prob_vec) in enumerate(zip(y_true, y_pred, paths, probs)):
        prob_arr = np.asarray(prob_vec, dtype=float)
        row = {
            "index_in_evaluation": i,
            "image_path": path,
            "file_name": os.path.basename(path),
            "true_index": int(true_idx),
            "pred_index": int(pred_idx),
            "true_label": class_names[true_idx],
            "pred_label": class_names[pred_idx],
            "is_correct": bool(true_idx == pred_idx),
            "pred_confidence": float(prob_arr[pred_idx]),
            "true_class_probability": float(prob_arr[true_idx]),
            "binary_error_type": get_binary_error_type(int(true_idx), int(pred_idx), normal_idx),
        }
        for cls_idx, cls_name in enumerate(class_names):
            row[f"prob_{cls_name}"] = float(prob_arr[cls_idx])
        rows.append(row)
    return pd.DataFrame(rows)


# 逐类别导出 Precision/Recall/F1/support；support 是该类真实样本数，correct_num 是判对数。
# 这里显式指定完整标签范围，所以缺失类别也保留对应行。
def export_class_metrics_csv(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
    save_path: str
) -> pd.DataFrame:
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    correct = np.diag(cm).astype(int)
    rows = []
    for i, cls_name in enumerate(class_names):
        rows.append({
            "class_name": cls_name,
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1_score": float(f1[i]),
            "support": int(support[i]),
            "correct_num": int(correct[i]),
            "error_num": int(support[i] - correct[i]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    return df


# 统一的“评估并导出”入口，训练结束和 eval-only 共用它。
# 产生分类报告、混淆矩阵 PNG/CSV、逐图预测、错图列表/副本/拼图、总体及二分类指标。
# 它会写文件并可能覆盖同名报告；使用新目录可避免旧错图残留，不会更新模型参数。
def evaluate_and_export_split(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
    output_dir: str,
    split: str,
    task_name: str,
    model_name: str,
    seed: int,
    trainable_params: int,
    checkpoint_path: str = "",
    best_epoch: int = -1,
    best_val_macro_f1: float = float("nan"),
    dataset_sizes: Optional[Dict[str, int]] = None,
) -> Dict:
    """
    Evaluate a model on one split and export all files needed for reporting,
    later re-testing, and label-cleaning inspection.
    """
    ensure_dir(output_dir)

    metrics, y_true, y_pred, paths, probs = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        return_paths=True,
        return_probs=True,
    )

    detection_metrics = compute_detection_metrics_from_multiclass(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    # Text report and class-level CSV report.
    report_txt = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    with open(os.path.join(output_dir, f"{split}_classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_txt)

    export_class_metrics_csv(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=os.path.join(output_dir, f"{split}_classification_report.csv"),
    )

    # Confusion matrices: PNG + CSV.
    plot_confusion_matrix_paper(
        y_true, y_pred, class_names,
        os.path.join(output_dir, f"{split}_confusion_matrix_raw.png"),
        normalize=False,
        title=None,
    )
    plot_confusion_matrix_paper(
        y_true, y_pred, class_names,
        os.path.join(output_dir, f"{split}_confusion_matrix_normalized.png"),
        normalize=True,
        title=None,
    )
    save_confusion_matrix_csv(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_dir=output_dir,
        prefix=split,
    )

    # Per-sample prediction table.
    pred_df = build_prediction_dataframe(
        y_true=y_true,
        y_pred=y_pred,
        paths=paths,
        probs=probs,
        class_names=class_names,
    )
    pred_df.to_csv(
        os.path.join(output_dir, f"{split}_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Misclassified samples and copied image files.
    mis_df = pred_df[pred_df["is_correct"] == False].copy().reset_index(drop=True)
    mis_csv_path = os.path.join(output_dir, f"{split}_misclassified_samples.csv")
    if mis_df.empty:
        mis_df.to_csv(mis_csv_path, index=False, encoding="utf-8-sig")
    else:
        mis_df = mis_df.sort_values(by="pred_confidence", ascending=False).reset_index(drop=True)
        mis_df = copy_misclassified_images(mis_df, class_names, output_dir, split=split)
        mis_df.to_csv(mis_csv_path, index=False, encoding="utf-8-sig")
        create_misclassified_grid(
            mis_df=mis_df,
            save_path=os.path.join(output_dir, f"{split}_misclassified_grid.png"),
            max_images=20,
            thumb_size=180,
        )

    total_samples = int(len(y_true))
    four_class_error_num = int((np.asarray(y_true) != np.asarray(y_pred)).sum())

    # Compact metrics row for report and aggregation.
    summary_row = {
        "task_name": task_name,
        "model_name": model_name,
        "seed": int(seed),
        "eval_split": split,
        "checkpoint_path": checkpoint_path,
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_macro_f1) if not pd.isna(best_val_macro_f1) else float("nan"),
        "trainable_params": int(trainable_params),
        "total_samples": total_samples,
        "four_class_error_num": four_class_error_num,
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_precision": metrics["weighted_precision"],
        "weighted_recall": metrics["weighted_recall"],
        "weighted_f1": metrics["weighted_f1"],
        "normal_class_index": detection_metrics["normal_class_index"],
        "normal_class_name": detection_metrics["normal_class_name"],
        "normal_total": detection_metrics["num_normal_samples"],
        "defect_total": detection_metrics["num_defect_samples"],
        "binary_tp": detection_metrics["binary_tp"],
        "binary_tn": detection_metrics["binary_tn"],
        "binary_fp": detection_metrics["binary_fp"],
        "binary_fn": detection_metrics["binary_fn"],
        "missed_defect_num": detection_metrics["binary_fn"],
        "false_alarm_num": detection_metrics["binary_fp"],
        "num_detection_errors": detection_metrics["num_detection_errors"],
        "defect_detection_rate": detection_metrics["defect_detection_rate"],
        "miss_rate": detection_metrics["miss_rate"],
        "false_alarm_rate": detection_metrics["false_alarm_rate"],
        "detection_error_rate": detection_metrics["detection_error_rate"],
        "num_errors": four_class_error_num,
        f"{split}_loss": metrics["loss"],
        f"{split}_accuracy": metrics["accuracy"],
        f"{split}_macro_precision": metrics["macro_precision"],
        f"{split}_macro_recall": metrics["macro_recall"],
        f"{split}_macro_f1": metrics["macro_f1"],
        f"{split}_weighted_precision": metrics["weighted_precision"],
        f"{split}_weighted_recall": metrics["weighted_recall"],
        f"{split}_weighted_f1": metrics["weighted_f1"],
    }
    if dataset_sizes:
        summary_row.update(dataset_sizes)

    pd.DataFrame([summary_row]).to_csv(
        os.path.join(output_dir, f"{split}_metrics_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    binary_row = {
        "task_name": task_name,
        "model_name": model_name,
        "seed": int(seed),
        "eval_split": split,
        "normal_total": detection_metrics["num_normal_samples"],
        "defect_total": detection_metrics["num_defect_samples"],
        "binary_tn": detection_metrics["binary_tn"],
        "binary_fp": detection_metrics["binary_fp"],
        "binary_fn": detection_metrics["binary_fn"],
        "binary_tp": detection_metrics["binary_tp"],
        "defect_detection_rate": detection_metrics["defect_detection_rate"],
        "miss_rate": detection_metrics["miss_rate"],
        "false_alarm_rate": detection_metrics["false_alarm_rate"],
        "detection_error_rate": detection_metrics["detection_error_rate"],
        "num_detection_errors": detection_metrics["num_detection_errors"],
    }
    pd.DataFrame([binary_row]).to_csv(
        os.path.join(output_dir, f"{split}_binary_project_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_json(summary_row, os.path.join(output_dir, f"{split}_metrics.json"))

    print(
        f"[{split.upper()} Result] "
        f"Acc={metrics['accuracy']:.4f}, "
        f"Macro-F1={metrics['macro_f1']:.4f}, "
        f"Weighted-F1={metrics['weighted_f1']:.4f}, "
        f"Defect-DR={detection_metrics['defect_detection_rate']:.6f}, "
        f"Detect-ER={detection_metrics['detection_error_rate']:.6f}, "
        f"FP/FN={detection_metrics['binary_fp']}/{detection_metrics['binary_fn']}, "
        f"4-class errors={four_class_error_num}"
    )

    return summary_row

# 加载一个已有 best_model.pth 并评价指定集合，不进行训练。
# 权重路径由 checkpoint_root/task_name/model_name/seed_N 拼成；找不到只警告并跳过。
# 重建结构时 pretrained=False，再加载已有权重；类别顺序必须与权重中的 class_names 一致。
def evaluate_one_best_checkpoint_on_split(
    data_root: str,
    checkpoint_root: str,
    eval_output_root: str,
    task_name: str,
    model_name: str,
    seed: int,
    split: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    device: torch.device
) -> Optional[Dict]:
    """
    Load an existing best_model.pth and evaluate it on train/val/test.
    The export format is the same as training-time final evaluation.
    """
    ckpt_path = os.path.join(
        checkpoint_root,
        task_name,
        model_name,
        f"seed_{seed}",
        "best_model.pth"
    )

    if not os.path.exists(ckpt_path):
        print(f"[Warning] Best checkpoint not found, skipped: {ckpt_path}")
        return None

    run_dir = os.path.join(
        eval_output_root,
        task_name,
        model_name,
        f"seed_{seed}",
        split
    )
    ensure_dir(run_dir)

    print("\n" + "=" * 90)
    print(f"[Eval Only] Task       : {task_name}")
    print(f"[Eval Only] Split      : {split}")
    print(f"[Eval Only] Model      : {model_name}")
    print(f"[Eval Only] Seed       : {seed}")
    print(f"[Eval Only] Checkpoint : {ckpt_path}")
    print(f"[Eval Only] Output Dir : {run_dir}")
    print("=" * 90)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_class_names = ckpt.get("class_names", None)

    eval_loader, data_info = build_eval_loader_for_split(
        data_root=data_root,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=img_size,
        with_paths=True,
        expected_class_names=ckpt_class_names,
    )

    class_names = data_info["class_names"]
    num_classes = data_info["num_classes"]

    model = build_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # CrossEntropyLoss（交叉熵损失）衡量分类分数与真实类别差多远；越小通常表示拟合越好。
    # 它内部已处理 logits，不要在模型末尾先加 Softmax 再把概率送进来。
    criterion = nn.CrossEntropyLoss()
    param_count = count_trainable_params(model)

    result = evaluate_and_export_split(
        model=model,
        dataloader=eval_loader,
        criterion=criterion,
        device=device,
        class_names=class_names,
        output_dir=run_dir,
        split=split,
        task_name=task_name,
        model_name=model_name,
        seed=seed,
        trainable_params=param_count,
        checkpoint_path=ckpt_path,
        best_epoch=int(ckpt.get("epoch", -1)),
        best_val_macro_f1=float(ckpt.get("best_val_macro_f1", -1.0)),
        dataset_sizes={"split_size": data_info["split_size"]},
    )

    # Keep old flat metric file location/name for aggregation compatibility.
    flat_run_dir = os.path.join(eval_output_root, task_name, model_name, f"seed_{seed}")
    ensure_dir(flat_run_dir)
    save_json(result, os.path.join(flat_run_dir, f"{split}_metrics.json"))

    return result

# 汇总仅评估模式的各模型/seed 结果，兼容旧/新指标字段名称。
# 返回的是每次实验明细和按当前 split Macro-F1 排名的表，不是跨 seed 平均结果；旧文件可能被读取，宜用新输出目录。
def aggregate_eval_only_results(
    eval_output_root: str,
    task_name: str,
    models_to_run: List[str],
    seeds: List[int],
    split: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate evaluation-only results across models and seeds.

    This version does NOT compute mean/std across seeds. It saves one row per
    seed and one per-model table for the completed seeds.
    """
    all_rows = []

    for model_name in models_to_run:
        for seed in seeds:
            metrics_path = os.path.join(
                eval_output_root,
                task_name,
                model_name,
                f"seed_{seed}",
                f"{split}_metrics.json"
            )
            if not os.path.exists(metrics_path):
                print(f"[Warning] Missing eval metric file: {metrics_path}")
                continue

            metrics = load_json(metrics_path)
            # Compatibility fix:
            # New evaluation exports use "best_epoch" and "best_val_macro_f1".
            # Some older aggregation code expected "best_epoch_from_checkpoint" and
            # "best_val_macro_f1_from_checkpoint". Support both formats.
            best_epoch_from_checkpoint = metrics.get(
                "best_epoch_from_checkpoint",
                metrics.get("best_epoch", -1)
            )
            best_val_macro_f1_from_checkpoint = metrics.get(
                "best_val_macro_f1_from_checkpoint",
                metrics.get("best_val_macro_f1", np.nan)
            )

            row = {
                "task_name": task_name,
                "model_name": model_name,
                "seed": seed,
                "eval_split": split,
                "trainable_params": metrics["trainable_params"],
                "best_epoch_from_checkpoint": best_epoch_from_checkpoint,
                "best_val_macro_f1_from_checkpoint": best_val_macro_f1_from_checkpoint,

                f"{split}_accuracy": metrics[f"{split}_accuracy"],
                f"{split}_macro_precision": metrics[f"{split}_macro_precision"],
                f"{split}_macro_recall": metrics[f"{split}_macro_recall"],
                f"{split}_macro_f1": metrics[f"{split}_macro_f1"],
                f"{split}_weighted_f1": metrics[f"{split}_weighted_f1"],

                "defect_detection_rate": metrics.get("defect_detection_rate", np.nan),
                "detection_error_rate": metrics.get("detection_error_rate", np.nan),
                "miss_rate": metrics.get("miss_rate", np.nan),
                "false_alarm_rate": metrics.get("false_alarm_rate", np.nan),
                "num_defect_samples": metrics.get("num_defect_samples", np.nan),
                "num_normal_samples": metrics.get("num_normal_samples", np.nan),
                "num_detection_errors": metrics.get("num_detection_errors", np.nan),
                "binary_tp": metrics.get("binary_tp", np.nan),
                "binary_tn": metrics.get("binary_tn", np.nan),
                "binary_fp": metrics.get("binary_fp", np.nan),
                "binary_fn": metrics.get("binary_fn", np.nan),

                "num_errors": metrics["num_errors"]
            }
            all_rows.append(row)

    detail_df = pd.DataFrame(all_rows)
    if detail_df.empty:
        print("[Warning] No evaluation-only results available for aggregation.")
        return detail_df, pd.DataFrame()

    task_dir = os.path.join(eval_output_root, task_name)
    ensure_dir(task_dir)

    detail_df = detail_df.sort_values(by=["model_name", "seed"]).reset_index(drop=True)

    detail_path = os.path.join(task_dir, f"summary_{split}_all_models_per_seed.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    for model_name, group in detail_df.groupby("model_name", sort=False):
        model_df = group.sort_values(by="seed").reset_index(drop=True)
        model_path = os.path.join(task_dir, f"summary_{split}_{model_name}_per_seed.csv")
        model_df.to_csv(model_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 90)
        print(f"[Eval Only][{task_name} | {model_name} | {split}] Per-seed result table saved:")
        print(model_path)
        display_cols = [
            "seed", "best_epoch_from_checkpoint",
            f"{split}_accuracy", f"{split}_macro_f1",
            "defect_detection_rate", "detection_error_rate",
            "binary_fp", "binary_fn", "num_detection_errors"
        ]
        display_cols = [c for c in display_cols if c in model_df.columns]
        print(model_df[display_cols].to_string(index=False))
        print("=" * 90)

    ranking_df = detail_df.sort_values(by=f"{split}_macro_f1", ascending=False).reset_index(drop=True)
    ranking_path = os.path.join(task_dir, f"summary_{split}_single_run_ranking.csv")
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print(f"[Eval Only][{task_name}][{split}] Per-seed aggregation outputs saved:")
    print(detail_path)
    print(ranking_path)
    print("=" * 90)

    return detail_df, ranking_df


# 根据评估排名选择一条实验，把其部分图表复制到 paper_outputs。
# 原 selection_rule 文本写“跨 seed 均值”，当前实际接收的 summary_df 是逐次实验排名，文字与实现不一致。
# 若 split=test，选中的只是测试成绩最高的展示案例，不能视为独立验证得到的最优模型结论。
def copy_best_eval_outputs(
    eval_output_root: str,
    task_name: str,
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    split: str
):
    """Copy the best architecture/seed confusion matrices to paper_outputs."""
    if detail_df.empty or summary_df.empty:
        return

    best_arch = summary_df.iloc[0]["model_name"]
    arch_runs = detail_df[detail_df["model_name"] == best_arch].copy()
    arch_runs = arch_runs.sort_values(by=f"{split}_macro_f1", ascending=False)
    best_run = arch_runs.iloc[0]
    best_seed = int(best_run["seed"])

    paper_dir = os.path.join(eval_output_root, task_name, "paper_outputs")
    ensure_dir(paper_dir)

    selection_info = {
        "eval_split": split,
        "best_architecture": best_arch,
        "best_seed": best_seed,
        "best_run_macro_f1": float(best_run[f"{split}_macro_f1"]),
        "best_run_accuracy": float(best_run[f"{split}_accuracy"]),
        "selection_rule": (
            f"Best architecture selected by highest mean {split} Macro-F1 across seeds; "
            f"representative checkpoint selected by highest single-run {split} Macro-F1 within that architecture."
        )
    }
    save_json(selection_info, os.path.join(paper_dir, f"best_model_selection_{split}.json"))

    src_dir = os.path.join(eval_output_root, task_name, best_arch, f"seed_{best_seed}", split)
    copy_items = [
        f"{split}_confusion_matrix_raw.png",
        f"{split}_confusion_matrix_normalized.png",
        f"{split}_confusion_matrix_raw.csv",
        f"{split}_confusion_matrix_normalized.csv",
        f"{split}_classification_report.txt",
        f"{split}_metrics.json",
        f"{split}_misclassified_samples.csv",
    ]

    for item in copy_items:
        src = os.path.join(src_dir, item)
        if os.path.exists(src):
            dst = os.path.join(paper_dir, item)
            shutil.copy2(src, dst)

    print("\n" + "=" * 90)
    print(f"[Eval Only] Best {split} outputs copied to:")
    print(paper_dir)
    print("=" * 90)


# 按模型×seed 批量调用已有权重评估，随后汇总和复制展示图表。
# --eval_split val,test 会由 main 分两次调用这里；读取的集合不同，但使用的权重相同。
def evaluate_existing_best_models_on_split(
    data_root: str,
    checkpoint_root: str,
    eval_output_root: str,
    task_name: str,
    models_to_run: List[str],
    seeds: List[int],
    split: str,
    batch_size: int,
    num_workers: int,
    img_size: int,
    device: torch.device
):
    """Evaluate all existing best checkpoints on val/test/train split."""
    ensure_dir(eval_output_root)
    ensure_dir(os.path.join(eval_output_root, task_name))

    for model_name in models_to_run:
        for seed in seeds:
            evaluate_one_best_checkpoint_on_split(
                data_root=data_root,
                checkpoint_root=checkpoint_root,
                eval_output_root=eval_output_root,
                task_name=task_name,
                model_name=model_name,
                seed=seed,
                split=split,
                batch_size=batch_size,
                num_workers=num_workers,
                img_size=img_size,
                device=device
            )

    detail_df, summary_df = aggregate_eval_only_results(
        eval_output_root=eval_output_root,
        task_name=task_name,
        models_to_run=models_to_run,
        seeds=seeds,
        split=split
    )

    copy_best_eval_outputs(
        eval_output_root=eval_output_root,
        task_name=task_name,
        detail_df=detail_df,
        summary_df=summary_df,
        split=split
    )

    return detail_df, summary_df


# ============================================================
# 12. Main
# ============================================================

# 命令行参数总入口。优先在运行命令里改参数；改 default 只影响未显式传参的情况。
# 每个参数的解释在下面对应行附近；数据、模型、结果路径必须区分，避免把原始目录误作输出。
def parse_args():
    parser = argparse.ArgumentParser(description="Grain Defect Dataset Benchmark Training")

    # 数据根目录，直接含 train/val/test；不填会报必填参数缺失。
    parser.add_argument("--data_root", type=str, required=True,
                        help="Split dataset root, containing train/val/test")

    # 任务名称同时作为输出子目录名；改名不会自动改变类别或损失函数。
    parser.add_argument("--task_name", type=str, required=True,
                        help="Task name, e.g., binary or defect_type")

    # 实验结果根目录，存权重/日志/图表；建议每套超参数使用新的目录。
    parser.add_argument("--output_root", type=str, default="./results",
                        help="Root directory to save all results")

    # 逗号分隔模型名，例如 repvit_v4 或 repvit,repvit_v4；不要用中文逗号。
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                        help="Models to train, comma-separated")

    # 逗号分隔整数随机种子；影响 Python 训练随机性，不会重新划分现成数据。
    parser.add_argument("--seeds", type=str, default=",".join(map(str, DEFAULT_SEEDS)),
                        help="Random seeds, comma-separated")

    # 最大训练轮数；先小规模跑通只能验证流程，不能证明充分收敛或工程达标。
    parser.add_argument("--epochs", type=int, default=40,
                        help="Maximum training epochs")

    # 每批同时处理图片数；过大可能 CUDA out of memory（显存不足），可先减半。
    # 批量改变会影响训练行为，不只是速度；需在比较表中记录。
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")

    # 读取图片的子进程数；Windows 多进程读图报错时可用 0 简化排查，可能变慢。
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers")

    # 输入缩放的高和宽；尺寸越大通常计算量越大，某些模型有固定输入限制。
    # 界面 app.py 另有固定 224，改变训练尺寸时必须同步修改并验证模型支持。
    parser.add_argument("--img_size", type=int, default=224,
                        help="Input image size")

    # 学习率（参数每次更新的步幅），1e-4 等于 0.0001；优先通过命令覆盖默认值。
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")

    # 权重衰减（限制参数过大的一种办法）；过强可能使模型学不充分，不是越大越好。
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay")

    # 早停等待轮数，单位是 epoch；不是训练批次数，也不保证一定训练这么多轮。
    parser.add_argument("--patience", type=int, default=8,
                        help="Early stopping patience")

    # RepViT/常用 torchvision 分支：1 使用通用图像预训练初始化，0 随机初始化。
    # 首次使用 1 可能联网下载；已有权重的 eval-only 不靠本参数恢复参数。MambaVision 备用分支有例外，见对应类。
    parser.add_argument("--pretrained", type=int, default=1,
                        help="1: use ImageNet pretrained weights; 0: train from scratch")

    # cuda 使用 NVIDIA GPU；cpu 使用处理器。仅当值恰为 cuda 且不可用时 main 自动退回 CPU。
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")

    # 开关参数：写出 --eval_only 就只评估，不要在后面加 1；省略则进入训练。
    parser.add_argument("--eval_only", action="store_true",
                        help="Only evaluate existing best_model.pth checkpoints without training")

    # 仅评估时选择 train/val/test，可写 val,test；训练集成绩不能代替泛化成绩。
    parser.add_argument("--eval_split", type=str, default="val",
                        help="Dataset split(s) to evaluate in eval_only mode, e.g. val, test, or val,test")

    # 已有权重的根目录；其下仍需有 task_name/model_name/seed_N/best_model.pth。
    parser.add_argument("--checkpoint_root", type=str, default=None,
                        help="Root directory of trained checkpoints. If None, use --output_root")

    # 仅评估生成文件的位置，建议与训练结果分开，并为每次评估使用新目录。
    parser.add_argument("--eval_output_root", type=str, default="./val_result",
                        help="Root directory to save eval_only outputs")

    return parser.parse_args()


# 程序总调度：解析参数→选设备→解析模型/seed 列表→仅评估或训练。
# 正常训练按“一个模型的全部 seeds→汇总→下一个模型”执行；不会自动搜索最佳超参数。
# Windows 必须保留文件末尾的 __main__ 保护，避免多进程读取图片时重复启动训练。
def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; switching to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ensure_dir(args.output_root)
    ensure_dir(os.path.join(args.output_root, args.task_name))

    models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("\nExperiment configuration:")
    print(f"Data Root   : {args.data_root}")
    print(f"Task Name   : {args.task_name}")
    print(f"Output Root : {args.output_root}")
    print(f"Models      : {models_to_run}")
    print(f"Seeds       : {seeds}")
    print(f"Epochs      : {args.epochs}")
    print(f"Batch Size  : {args.batch_size}")
    print(f"Num Workers : {args.num_workers}")
    print(f"Device      : {device}")
    print(f"Pretrained  : {bool(args.pretrained)}")

    # 分流点：仅评估走这个分支并在完成后 return，不会进入下面训练循环。
    if args.eval_only:
        checkpoint_root = args.checkpoint_root if args.checkpoint_root is not None else args.output_root
        print("\n[Eval Only Mode]")
        print(f"Checkpoint Root : {checkpoint_root}")
        print(f"Eval Split      : {args.eval_split}")
        print(f"Eval Output Root: {args.eval_output_root}")

        eval_splits = [s.strip() for s in args.eval_split.split(",") if s.strip()]
        valid_splits = {"train", "val", "test"}
        for split in eval_splits:
            if split not in valid_splits:
                raise ValueError(f"Unsupported eval split: {split}. Valid options: train, val, test")
            evaluate_existing_best_models_on_split(
                data_root=args.data_root,
                checkpoint_root=checkpoint_root,
                eval_output_root=args.eval_output_root,
                task_name=args.task_name,
                models_to_run=models_to_run,
                seeds=seeds,
                split=split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=args.img_size,
                device=device
            )
        return

    # Train one model across all seeds, then immediately summarize that model's
    # seed-level results into a per-model CSV table.
    for model_name in models_to_run:
        for seed in seeds:
            train_one_run(
                model_name=model_name,
                seed=seed,
                task_name=args.task_name,
                data_root=args.data_root,
                output_root=args.output_root,
                epochs=args.epochs,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                img_size=args.img_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                patience=args.patience,
                pretrained=args.pretrained,
                device=device
            )

        # Per-model per-seed table after this model's seeds are completed.
        aggregate_task_results(
            output_root=args.output_root,
            task_name=args.task_name,
            models_to_run=[model_name],
            seeds=seeds
        )

    # Combined per-seed table for all models. No mean/std table is produced.
    aggregate_task_results(
        output_root=args.output_root,
        task_name=args.task_name,
        models_to_run=models_to_run,
        seeds=seeds
    )


# 只有直接运行本文件才调用 main；app.py 导入 build_model 时不训练。
# 导入仍会加载顶层依赖和绘图设置，因此缺少依赖时展示也可能启动失败。
if __name__ == "__main__":
    main()
