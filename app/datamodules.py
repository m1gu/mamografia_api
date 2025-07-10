from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from monai.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, RandGaussianNoised,
    Resized, ScaleIntensityd, EnsureTyped, RandAffined, RandAdjustContrastd,
    RandFlipd, RandRotate90d, Lambdad, CropForegroundd, RandZoomd, Rand2DElasticd,
)
from sklearn.model_selection import StratifiedKFold

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
ROOT = Path(r"D:/UTEG/TESIS FINAL/v6/data")
CSV = ROOT / "dataset_final.csv"
SIZE = (256, 256)
REPEAT = 6
BATCH_CLS = 16
BATCH_SEG = 12
SEED = 42

# =========================
# FUNCIONES AUXILIARES
# =========================
def normalize_mask(x):
    return ((x / 255.0) > 0.5).astype(np.float32)

def label_to_int(label):
    return 0 if label.upper() == "BENIGN" else 1

# =========================
# TRANSFORMACIONES
# =========================

def get_transforms_clasificacion(train=True):
    base = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=SIZE, mode="bilinear"),
        ScaleIntensityd(keys="image"),
        EnsureTyped(keys=["image", "label_cls"], dtype=(torch.float32, torch.long), allow_missing_keys=True),
    ]
    aug = [
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image"], prob=0.5),
        RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandAffined(keys=["image"], rotate_range=(0.1,), prob=0.3),
        RandAdjustContrastd(keys=["image"], prob=0.2),
    ]
    return Compose(base + aug if train else base)

def get_transforms_segmentacion(train=True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        CropForegroundd(keys=["image", "label"], source_key="label", margin=15, allow_smaller=True),
        Resized(keys=["image", "label"], spatial_size=SIZE, mode=("bilinear", "nearest")),
        ScaleIntensityd(keys="image"),
        Lambdad(keys="label", func=normalize_mask),
        EnsureTyped(keys=["image", "label"], dtype=("float32", "uint8")),
    ]
    aug = [
        RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
        RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
        RandRotate90d(keys=["image", "label"], prob=0.5),
        RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.3, keep_size=True),
        RandGaussianNoised(keys=["image"], prob=0.2, std=0.01),
        RandAffined(keys=["image", "label"], rotate_range=(0.1,), prob=0.3),
        Rand2DElasticd(
            keys=["image", "label"], spacing=(10, 10), magnitude_range=(1, 2),
            rotate_range=(0.1, 0.1), scale_range=(0.1, 0.1), prob=0.2, padding_mode="border"
        )
    ]
    return Compose(base + aug if train else base)

# =========================
# LOADERS
# =========================

def loaders_clasificacion(batch=BATCH_CLS, repeat=REPEAT, k_fold=5, fold=0, seed=42):
    df = pd.read_csv(CSV)
    df["label"] = df["pathology"].apply(label_to_int)
    X = df["mamografia_image_file"].tolist()
    y = df["label"].tolist()

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
    train_idx, val_idx = list(skf.split(X, y))[fold]

    train_files = [{"image": ROOT / X[i], "label_cls": y[i]} for i in train_idx]
    val_files   = [{"image": ROOT / X[i], "label_cls": y[i]} for i in val_idx]
    test_split = int(0.15 * len(X))
    test_files = [{"image": ROOT / X[i], "label_cls": y[i]} for i in range(test_split)]

    train_ds = Dataset(train_files, transform=get_transforms_clasificacion(train=True))
    val_ds   = Dataset(val_files, transform=get_transforms_clasificacion(train=False))
    test_ds  = Dataset(test_files, transform=get_transforms_clasificacion(train=False))

    train_final_ds = ConcatDataset([train_ds] * repeat)

    return (
        DataLoader(train_final_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True),
    )

def loaders_segmentacion(batch=BATCH_SEG, repeat=REPEAT, k_fold=None, fold=0):
    df = pd.read_csv(CSV)
    all_data = [
        {"image": ROOT / row["mamografia_image_file"], "label": ROOT / row["mascara_image_file"]}
        for _, row in df.iterrows()
    ]
    np.random.seed(SEED)

    if k_fold is not None and k_fold > 1:
        kf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=SEED)
        labels = df["pathology"].apply(label_to_int).tolist()
        train_idx, val_idx = list(kf.split(all_data, labels))[fold]
        train_files = [all_data[i] for i in train_idx]
        val_files   = [all_data[i] for i in val_idx]
    else:
        total = len(all_data)
        np.random.shuffle(all_data)
        train_end = int(0.72 * total)
        val_end   = int(0.90 * total)
        train_files = all_data[:train_end]
        val_files   = all_data[train_end:val_end]

    train_ds = Dataset(data=train_files, transform=get_transforms_segmentacion(train=True))
    val_ds   = Dataset(data=val_files, transform=get_transforms_segmentacion(train=False))
    test_ds  = Dataset(data=val_files, transform=get_transforms_segmentacion(train=False))

    train_final_ds = ConcatDataset([train_ds] * repeat)

    return (
        DataLoader(train_final_ds, batch_size=batch, shuffle=True, num_workers=2, pin_memory=True),
        DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2, pin_memory=True),
        DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True),
    )