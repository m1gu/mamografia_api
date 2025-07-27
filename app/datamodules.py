from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Resized, ScaleIntensityd, EnsureTyped
)
import torch
import numpy as np
import cv2

SIZE = (256, 256)

def normalize_mask(x):
    return ((x / 255.0) > 0.5).astype(np.float32)

def get_transforms_clasificacion():
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=SIZE, mode="bilinear"),
        ScaleIntensityd(keys="image"),
        EnsureTyped(keys=["image"], dtype=torch.float32)
    ])

def get_transforms_segmentacion(pred=True):
    return Compose([
        Resized(keys=["image"], spatial_size=SIZE, mode="bilinear"),
        ScaleIntensityd(keys="image", allow_missing_keys=True),
        EnsureTyped(keys=["image"], dtype=torch.float32, allow_missing_keys=True)
    ])
