import torch
from huggingface_hub import hf_hub_download
from torchvision.models import densenet121
import torch.nn as nn
import segmentation_models_pytorch as smp
from app.datamodules import get_transforms_clasificacion, get_transforms_segmentacion
import cv2
import numpy as np

HF_REPO_ID = "mrodriguezegues/mamografia-modelos"

def descargar_modelo(nombre_archivo):
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=nombre_archivo,
        cache_dir="models"
    )

def densenet121_m(in_channels=1, num_classes=2):
    model = densenet121(weights=None)
    model.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

def unet_pretrained():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=1,
        classes=1,
        activation=None
    )

def load_models():
    clasificador_path = descargar_modelo("clasificador.pth")
    segmentador_path = descargar_modelo("segmentador_full_11.pth")

    # Clasificador
    model_cls = densenet121_m(in_channels=1, num_classes=2)
    checkpoint_cls = torch.load(clasificador_path, map_location="cpu")
    state_cls = checkpoint_cls["model_state_dict"] if "model_state_dict" in checkpoint_cls else checkpoint_cls
    model_cls.load_state_dict(state_cls)
    model_cls.eval()

    # Segmentador
    model_seg = unet_pretrained()
    checkpoint_seg = torch.load(segmentador_path, map_location="cpu")
    state_seg = checkpoint_seg["model_state_dict"] if "model_state_dict" in checkpoint_seg else checkpoint_seg
    model_seg.load_state_dict(state_seg)
    model_seg.eval()

    transforms_cls = get_transforms_clasificacion()
    transforms_seg = get_transforms_segmentacion()

    return model_cls, model_seg, transforms_cls, transforms_seg

def predict_clasificacion(model, transform, image_path):
    data = {"image": image_path}
    x = transform(data)["image"].unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1).cpu().numpy()[0]
    label = "Benigno" if prob[0] > prob[1] else "Maligno"
    return label, float(max(prob))

def predict_segmentacion(model, transform, image_tensor):
    data = {"image": image_tensor}
    tensor = transform(data)["image"]  # aplicar las transforms de MONAI
    return sliding_window_segmentation(model, tensor)

def sliding_window_segmentation(model, image_tensor, patch_size=256, stride=64, device="cpu"):
    model.eval()
    c, h, w = image_tensor.shape
    assert c == 1

    if image_tensor.dtype != torch.float32:
        image_tensor = image_tensor.float()
    if image_tensor.max() > 1.0:
        image_tensor = image_tensor / 255.0

    final_mask = torch.zeros((h, w), dtype=torch.float32)
    count_map = torch.zeros((h, w), dtype=torch.float32)

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image_tensor[:, y:y+patch_size, x:x+patch_size].unsqueeze(0)
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)

            with torch.no_grad():
                pred = model(patch.to(device))
                pred = torch.sigmoid(pred).squeeze().cpu()
                pred = torch.from_numpy(cv2.GaussianBlur(pred.numpy(), (5, 5), 0))

            final_mask[y:y+patch_size, x:x+patch_size] += pred
            count_map[y:y+patch_size, x:x+patch_size] += 1

    count_map[count_map == 0] = 1
    final_mask /= count_map

    mask_np = final_mask.numpy()
    mask_np = cv2.medianBlur(mask_np, 5)
    return mask_np
