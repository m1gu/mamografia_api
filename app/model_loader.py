# app/model_loader.py
import torch
import monai
from pathlib import Path

# Ruta de modelos
MODELS_DIR = Path("models")
CLASSIFICADOR_PTH = MODELS_DIR / "clasificador.pth"
SEGMENTADOR_PTH = MODELS_DIR / "segmentador.pth"

# Transformaciones (usamos datamodules reales)
from app.datamodules import get_transforms_clasificacion, get_transforms_segmentacion

def load_models():
    # Clasificador
    model_cls = monai.networks.nets.DenseNet121(spatial_dims=2, in_channels=1, out_channels=2)
    model_cls.load_state_dict(torch.load(CLASSIFICADOR_PTH, map_location=torch.device("cpu")))
    model_cls.eval()

    # Segmentador
    model_seg = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model_seg.load_state_dict(torch.load(SEGMENTADOR_PTH, map_location=torch.device("cpu")))
    model_seg.eval()

    # Transformaciones
    transforms_cls = get_transforms_clasificacion()
    transforms_seg = get_transforms_segmentacion()

    return model_cls, model_seg, transforms_cls, transforms_seg

def predict_clasificacion(model, transform, image_tensor):
    x = transform({"image": image_tensor})["image"].unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1).squeeze().tolist()
        etiqueta = "Benigno" if torch.argmax(logits) == 0 else "Maligno"
    return etiqueta, max(prob)

def predict_segmentacion(model, transform, image_tensor):
    x = transform({"image": image_tensor})["image"].unsqueeze(0)
    with torch.no_grad():
        output = model(x)
        mask = torch.sigmoid(output).squeeze().numpy()
    return mask
