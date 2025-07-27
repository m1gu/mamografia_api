import cv2
import numpy as np
import torch
import torch.serialization
from torch.nn import Sequential
from torchvision import transforms
from PIL import Image
import base64
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

# Cargar modelo YOLO una sola vez
#model_yolo = YOLO("yolov8n.pt").to("cpu")


model_yolo = None

def preprocess_image(image_path: str):
    """
    Carga imagen, convierte a escala de grises, la redimensiona a 256x256,
    y la convierte a tensor [1, 256, 256] en rango [0, 1].
    """
    img = Image.open(image_path).convert("L")
    img_resized = img.resize((256, 256), Image.BILINEAR)
    tensor = transforms.ToTensor()(img_resized).float()
    return tensor

def postprocess_segmentacion(image_tensor, mask, threshold=0.5):
    """
    Postprocesa la máscara segmentada para visualización con contorno y overlay.
    Devuelve la imagen como string base64 codificada en PNG.
    """
    if mask.max() == 0:
        return None  # Sin predicción válida

    # Paso 1: Suavizado
    mask_filtered = cv2.bilateralFilter(mask.astype(np.float32), 15, 75, 75)

    # Paso 2: Umbralización adaptativa
    binary = cv2.threshold((mask_filtered * 255).astype(np.uint8), 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Paso 3: Detección de contorno principal
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea, default=None)

    final_mask = np.zeros_like(binary)
    if main_contour is not None and cv2.contourArea(main_contour) > 50:
        # Aproximación poligonal y relleno
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        cv2.drawContours(final_mask, [approx], -1, 255, thickness=cv2.FILLED)

        # Cierre morfológico
        kernel = np.ones((7, 7), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Visualización
    img = image_tensor.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if main_contour is not None:
        cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
        overlay = img.copy()
        cv2.drawContours(overlay, [approx], -1, (0, 0, 255), thickness=cv2.FILLED)
        img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

    _, buffer = cv2.imencode(".png", img)
    return base64.b64encode(buffer).decode("utf-8")

def detect_roi(image_tensor, target_size=256):
    """
    Detecta una región de interés (ROI) centrada dentro del bounding box de YOLO,
    con tamaño fijo (por ejemplo 256x256).
    """

    safe_classes = [DetectionModel, Sequential, Conv]
    with torch.serialization.safe_globals(safe_classes):
        model_yolo = YOLO("https://huggingface.co/ultralytics/yolov8/resolve/main/yolov8n.pt").to("cpu")


    img = image_tensor.squeeze().cpu().numpy()
    img_rgb = np.stack([img] * 3, axis=-1)  # Convertir a RGB
    img_rgb = (img_rgb * 255).astype(np.uint8)

    results = model_yolo(img_rgb)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    h_img, w_img = img.shape

    if boxes.shape[0] == 0:
        # Si no detecta nada, usar el centro de la imagen
        center_x, center_y = w_img // 2, h_img // 2
    else:
        x1, y1, x2, y2 = boxes[0].astype(int)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

    # Calcular nuevo ROI centrado de tamaño target_size
    half_size = target_size // 2
    x1_new = max(0, center_x - half_size)
    y1_new = max(0, center_y - half_size)
    x2_new = min(w_img, center_x + half_size)
    y2_new = min(h_img, center_y + half_size)

    return np.array([x1_new, y1_new, x2_new, y2_new])


