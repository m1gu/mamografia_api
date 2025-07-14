import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import base64
from ultralytics import YOLO

# Cargar modelo YOLO una sola vez
model_yolo = YOLO("yolov8n.pt").to("cpu")

def preprocess_image(image_path: str):
    """
    Carga imagen, convierte a escala de grises, la redimensiona a 256x256,
    y la convierte a tensor [1, 256, 256] en rango [0, 1].
    """
    img = Image.open(image_path).convert("L")
    img_resized = img.resize((256, 256), Image.BILINEAR)
    tensor = transforms.ToTensor()(img_resized).float()
    return tensor

def postprocess_segmentacion(image_tensor, mask, threshold=0.9):
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

def detect_roi(image_tensor, padding=20):
    """
    Detecta ROI usando YOLO sobre una versión RGB de la imagen 1 canal.
    Retorna coordenadas [x1, y1, x2, y2].
    """
    img = image_tensor.squeeze().cpu().numpy()
    if image_tensor.shape[0] != 1:
        raise ValueError("Se esperaba un tensor de 1 canal")

    img_rgb = np.stack([img] * 3, axis=-1)  # [H, W, 3]
    img_rgb = (img_rgb * 255).astype(np.uint8)

    results = model_yolo(img_rgb)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if boxes.shape[0] == 0:
        # No hay detecciones, usar imagen completa
        h, w = img.shape
        return np.array([0, 0, w, h])

    x1, y1, x2, y2 = boxes[0].astype(int)

    # Aplicar padding limitado por bordes
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img.shape[1], x2 + padding)
    y2 = min(img.shape[0], y2 + padding)

    return np.array([x1, y1, x2, y2])
