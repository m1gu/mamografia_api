# app/utils.py
import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import base64
import io

def preprocess_image(image_path: str):
    """
    Carga una imagen, la convierte a escala de grises, la hace cuadrada con padding negro
    y la transforma a tensor.
    """
    img = Image.open(image_path).convert("L")  # Escala de grises (1 canal)
    w, h = img.size
    size = max(w, h)

    # Crear nueva imagen cuadrada con fondo negro
    new_img = Image.new("L", (size, size))
    new_img.paste(img, ((size - w) // 2, (size - h) // 2))

    # Convertir a tensor [C, H, W] y escalar a [0, 1]
    tensor = transforms.ToTensor()(new_img)
    return tensor

def postprocess_segmentacion(image_path, mask, threshold=0.5):
    """
    Superpone una máscara binaria como contorno rojo sobre la imagen original.
    Devuelve la imagen codificada en base64 (opcional para mostrar en frontend).
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_bin = (mask > threshold).astype(np.uint8)

    # Encontrar contornos
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 2)  # Contorno rojo

    # Codificar como base64 para retornar por API
    _, buffer = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    b64 = base64.b64encode(buffer).decode("utf-8")
    return b64
