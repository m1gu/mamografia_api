# app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.model_loader import load_models, predict_clasificacion, predict_segmentacion
from app.utils import preprocess_image, postprocess_segmentacion
import shutil
import os
from pathlib import Path
from uuid import uuid4

app = FastAPI(title="API Diagnóstico Mamografía")

# Cargar modelos al iniciar la API
model_cls, model_seg, transforms_cls, transforms_seg = load_models()

# Crear carpeta temporal para archivos subidos
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Guardar archivo temporalmente
    file_id = str(uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}.png"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Preprocesar imagen
    img_tensor = preprocess_image(str(temp_path))

    # Clasificación
    etiqueta, prob = predict_clasificacion(model_cls, transforms_cls, img_tensor)

    # Segmentación (si no es "normal")
    segmentada = None
    if etiqueta in ["Benigno", "Maligno"]:
        mask = predict_segmentacion(model_seg, transforms_seg, img_tensor)
        segmentada = postprocess_segmentacion(temp_path, mask)

    # Respuesta
    return JSONResponse({
        "etiqueta": etiqueta,
        "probabilidad": prob,
        "segmentacion": segmentada  # Puede ser base64 o path temporal si se desea mostrar
    })
