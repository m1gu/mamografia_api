from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.model_loader import load_models, predict_clasificacion, predict_segmentacion
from app.utils import preprocess_image, postprocess_segmentacion, detect_roi
from pathlib import Path
from uuid import uuid4
import shutil
import torch
import numpy as np

app = FastAPI(title="API Diagnóstico Mamografía")

# Carpeta temporal
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_id = str(uuid4())
    temp_path = UPLOAD_DIR / f"{file_id}.png"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        #  Cargar modelos solo cuando se necesiten
        model_cls, model_seg, transforms_cls, transforms_seg = load_models()

        img_tensor = preprocess_image(str(temp_path))
        etiqueta, prob = predict_clasificacion(model_cls, transforms_cls, str(temp_path))
    except Exception as e:
        return JSONResponse({"error": f"Error en clasificación: {str(e)}"}, status_code=500)

    segmentada = None
    if etiqueta in ["Benigno", "Maligno"]:
        try:
            roi = detect_roi(img_tensor)
            x1, y1, x2, y2 = map(int, roi)
            print(f"ROI detectada: {roi}")

            cropped_tensor = img_tensor[:, y1:y2, x1:x2]
            cropped_resized = torch.nn.functional.interpolate(
                cropped_tensor.unsqueeze(0), size=(256, 256), mode="bilinear"
            ).squeeze(0)

            mask = predict_segmentacion(model_seg, transforms_seg, cropped_resized)
            print(f"Máximo valor de la máscara original: {np.max(mask)}")

            mask_resized = torch.nn.functional.interpolate(
                torch.from_numpy(mask).unsqueeze(0).unsqueeze(0),
                size=(y2 - y1, x2 - x1),
                mode="nearest"
            ).squeeze().numpy()

            mascara_binaria = (mask_resized > 0.5).astype(np.uint8)

            full_mask = np.zeros(img_tensor.shape[1:], dtype=np.float32)
            full_mask[y1:y2, x1:x2] = mascara_binaria

            segmentada = postprocess_segmentacion(img_tensor, full_mask)

        except Exception as e:
            return JSONResponse({"error": f"Error en segmentación: {str(e)}"}, status_code=500)

    return JSONResponse({
        "etiqueta": etiqueta,
        "probabilidad": prob,
        "segmentacion": segmentada
    })

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return FileResponse("static/index.html")
