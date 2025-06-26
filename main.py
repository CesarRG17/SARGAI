from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import pandas as pd

# ---------- 1. MODELO Y CATEGORÍAS ----------
CLASSES = ['nada', 'bajo', 'moderado', 'abundante', 'excesivo']

model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
model.load_state_dict(torch.load("modelo_sargazo_efficientnet.pt", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------- 2. CONFIG FASTAPI ----------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- 3. RUTA PÁGINA PRINCIPAL ----------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------- 4. RUTA DE PREDICCIÓN MULTIIMAGEN ----------
@app.post("/predecir")
async def predecir(archivos: list[UploadFile] = File(...)):
    resultados = []

    os.makedirs("temp", exist_ok=True)

    for archivo in archivos:
        ruta = f"temp/{archivo.filename}"
        with open(ruta, "wb") as buffer:
            shutil.copyfileobj(archivo.file, buffer)

        try:
            img = Image.open(ruta).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                pred_idx = output.argmax(1).item()
                pred = CLASSES[pred_idx]

            resultados.append({"nombre": archivo.filename, "prediccion": pred})
        except Exception as e:
            resultados.append({"nombre": archivo.filename, "prediccion": f"Error: {str(e)}"})

        os.remove(ruta)

    df = pd.DataFrame(resultados)
    df.to_csv("temp/predicciones.csv", index=False)

    return {"resultados": resultados, "csv": "/descargar_csv"}


# ---------- 5. RUTA PARA DESCARGAR CSV ----------
@app.get("/descargar_csv")
def descargar_csv():
    return FileResponse("temp/predicciones.csv", media_type="text/csv", filename="predicciones_sargazo.csv")
