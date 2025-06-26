from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# CATEGORÍAS
CLASSES = ['nada', 'bajo', 'moderado', 'abundante', 'excesivo']

# ---------- 1. MODELO DEFINICIÓN ----------
# En lugar de: model = CNN()
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Cargar modelo EfficientNet
model = efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
model.load_state_dict(torch.load("modelo_sargazo_efficientnet.pt", map_location=torch.device('cpu')))
model.eval()

# ---------- 2. CARGA DE MODELO ----------


# ---------- 3. CONFIGURACIÓN FASTAPI ----------
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------- 4. RUTA HTML ----------
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------- 5. RUTA PREDICCIÓN ----------
@app.post("/predecir")
async def predecir(imagen: UploadFile = File(...)):
    if imagen.filename == "":
        return JSONResponse(content={"error": "No se subió ninguna imagen"}, status_code=400)

    ruta = f"temp/{imagen.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(ruta, "wb") as buffer:
        shutil.copyfileobj(imagen.file, buffer)

    try:
        img = Image.open(ruta).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(1).item()
            pred = CLASSES[pred_idx]

        os.remove(ruta)
        return {"prediction": pred}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
