from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

from PIL import Image
import uuid
import os
import json
import shutil

app = FastAPI()

# -----------------------
# Paths
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODEL_PATH = os.path.join(BASE_DIR, "resnet18_model.pth")
HISTORY_FILE = os.path.join(BASE_DIR, "history.json")

os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)

templates = Jinja2Templates(directory=TEMPLATE_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -----------------------
# History Functions
# -----------------------
def save_history(image_path, prediction, confidence):
    history = load_history()

    history.append({
        "image": image_path,
        "prediction": prediction,
        "confidence": confidence
    })

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []

    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except:
        return []


# -----------------------
# Load Model
# -----------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(512, 10)

state_dict = torch.load(MODEL_PATH, map_location="cpu")

clean_state_dict = {}
for k, v in state_dict.items():
    clean_state_dict[k.replace("model.", "")] = v

model.load_state_dict(clean_state_dict)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = [
    "airplane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# -----------------------
# Home
# -----------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None,
            "confidence": None,
            "image_path": None,
            "history": load_history()
        }
    )


# -----------------------
# Prediction
# -----------------------
@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):

    filename = f"{uuid.uuid4()}.jpg"
    filepath = os.path.join(STATIC_DIR, "uploads", filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    img = Image.open(filepath).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    prediction = classes[pred.item()]
    confidence = round(conf.item() * 100, 2)

    image_path = f"/static/uploads/{filename}"

    save_history(image_path, prediction, confidence)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": prediction,
            "confidence": confidence,
            "image_path": image_path,
            "history": load_history()
        }
    )


# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000
    )