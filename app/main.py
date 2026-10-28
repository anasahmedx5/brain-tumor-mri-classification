# app/main.py

import io
import os
import time
from contextlib import asynccontextmanager
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
# Directories setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

import sys
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from app.config import FRAMEWORK, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE
except ImportError:
    from config import FRAMEWORK, MODEL_PATH, CLASS_NAMES, IMAGE_SIZE

# Global variables to hold model and preprocessing pipeline
model = None
transform = None


def load_pytorch_model(model_path: str, num_classes: int) -> torch.nn.Module:
    """
    Instantiates the ResNet-18 architecture, replaces the final FC layer,
    loads the trained weights, and sets the model to evaluation mode on CPU.
    """
    net = models.resnet18(weights=None)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_classes)
    
    # Resolve model path relative to project root if not found directly
    if not os.path.exists(model_path):
        alt_path = os.path.join(PROJECT_ROOT, model_path)
        if os.path.exists(alt_path):
            model_path = alt_path

    # Load state dict strictly to CPU
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(state_dict)
    net.eval()
    return net


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to load the model ONCE at startup.
    """
    global model, transform
    
    # Define standard ImageNet transform (matching training preprocessing)
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load model state dict
    try:
        model = load_pytorch_model(MODEL_PATH, len(CLASS_NAMES))
        print(f"Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    yield


app = FastAPI(
    title="Brain Tumor Classification API",
    lifespan=lifespan
)

# Enable CORS for browser UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static web UI assets and dataset images
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if os.path.exists(DATA_DIR):
    app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


@app.get("/")
def read_root():
    """
    Serves the Brain Tumor MRI Diagnostic Web Interface.
    """
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Brain Tumor Classification API. Visit /docs for Swagger UI."}


@app.get("/health")
def health_check():
    """
    Health check endpoint matching the assignment contract.
    """
    return {
        "status": "ok",
        "framework": FRAMEWORK,
        "classes": CLASS_NAMES
    }


@app.get("/api/samples")
def list_samples():
    """
    Returns available sample MRI images in the data folder for quick testing.
    """
    if not os.path.exists(DATA_DIR):
        return []
    valid_extensions = (".jpg", ".jpeg", ".png")
    return [
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(valid_extensions)
    ]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint accepting an image file via multipart/form-data.
    """
    start_time = time.perf_counter()

    # 1. Validate file extension/type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image (JPEG or PNG)."
        )

    # 2. Read image content
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or corrupted format."
        )

    # 3. Apply preprocessing matching training
    input_tensor = transform(image).unsqueeze(0)  # Shape: [1, 3, 224, 224]

    # 4. Perform inference on CPU
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze(0)

    # 5. Extract probabilities, confidence, and predicted class
    prob_dict = {
        class_name: float(probabilities[i])
        for i, class_name in enumerate(CLASS_NAMES)
    }
    
    top_prob, top_idx = torch.max(probabilities, dim=0)
    predicted_label = CLASS_NAMES[top_idx.item()]
    confidence = round(float(top_prob), 2)

    # Format output probabilities rounded to 2 decimal places
    prob_dict_formatted = {k: round(v, 2) for k, v in prob_dict.items()}

    # Calculate latency in milliseconds
    end_time = time.perf_counter()
    latency_ms = int((end_time - start_time) * 1000)

    # 6. Return response matching assignment contract
    return {
        "framework": FRAMEWORK,
        "label": predicted_label,
        "confidence": confidence,
        "probabilities": prob_dict_formatted,
        "latency_ms": latency_ms
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)