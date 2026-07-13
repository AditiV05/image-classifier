"""
Slice 5: serve the trained CNN via FastAPI.
POST /predict with an image file -> returns predicted class + confidence scores.
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io

from model import CNNModel
from dataset import CLASS_NAMES

app = FastAPI()

# Allow requests from the frontend later (Slice 6)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup — not on every request
model = CNNModel()
model.load_state_dict(torch.load("../models/cnn_model.pt", weights_only=True))
model.eval()

# Same transform as training — input must look identical to what the model learned on
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


@app.get("/")
def root():
    return {"status": "ok", "message": "Fashion-MNIST classifier is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess
    tensor = transform(image).unsqueeze(0)  # add batch dimension: (1, 1, 28, 28)

    # Inference
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze()  # convert logits to probabilities

    # Build response
    predicted_idx = probs.argmax().item()
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "confidence": round(probs[predicted_idx].item(), 4),
        "all_scores": {
            CLASS_NAMES[i]: round(probs[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        }
    }