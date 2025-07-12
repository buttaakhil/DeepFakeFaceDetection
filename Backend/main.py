from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from model import load_model
from utils import preprocess_image
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://deep-fake-face-detection.vercel.app"],  # ✅ Use your frontend domain here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_tensor = preprocess_image(file.file)
        with torch.no_grad():
            output = model(image_tensor)
            prob = output.item()
            prob = 1 - prob
            fake_percentage = round(prob * 100, 2)

        return {
            "percentage_morphed": fake_percentage
        }
    except Exception as e:
        return {"error": str(e)}
