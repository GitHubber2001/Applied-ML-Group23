import os
import sys
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_network import predict
from models.pca import pca_transform

SIZE = (256, 256)


class PredictionResponse(BaseModel):
    """Response model for pneumonia classification prediction."""
    prediction: str


app = FastAPI()


def process_input(img: np.ndarray):
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, SIZE)
    img_flat = img.flatten().reshape(1, -1)
    img = pca_transform(img_flat)
    return img


@app.post("/classify_pneumonia", response_model=PredictionResponse)
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Classify chest X-ray image for pneumonia detection.
    
    - **file**: Chest X-ray image file (grayscale or RGB)
    
    Returns pneumonia classification prediction.
    """
    image = await file.read()
    tansformed_image = process_input(image)
    prediction = predict(tansformed_image)
    return {"prediction": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
