import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.forest_model import predict
from models.pca import pca_transform

SIZE = (256, 256)


app = FastAPI()


def process_input(img: np.ndarray):
    nparr = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, SIZE)
    img_flat = img.flatten().reshape(1, -1)
    img = pca_transform(img_flat)
    return img


@app.post("/classify_pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    image = await file.read()
    tansformed_image = process_input(image)
    prediction = predict(tansformed_image)
    return {"pridiction": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
