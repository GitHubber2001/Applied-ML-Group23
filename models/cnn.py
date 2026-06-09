import joblib
import numpy as np
import os
_cnn_path = os.path.join(os.path.dirname(__file__), "cnn.pkl")
model = joblib.load(_cnn_path)

def predict(img: np.ndarray):
    label = model.predict(img)
    if label == 1:
        return "PNEUMONIA"
    elif label == 0:
        return "NORMAL"
