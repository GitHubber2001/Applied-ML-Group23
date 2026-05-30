import joblib
import numpy as np

cnn_model = joblib.load("_.pkl")

def predict(img: np.ndarray):
    prediction = cnn_model.predict(img)

    if prediction == 0:
        return "PNEUMONIA"
    elif prediction == 1:
        return "NORMAL"
