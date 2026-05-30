import joblib
import numpy as np

model = joblib.load("random_forest.pkl")


def predict(img: np.ndarray):
    label = model.predict(img)
    if label == 0:
        return "PNEUMONIA"
    elif label == 1:
        return "NORMAL"
