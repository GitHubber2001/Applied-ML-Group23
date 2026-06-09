import joblib
import numpy as np
import os
_rand_forest_path = os.path.join(os.path.dirname(__file__), "random_forest.pkl")
model = joblib.load(_rand_forest_path)


def predict(img: np.ndarray):
    label = model.predict(img)
    if label == 1:
        return "PNEUMONIA"
    elif label == 0:
        return "NORMAL"
