import os

import joblib
import numpy as np

# Get path to pca_transform.pkl file in models directory
_pca_path = os.path.join(os.path.dirname(__file__), "pca_transform.pkl")
ipca = joblib.load(_pca_path)


def pca_transform(img: np.ndarray) -> np.ndarray:
    img = ipca.transform(img)
    return img
