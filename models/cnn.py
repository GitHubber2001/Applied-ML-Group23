import joblib
import numpy as np

trained_cnn_model = joblib.load("trained_cnn.pkl")


class Trained_CNN:
    """A Trained CNN that can predict PNEUMONIA cases"""

    @staticmethod
    def predict(img: np.ndarray):
        prediction = trained_cnn_model.predict(img)

        if prediction == 0:
            return "PNEUMONIA"
        elif prediction == 1:
            return "NORMAL"
