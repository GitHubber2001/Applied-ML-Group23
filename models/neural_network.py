import joblib
import numpy as np

trained_model = joblib.load("neural_network.py")


class Trained_Neural_Network:
    """A Trained CNN that can predict PNEUMONIA cases"""

    @staticmethod
    def predict(img: np.ndarray):
        prediction = trained_model.predict(img)

        if prediction == 0:
            return "PNEUMONIA"
        elif prediction == 1:
            return "NORMAL"
