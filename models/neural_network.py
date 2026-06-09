import joblib
import numpy as np
import os
from model_building.nn_builder import NeuralNetwork
import torch
import torch.nn as nn
fc1_size = 256
fc2_size = 64
_nn_path = os.path.join(os.path.dirname(__file__), "neural_network.pt")

model = NeuralNetwork(fc1_size,fc2_size)
model.load_state_dict(torch.load(_nn_path, map_location="cpu"))
model.eval()

def predict(img: np.ndarray):
    with torch.no_grad():
        x = torch.tensor(img, dtype=torch.float32)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()


    if pred == 1:
        return "PNEUMONIA"
    elif pred == 0:
        return "NORMAL"
