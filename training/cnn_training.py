import os
import sys

file = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if file not in sys.path:
    sys.path.insert(0, file)

from utilities.timer import TimeManager

with TimeManager("Imports"):
    import joblib
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

BATCH_SIZE = 64


def main():
    # use GPU
    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    with TimeManager("Datasets"):
        dev_df = pd.read_csv("Notebooks/dev_data.csv")
        image_df = pd.read_csv("Notebooks/image_data.csv")
        test_df = pd.read_csv("Notebooks/test_data.csv")

        combined_df = pd.concat([image_df, dev_df, test_df], ignore_index=True)
        x = combined_df.drop(columns=["label"])
        y = combined_df["label"]

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.33, random_state=42, stratify=y
        )

    with TimeManager("Training"):
        # temporary placeholder model
        cnn_model = models.mobilenet_v3_small()
        cnn_model.train()

    with TimeManager("Evaluating"):
        cnn_model.eval()
        with torch.no_grad():
            pass

        # temp
        logits = None
        probabilities = torch.nn.functional.softmax(logits)
        predictions = (probabilities > 0.5).int()

        print(f"\n\nAccuracy: {accuracy_score(y_test, predictions)}\n\n")

    with TimeManager("Saving trained model"):
        # joblib.dump(cnn_model, "trained_cnn.pkl")
        pass


if __name__ == "__main__":
    with TimeManager("Program", True):
        main()
