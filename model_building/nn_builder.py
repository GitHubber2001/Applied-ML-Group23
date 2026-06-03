import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42

# based on .csv files
AMOUNT_INPUT_FEATURES = 648
AMOUNT_IMAGE_CHANNELS = 1

AMOUNT_CLASSES = 2

# later change with hyper parameter tuning
AMOUNT_TRAINING_EPOCHS = 30
BATCH_SIZE = 64
LEARNINGRATE = 1e-4
ACTIVATION_FUNCTION = nn.ReLU


def display_evaluation_metrics(model_name: str, y, predictions) -> None:
    print(f"Evaluation metric of {model_name}")

    test_report = classification_report(y, predictions, digits=4)
    print(test_report)

    test_confusion_matrix = confusion_matrix(y, predictions)
    print(test_confusion_matrix)


def set_random_state(random_seed) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(X: pd.DataFrame, y: pd.Series) -> DataLoader:
    x_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    return data_loader


def get_device() -> str:
    current_accelerator = torch.accelerator.current_accelerator(True)
    if current_accelerator is not None:
        device = current_accelerator.type
    else:
        device = "cpu"

    return device


def get_dataset_split():
    dev_df = pd.read_csv("data/dev_data.csv")
    image_df = pd.read_csv("data/image_data.csv")
    test_df = pd.read_csv("data/test_data.csv")

    combined_df = pd.concat([image_df, dev_df, test_df], ignore_index=True)
    X = combined_df.drop(columns=["label"])
    y = combined_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.name = "Neural Network"

        self.classifier = nn.Sequential(
            nn.Linear(AMOUNT_INPUT_FEATURES, 128),
            nn.BatchNorm1d(128),
            ACTIVATION_FUNCTION(),
            nn.Dropout(0.4),
            nn.Linear(128, AMOUNT_CLASSES),
        )

    def forward(self, x):
        x = self.classifier(x)

        return x


def main():
    set_random_state(RANDOM_STATE)

    model_file_path = "neural_network.pkl"

    device = get_device()
    X_train, X_test, y_train, y_test = get_dataset_split()

    model = NeuralNetwork().to(device)
    model.train()

    train_data_loader = get_dataloader(X_train, y_train)
    test_data_loader = get_dataloader(X_test, y_test)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNINGRATE)

    print("training")
    model.train()
    for epoch in range(AMOUNT_TRAINING_EPOCHS):
        print(f"training epoch: {epoch + 1}/{AMOUNT_TRAINING_EPOCHS}")

        for data in train_data_loader:
            optimizer.zero_grad()

            X, y = data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

    print("eval")
    all_test_predictions = []
    model.eval()
    with torch.no_grad():
        for data in test_data_loader:
            X, y = data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            predictions = outputs.argmax(1)
            all_test_predictions.extend(predictions.cpu().tolist())

    test_accuracy = accuracy_score(y_test, all_test_predictions)

    display_evaluation_metrics(model.name, y_test, all_test_predictions)

    joblib.dump(model, model_file_path)
    print(f"{model.name} model: SAVED (accuracy={test_accuracy})")


if __name__ == "__main__":
    main()
