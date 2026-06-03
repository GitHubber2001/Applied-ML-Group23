import copy
import random

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.study import Study
from optuna.trial import Trial
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42

# based on .csv files
AMOUNT_INPUT_FEATURES = 648
AMOUNT_IMAGE_CHANNELS = 1

AMOUNT_CLASSES = 2

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


def get_dataloader(X: pd.DataFrame, y: pd.Series, batch_size) -> DataLoader:
    x_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size)

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
            nn.Linear(AMOUNT_INPUT_FEATURES, 256),
            nn.BatchNorm1d(256),
            ACTIVATION_FUNCTION(),
            #
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            ACTIVATION_FUNCTION(),
            #
            nn.Linear(128, AMOUNT_CLASSES),
        )

    def forward(self, x):
        x = self.classifier(x)

        return x


def objective(
    trial: Trial,
    best_model_info: dict,
    device: str,
    X_train,
    X_test,
    y_train,
    y_test,
) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    amount_training_epochs = trial.suggest_int("amount_training_epochs", 10, 75)

    model = NeuralNetwork().to(device)

    train_data_loader = get_dataloader(X_train, y_train, batch_size)
    test_data_loader = get_dataloader(X_test, y_test, batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(amount_training_epochs):
        for data in train_data_loader:
            optimizer.zero_grad()

            X, y = data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

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

    test_accuracy = float(accuracy_score(y_test, all_test_predictions))
    best_accuracy = best_model_info["score"]

    if test_accuracy > best_accuracy:
        best_model_info["model"] = copy.deepcopy(model).to("cpu")
        best_model_info["score"] = test_accuracy
        best_model_info["predictions"] = all_test_predictions

    return test_accuracy


def main():
    set_random_state(RANDOM_STATE)

    model_file_path = "neural_network.pkl"

    device = get_device()
    X_train, X_test, y_train, y_test = get_dataset_split()

    best_model_info = {"model": None, "score": 0, "predictions": []}

    amount_optuna_trials = 30
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial, best_model_info, device, X_train, X_test, y_train, y_test
        ),
        n_trials=amount_optuna_trials,
    )

    print("best model params:")
    print(study.best_params)

    best_model = best_model_info["model"]
    best_model_predictions = best_model_info["predictions"]

    test_accuracy = accuracy_score(y_test, best_model_predictions)

    display_evaluation_metrics(best_model.name, y_test, best_model_predictions)

    joblib.dump(best_model, model_file_path)
    print(f"{best_model.name} model: SAVED (accuracy={test_accuracy})")


if __name__ == "__main__":
    main()
