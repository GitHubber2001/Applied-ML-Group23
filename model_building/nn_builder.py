import random

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from optuna.trial import Trial
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
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


def set_random_state(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataloader(
    X: pd.DataFrame, y: pd.Series, batch_size: int, device: str
) -> DataLoader:
    x_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y.values, dtype=torch.long).to(device)
    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_dataset_split():
    
    image_df = pd.read_csv("data/image_data_relabeled.csv")

    X = image_df.drop(columns=["label"])
    y = image_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test


class NeuralNetwork(nn.Module):
    def __init__(self, fc1_size, fc2_size) -> None:
        super().__init__()

        self.name = "Neural Network"

        self.classifier = nn.Sequential(
            nn.Linear(AMOUNT_INPUT_FEATURES, fc1_size),
            nn.BatchNorm1d(fc1_size),
            ACTIVATION_FUNCTION(),
            #
            nn.Linear(fc1_size, fc2_size),
            nn.BatchNorm1d(fc2_size),
            ACTIVATION_FUNCTION(),
            #
            nn.Linear(fc2_size, AMOUNT_CLASSES),
        )

    def forward(self, x):
        x = self.classifier(x)

        return x


def train_model(
    model: NeuralNetwork,
    criterion,
    optimizer,
    amount_training_epochs,
    data_loader,
):
    model.train()
    for epoch in range(amount_training_epochs):
        total_epoch_loss = 0.0
        for data in data_loader:
            optimizer.zero_grad()

            X, y = data

            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()


def objective(
    trial: Trial,
    device: str,
    X_train,
    y_train,
) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    amount_training_epochs = trial.suggest_int("amount_training_epochs", 20, 50)
    fc1_size = trial.suggest_categorical("fc1_size", [64, 128, 256])
    fc2_size = trial.suggest_categorical("fc2_size", [16, 32, 64, 128])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        X_train_fold, X_val_fold = (
            X_train.iloc[train_index],
            X_train.iloc[val_index],
        )
        y_train_fold, y_val_fold = (
            y_train.iloc[train_index],
            y_train.iloc[val_index],
        )

        model = NeuralNetwork(fc1_size, fc2_size).to(device)

        train_data_loader = get_dataloader(
            X_train_fold, y_train_fold, batch_size, device
        )
        val_data_loader = get_dataloader(X_val_fold, y_val_fold, batch_size, device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, fused=True)

        train_model(
            model,
            criterion,
            optimizer,
            amount_training_epochs,
            train_data_loader,
        )

        all_test_predictions = []
        model.eval()
        with torch.no_grad():
            for data in val_data_loader:
                X, y = data

                outputs = model(X)
                predictions = outputs.argmax(1)
                all_test_predictions.extend(predictions.tolist())

        fold_accuracy = float(accuracy_score(y_val_fold, all_test_predictions))
        fold_accuracies.append(fold_accuracy)

    aggregated_accuracy = float(np.mean(fold_accuracies))

    return aggregated_accuracy


def main():
    set_random_state(RANDOM_STATE)

    model_file_path = "neural_network.pkl"

    device = get_device()

    X_train, X_test, y_train, y_test = get_dataset_split()

    amount_optuna_trials = 20
    study = optuna.create_study(
        direction="maximize",
    )

    study.optimize(
        lambda trial: objective(trial, device, X_train, y_train),
        n_trials=amount_optuna_trials,
    )

    print("best model params:")
    print(study.best_params)

    learning_rate = study.best_params["learning_rate"]
    batch_size = study.best_params["batch_size"]
    amount_training_epochs = study.best_params["amount_training_epochs"]
    fc1_size = study.best_params["fc1_size"]
    fc2_size = study.best_params["fc2_size"]

    train_data_loader = get_dataloader(X_train, y_train, batch_size, device)
    test_data_loader = get_dataloader(X_test, y_test, batch_size, device)

    model = NeuralNetwork(fc1_size, fc2_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, fused=True)

    train_model(model, criterion, optimizer, amount_training_epochs, train_data_loader)

    all_test_predictions = []
    model.eval()
    with torch.no_grad():
        for data in test_data_loader:
            X, y = data

            outputs = model(X)
            predictions = outputs.argmax(1)
            all_test_predictions.extend(predictions.cpu().tolist())

    test_accuracy = accuracy_score(y_test, all_test_predictions)

    display_evaluation_metrics(model.name, y_test, all_test_predictions)

    model.to("cpu")
    torch.save(model.state_dict(), "neural_network.pt")
    print(f"{model.name} model: SAVED (accuracy={test_accuracy})")


if __name__ == "__main__":
    main()
