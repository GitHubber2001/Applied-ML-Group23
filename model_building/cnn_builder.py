import random
from typing import Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms

RANDOM_STATE = 42

NORMALIZED_IMAGE_SIZE = (256, 256)
AMOUNT_IMAGE_CHANNELS = 1

AMOUNT_CLASSES = 2

BATCH_SIZE = 128
AMOUNT_TRAINING_EPOCHS = 25
ACTIVATION_FUNCTION = nn.ReLU


def set_random_state(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    return device


def get_datasets() -> Tuple[Subset, Subset, Subset]:
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(
                NORMALIZED_IMAGE_SIZE,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    train_dataset = datasets.ImageFolder(root="chest_xray/train", transform=transform)
    validation_dataset = datasets.ImageFolder(
        root="chest_xray/val", transform=transform
    )
    test_dataset = datasets.ImageFolder(root="chest_xray/test", transform=transform)

    full_dataset = ConcatDataset([train_dataset, validation_dataset, test_dataset])

    all_indexes = np.arange(len(full_dataset))
    all_y = []
    for dataset in [train_dataset, validation_dataset, test_dataset]:
        all_y.extend(dataset.targets)

    indexes_train, indexes_test, y_train, y_test = train_test_split(
        all_indexes,
        all_y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=all_y,
    )

    indexes_validation, indexes_test, y_validation, y_test = train_test_split(
        indexes_test,
        y_test,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_test,
    )

    train_dataset = Subset(full_dataset, indexes_train)
    validation_dataset = Subset(full_dataset, indexes_validation)
    test_dataset = Subset(full_dataset, indexes_test)

    return train_dataset, validation_dataset, test_dataset


def get_data_loaders(
    train_dataset: Subset, validation_dataset: Subset, test_dataset: Subset
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_data_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    validation_data_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_data_loader, validation_data_loader, test_data_loader


def display_evaluation_metrics(model_name: str, y, predictions) -> None:
    print(f"Evaluation metric of {model_name}")

    test_report = classification_report(
        y, predictions, digits=4, target_names=["NORMAL", "PNEUMONIA"]
    )
    print(test_report)

    test_confusion_matrix = confusion_matrix(y, predictions)
    print(test_confusion_matrix)


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.name = "CNN"

        self.convolution = nn.Sequential(
            nn.Conv2d(AMOUNT_IMAGE_CHANNELS, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            ACTIVATION_FUNCTION(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            ACTIVATION_FUNCTION(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * (64 * 64), 256),
            nn.BatchNorm1d(256),
            ACTIVATION_FUNCTION(),
            nn.Dropout(0.3),
            #
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            ACTIVATION_FUNCTION(),
            nn.Dropout(0.2),
            #
            nn.Linear(128, AMOUNT_CLASSES),
        )

    def forward(self, x):
        x = self.convolution(x)
        x = self.classifier(x)

        return x


def main():
    set_random_state(RANDOM_STATE)

    device = get_device()

    train_dataset, validation_dataset, test_dataset = get_datasets()
    train_dataloader, validation_dataloader, test_dataloader = get_data_loaders(
        train_dataset, validation_dataset, test_dataset
    )

    # assertions because wrong type errors
    assert isinstance(train_dataset.dataset, ConcatDataset)

    all_targets = []
    for dataset in train_dataset.dataset.datasets:
        assert isinstance(dataset, datasets.ImageFolder)
        all_targets.extend(dataset.targets)
    all_targets = np.array(all_targets)

    y_train = all_targets[train_dataset.indices]

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=AMOUNT_TRAINING_EPOCHS
    )

    model.train()
    for epoch in range(AMOUNT_TRAINING_EPOCHS):
        print(f"\ntrain epoch {epoch + 1}/{AMOUNT_TRAINING_EPOCHS} - started")

        running_loss = 0.0
        for data in train_dataloader:
            optimizer.zero_grad()

            X, y = data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        lr_scheduler.step()

        avg_train_loss = running_loss / len(train_dataloader)

        print(
            f"train epoch {epoch + 1}/{AMOUNT_TRAINING_EPOCHS} - loss: {avg_train_loss:.4f}"
        )

    all_test_y = []
    all_test_predictions = []
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            X, y = data
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            predictions = outputs.argmax(1)

            all_test_y.extend(y.cpu().tolist())
            all_test_predictions.extend(predictions.cpu().tolist())

    test_accuracy = np.mean(np.array(all_test_predictions) == np.array(all_test_y))
    display_evaluation_metrics(model.name, all_test_y, all_test_predictions)
    
    model_file_path = "models/cnn.pkl"
    model.to("cpu")
    joblib.dump(model, model_file_path)
    print(f"{model.name} model: SAVED (accuracy={test_accuracy})")


if __name__ == "__main__":
    print("START")
    main()
    print("END")
