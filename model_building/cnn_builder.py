import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
AMOUNT_TRAINING_EPOCHS = 10

# based on .csv files
AMOUNT_INPUT_FEATURES = 648

AMOUNT_IMAGE_CHANNELS = 1
AMOUNT_CLASSES = 2

# later change with hyper parameter tuning
BATCH_SIZE = 64
LEARNINGRATE = 1e-4


def get_dataloader(X: pd.DataFrame, y: pd.Series) -> DataLoader:
    x_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return data_loader


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(AMOUNT_INPUT_FEATURES, 32)
        self.fc2 = nn.Linear(32, 16)

        self.fc3 = nn.Linear(16, AMOUNT_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


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


model_file_path = "cnn.pkl"

device = get_device()
X_train, X_test, y_train, y_test = get_dataset_split()

cnn_model = CNN().to(device)
cnn_model.train()

train_data_loader = get_dataloader(X_train, y_train)
test_data_loader = get_dataloader(X_test, y_test)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=LEARNINGRATE)

print("training")
for epoch in range(AMOUNT_TRAINING_EPOCHS):
    print(f"epoch: {epoch + 1}/{AMOUNT_TRAINING_EPOCHS}")

    for data in train_data_loader:
        X, y = data
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = cnn_model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

print("eval")
all_test_predictions = []
cnn_model.eval()
with torch.no_grad():
    for data in test_data_loader:
        X, y = data
        X = X.to(device)
        y = y.to(device)

        outputs = cnn_model(X)
        predictions = outputs.argmax(1)
        all_test_predictions.extend(predictions.cpu().tolist())


test_accuracy = accuracy_score(y_test, all_test_predictions)

print(f"Accuracy CNN: {test_accuracy}")

joblib.dump(cnn_model, model_file_path)
print(f"SAVED (accuracy={test_accuracy})")
