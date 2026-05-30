import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# use GPU
current_accelerator = torch.accelerator.current_accelerator(True)
if current_accelerator is not None:
    device = current_accelerator.type
else:
    device = "cpu"

# datasets
dev_df = pd.read_csv("dev_data.csv")
image_df = pd.read_csv("image_data.csv")
test_df = pd.read_csv("test_data.csv")

combined_df = pd.concat([image_df, dev_df, test_df], ignore_index=True)
x = combined_df.drop(columns=["label"])
y = combined_df["label"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42, stratify=y
)

# model training saving (NOT FINISHED!)

baseline_cnn = models.mobilenet_v3_small()  # placeholder

baseline_cnn.train()
