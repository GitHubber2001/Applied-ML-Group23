
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


train_dir = './chest_xray/chest_xray/train/'

size = (256, 256)

def extract_data(folder, label):
    data = []
    path = os.path.join(folder, label)
    data_path = os.listdir(path)
    for sample in data_path:
        img = cv2.imread(os.path.join(path, sample), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, size)
            data.append(img.flatten())
    return np.array(data)

penu_df = extract_data(train_dir, 'PNEUMONIA')
normal_df = extract_data(train_dir, 'NORMAL')
data = np.vstack([penu_df, normal_df])

dev_dir = "./chest_xray/val/"

penu_df_dev = extract_data(dev_dir, "PNEUMONIA")
normal_df_dev = extract_data(dev_dir, "NORMAL")
dev_data = np.vstack([penu_df_dev, normal_df_dev])

test_dir = "./chest_xray/test/"

penu_df_test = extract_data(test_dir, "PNEUMONIA")
normal_df_test = extract_data(test_dir, "NORMAL")
test_data = np.vstack([penu_df_test, normal_df_test])

n_components = 648
ipca = IncrementalPCA(n_components=n_components)
ipca.fit(data)
compressed = ipca.transform(data)
new_data = ipca.inverse_transform(compressed)

compressed_dev = ipca.transform(dev_data)
compressed_test = ipca.transform(test_data)

joblib.dump(ipca, 'pca_transform.pkl')

image_df = pd.DataFrame(compressed)
dev_df = pd.DataFrame(compressed_dev)
test_df = pd.DataFrame(compressed_test)


n_penu = penu_df.shape[0]
n_normal = normal_df.shape[0]
labels = np.array([0] * n_penu + [1] * n_normal)
image_df['label'] = labels

n_penu = penu_df_dev.shape[0]
n_normal = normal_df_dev.shape[0]
labels = np.array([0] * n_penu + [1] * n_normal)
dev_df["label"] = labels

n_penu = penu_df_test.shape[0]
n_normal = normal_df_test.shape[0]
labels = np.array([0] * n_penu + [1] * n_normal)
test_df["label"] = labels

image_df.to_csv('image_data.csv', index=False)
dev_df.to_csv("dev_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

random_forest = RandomForestClassifier(class_weight="balanced")

training_data = np.genfromtxt("image_data.csv", delimiter=",")

class_label = training_data[1:, -1]
dataset = training_data[1:, :-1]

X_train, X_test, y_train, y_test = train_test_split(
    dataset, class_label, test_size=0.33, random_state=42
)

random_forest.fit(X_train, y_train)

prediction = random_forest.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, prediction)}")
joblib.dump(random_forest, 'random_forest.pkl')