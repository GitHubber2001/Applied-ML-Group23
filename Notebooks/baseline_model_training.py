import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dev_df = pd.read_csv("data/dev_data.csv")
image_df = pd.read_csv("data/image_data.csv")
test_df = pd.read_csv("data/test_data.csv")

combined_df = pd.concat([image_df, dev_df, test_df], ignore_index=True)
X = combined_df.drop(columns=["label"])
y = combined_df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

test_freqency_classes = y_test.value_counts(normalize=False)
test_ratio_classes = y_test.value_counts(normalize=True)

print(test_freqency_classes)
print(test_ratio_classes)

random_forest = RandomForestClassifier(class_weight="balanced")

random_forest.fit(X_train, y_train)

prediction = random_forest.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, prediction)}")
joblib.dump(random_forest, "random_forest.pkl")
