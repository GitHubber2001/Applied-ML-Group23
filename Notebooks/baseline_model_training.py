import datetime
import json
import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split

RANDOM_STATE = 42


def display_frequency_labels(labels: pd.Series) -> None:
    test_freqency_classes = labels.value_counts(normalize=False)
    test_ratio_classes = labels.value_counts(normalize=True)

    print(test_freqency_classes)
    print(test_ratio_classes)


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


model_file_path = "random_forest.pkl"
model_info_file_path = "random_forest_info.json"

X_train, X_test, y_train, y_test = get_dataset_split()

search_parameters = {
    "n_estimators": list(range(20, 40, 5)),
    "max_depth": list(range(20, 50, 5)),
}

search_model = RandomForestClassifier(random_state=RANDOM_STATE)
search = GridSearchCV(search_model, param_grid=search_parameters, cv=3, verbose=2)
search.fit(X_train, y_train)

best_random_forest = search.best_estimator_
predictions = best_random_forest.predict(X_test)  # type: ignore
test_accuracy = accuracy_score(y_test, predictions)

best_parameters = {}
for hyper_parameter, value in best_random_forest.get_params().items():
    if hyper_parameter in search_parameters:
        best_parameters[hyper_parameter] = value

best_model_info = {
    "saving_datetime": str(datetime.datetime.now()),
    "test_accuracy": test_accuracy,
    "hyperparameters": best_parameters,
}

sorted_results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")[
    ["rank_test_score", "mean_test_score", "params"]
]
print(sorted_results)

if os.path.exists(model_info_file_path):
    with open(model_info_file_path, "r") as file:
        previous_model_info = json.load(file)
        previous_test_accuracy = previous_model_info["test_accuracy"]
else:
    previous_test_accuracy = -1

if test_accuracy > previous_test_accuracy:
    # save best model
    joblib.dump(best_random_forest, model_file_path)

    # save best hyper parameters and accuracy
    with open(model_info_file_path, "w", encoding="utf-8") as file:
        json.dump(best_model_info, file)

    print(f"SAVED ({test_accuracy} > {previous_test_accuracy})")
else:
    print(f"NOT SAVED ({test_accuracy} <= {previous_test_accuracy})")
