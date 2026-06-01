import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

RANDOM_STATE = 42

dev_df = pd.read_csv("data/dev_data.csv")
image_df = pd.read_csv("data/image_data.csv")
test_df = pd.read_csv("data/test_data.csv")

combined_df = pd.concat([image_df, dev_df, test_df], ignore_index=True)
X = combined_df.drop(columns=["label"])
y = combined_df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=RANDOM_STATE, stratify=y
)

# for getting accuracy highest frequency guessing
test_freqency_classes = y_test.value_counts(normalize=False)
test_ratio_classes = y_test.value_counts(normalize=True)
print(test_freqency_classes)
print(test_ratio_classes)

search_parameters = {
    "n_estimators": list(range(24, 27, 1)),
    "max_depth": [None, 30, 35, 40],
}

search_model = RandomForestClassifier(random_state=RANDOM_STATE)

search = GridSearchCV(search_model, param_grid=search_parameters, cv=3, verbose=3)
search.fit(X_train, y_train)

best_random_forest = search.best_estimator_
predictions = best_random_forest.predict(X_test)  # type: ignore

sorted_results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")[
    ["rank_test_score", "mean_test_score", "params"]
]
print(sorted_results)

print(f"Accuracy random forest: {accuracy_score(y_test, predictions)}")

joblib.dump(best_random_forest, "random_forest.pkl")
