from enum import Enum

import numpy as np
from sklearn.dummy import DummyClassifier


class RandomStrategy(Enum):
    most_frequent = "most_frequent"
    stratified = "stratified"
    uniform = "uniform"


class RandomGuesser:
    """A Benchmark model based on random strategies guessing"""

    def __init__(self, strategy: RandomStrategy) -> None:
        if not isinstance(strategy, RandomStrategy):
            raise TypeError(
                f"argument must be of type RandomStrategy ({type(strategy)} was given)"
            )

        self._model = DummyClassifier(strategy=strategy.value)

    def fit(self, X, y) -> None:
        self._model.fit(X, y)

    def predict(self, X, y) -> np.ndarray:
        self._model.fit(X, y)
        predictions = self._model.predict(X)

        return predictions

    def get_accuracy(self, X, y) -> float:
        accuracy = self._model.score(X, y)
        return accuracy


def display_random_strategies_performances(X, y) -> None:
    print("-" * 30)
    print("accuracies random strategies: ")

    for strategy in RandomStrategy:
        model = RandomGuesser(strategy)

        model.fit(X, y)
        predictions = model.predict(X, y)
        accuracy = model.get_accuracy(predictions, y)

        print(f"{strategy.value}: {accuracy:.4f}")

    print("-" * 30)
