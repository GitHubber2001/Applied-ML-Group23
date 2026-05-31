from enum import Enum

import numpy as np
from sklearn.dummy import DummyClassifier


class RandomStrategy(Enum):
    most_frequent = "most_frequent"
    prior = "prior"
    stratified = "stratified"
    uniform = "uniform"
    constant = "constant"


class RandomGuesser:
    """A Benchmark model based on random/simple strategy guessing"""

    def __init__(self, strategy: RandomStrategy) -> None:
        if not isinstance(strategy, RandomStrategy):
            raise TypeError(
                f"argument must be of type RandomStrategy ({type(strategy)} was given)"
            )

        self._model = DummyClassifier(strategy=strategy.value)

    def fit(self, X, y):
        self._model.fit(X, y)

    def predict(self, X, y):
        self._model.fit(X, y)
        predictions = self._model.predict(X)

        return predictions

    def get_accuracy(self, X, y):
        accuracy = self._model.score(X, y)
        return accuracy


# temp test
X = np.array([-1, 1, 1, 1])
y = np.array([0, 1, 1, 1])

most_frequent_guesser = RandomGuesser(RandomStrategy.most_frequent)

most_frequent_guesser.fit(X, y)
predictions = most_frequent_guesser.predict(X, y)
accuracy = most_frequent_guesser.get_accuracy(predictions, y)

print(accuracy)
