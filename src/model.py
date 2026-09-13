from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression


class ClassifierService:
    def __init__(self) -> None:
        features, labels = make_classification(
            n_samples=300,
            n_features=4,
            n_informative=3,
            n_redundant=0,
            n_classes=3,
            random_state=7,
        )
        self.model = LogisticRegression(max_iter=500, random_state=7).fit(features, labels)

    def predict(self, features: list[float]) -> tuple[int, list[float]]:
        if len(features) != 4:
            raise ValueError('exactly four features are required')
        row = np.asarray(features, dtype=float).reshape(1, -1)
        probabilities = self.model.predict_proba(row)[0]
        return int(self.model.classes_[probabilities.argmax()]), probabilities.tolist()

    @property
    def metadata(self) -> dict[str, object]:
        return {'model': 'LogisticRegression', 'feature_count': 4, 'classes': 3}
