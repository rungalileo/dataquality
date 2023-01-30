from typing import Callable, Dict, Optional

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from dataquality.loggers.data_logger.structured_classification import (
    StructuredClassificationDataLogger,
)


@pytest.fixture(scope="module")
def sc_dataset() -> Dict:
    wine = load_wine()

    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    n_classes = wine.target_names.shape[0]
    train_probs = np.random.rand(X_train.shape[0], n_classes)
    test_probs = np.random.rand(X_test.shape[0], n_classes)
    return {
        "feature_names": wine.feature_names,
        "labels": wine.target_names,
        "training": {"X": X_train, "y": y_train, "probs": train_probs},
        "test": {"X": X_test, "y": y_test, "probs": test_probs},
        "inf1": {
            "X": X_test,
            "probs": test_probs,
        },  # It's fine for inference data to be test data
        "inf2": {"X": X_test, "probs": test_probs},
    }


@pytest.fixture(scope="module")
def fit_xgboost(sc_dataset: Dict) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(objective="multi:softprob", random_state=42)
    model.fit(sc_dataset["training"]["X"], sc_dataset["training"]["y"])
    return model


@pytest.fixture
def sc_data_logger(sc_dataset: Dict, fit_xgboost: xgb.XGBClassifier) -> Callable:
    def curry(
        split: Optional[str] = None, inference_name: Optional[str] = None
    ) -> StructuredClassificationDataLogger:
        if not split:
            return StructuredClassificationDataLogger()

        key = inference_name if split == "inference" else split
        logger = StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_dataset[key]["X"],
            y=sc_dataset[key].get("y"),  # We use get since inf data doesn't have y
        )
        logger.dataset = logger.create_dataset_from_samples(
            X=logger.X, y=logger.y, feature_names=sc_dataset["feature_names"]
        )
        logger.split = split
        logger.inference_name = inference_name
        return logger

    return curry
