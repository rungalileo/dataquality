from typing import Callable, Dict, Optional

import pandas as pd
import pytest
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from dataquality.loggers.data_logger.structured_classification import (
    StructuredClassificationDataLogger,
)


@pytest.fixture(scope="module")
def sc_data() -> Dict:
    """Structured Classification data for tests"""
    wine = load_wine()

    X = wine.data
    y = wine.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    feature_names = [f"feature_{i}" for i in range(len(wine.feature_names))]
    train_df = pd.DataFrame(X_train, columns=feature_names)
    test_df = pd.DataFrame(X_test, columns=feature_names)

    return {
        "feature_names": feature_names,
        "labels": wine.target_names,
        "training": {"X": X_train, "y": y_train, "df": train_df},
        "test": {"X": X_test, "y": y_test, "df": test_df},
        "inf1": {
            "X": X_test,
            "df": test_df,
        },  # It's fine for inference data to be test data
        "inf2": {"X": X_test, "df": test_df},
    }


@pytest.fixture(scope="module")
def fit_xgboost(sc_data: Dict) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.fit(sc_data["training"]["X"], sc_data["training"]["y"])
    return model


@pytest.fixture
def create_logger(sc_data: Dict, fit_xgboost: xgb.XGBClassifier) -> Callable:
    def curry(
        split: str = "training", inference_name: Optional[str] = None
    ) -> StructuredClassificationDataLogger:
        key = inference_name if split == "inference" else split
        return StructuredClassificationDataLogger(
            model=fit_xgboost,
            X=sc_data[key]["X"],
            y=sc_data[key].get("y"),  # We use get since inf data doesn't have y
            feature_names=sc_data["feature_names"],
            split=split,
            inference_name=inference_name,
        )

    return curry
