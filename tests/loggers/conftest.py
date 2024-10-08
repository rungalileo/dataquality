from typing import Callable, Dict, Optional

import pandas as pd
import pytest
import torch
import xgboost as xgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from dataquality.loggers.data_logger.tabular_classification import (
    TabularClassificationDataLogger,
)


@pytest.fixture(scope="module")
def tab_data() -> Dict:
    """Tabular Classification data for tests"""
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
def fit_xgboost(tab_data: Dict) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.fit(tab_data["training"]["X"], tab_data["training"]["y"])
    return model


@pytest.fixture
def create_logger(tab_data: Dict, fit_xgboost: xgb.XGBClassifier) -> Callable:
    def curry(
        split: str = "training", inference_name: Optional[str] = None
    ) -> TabularClassificationDataLogger:
        key = inference_name if split == "inference" else split
        return TabularClassificationDataLogger(
            model=fit_xgboost,
            X=tab_data[key]["X"],
            y=tab_data[key].get("y"),  # We use get since inf data doesn't have y
            feature_names=tab_data["feature_names"],
            split=split,
            inference_name=inference_name,
        )

    return curry


@pytest.fixture
def seq2seq_generated_sample_output() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4, 5, 6]])
