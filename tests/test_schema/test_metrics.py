from dataquality.schemas.metrics import FilterParams


def test_nested_filter_params_are_hashable() -> None:
    f2 = {
        "misclassified_only": True,
        "ids": [1, 2, 3, 4],
        "lasso": {"x": [0.1, 0.1, 0.2], "y": [0.4, 0.5, 0.6]},
        "inference_filter": {"is_otb": True},
        "meta_filter": [{"name": "foo", "isin": [1, 2, 3]}],
    }
    assert FilterParams(**f2).__hash__()
