from typing import Callable

import dataquality
from dataquality.loggers import BaseGalileoLogger
from dataquality.loggers.data_logger import BaseGalileoDataLogger
from dataquality.loggers.model_logger import BaseGalileoModelLogger


def test_attribute_subsets() -> None:
    """All potential logging fields used by all subclass loggers should be encapsulated

    Any new logger that is created has a set of attributes that it expects from users.
    The `BaseLoggerAttributes` from the BaseGalileoLogger should be the superset of
    all child loggers.
    """
    all_attrs = set(BaseGalileoLogger.get_valid_attributes())
    sub_data_loggers = BaseGalileoDataLogger.__subclasses__()
    data_logger_attrs = set(
        [j for i in sub_data_loggers for j in i.get_valid_attributes()]
    )
    sub_model_loggers = BaseGalileoModelLogger.__subclasses__()
    model_logger_attrs = set(
        [j for i in sub_model_loggers for j in i.get_valid_attributes()]
    )
    all_sub_attrs = data_logger_attrs.union(model_logger_attrs)
    assert all_attrs.issuperset(
        all_sub_attrs
    ), f"Missing attrs: {all_sub_attrs - all_attrs}"


def test_int_labels(set_test_config: Callable) -> None:
    dataquality.set_labels_for_run(labels=[1, 2, 3, 4, 5])  # type: ignore
    assert dataquality.get_data_logger().logger_config.labels == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
