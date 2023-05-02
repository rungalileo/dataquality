from typing import Dict

from dataquality.loggers.logger_config.tabular_classification import (
    tabular_classification_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger


class TabularClassificationModelLogger(BaseGalileoModelLogger):
    # NOTE: Tabular Classification doesn't require a model logger
    __logger_name__ = "tabular_classification"
    logger_config = tabular_classification_logger_config

    def _get_data_dict(self) -> Dict:
        return {}
