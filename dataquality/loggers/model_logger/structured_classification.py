from typing import Dict

from dataquality.loggers.logger_config.structured_classification import (
    structured_classification_logger_config,
)
from dataquality.loggers.model_logger.base_model_logger import BaseGalileoModelLogger


class StructuredClassificationModelLogger(BaseGalileoModelLogger):
    # NOTE: Structured Classification doesn't require a model logger
    __logger_name__ = "structured_classification"
    logger_config = structured_classification_logger_config

    def _get_data_dict(self) -> Dict:
        return {}
