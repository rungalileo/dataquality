from typing import Optional, Union

# from dataquality.loggers.base_logger import BaseGalileoLogger
from dataquality.loggers.data_logger.base_data_logger import (
    BaseGalileoDataLogger,
    DataSet,
)
from dataquality.loggers.logger_config.structured_classification import (
    structured_classification_logger_config,
)
from dataquality.schemas.split import Split


class StructuredClassificationLogger(BaseGalileoDataLogger):
    __logger_name__ = "structured_classification"
    logger_config = structured_classification_logger_config

    def __init__(self) -> None:
        super().__init__()

    def log_structured_dataset(
        self,
        dataset: DataSet,
        label: Union[str, int] = "label",
        split: Optional[Split] = None,
    ) -> None:
        print("hi there my friend")
