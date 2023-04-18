from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig
from typing import Callable


class SemanticSegmentationLoggerConfig(BaseLoggerConfig):
    image_cloud_path: str = ""
    finish: Callable = None


semantic_segmentation_logger_config = SemanticSegmentationLoggerConfig()
