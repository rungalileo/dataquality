from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class SemanticSegmentationLoggerConfig(BaseLoggerConfig):
    image_cloud_path: str = ""


semantic_segmentation_logger_config = SemanticSegmentationLoggerConfig()
