from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class SemanticSegmentationLoggerConfig(BaseLoggerConfig):
    polygon_idx: int = 0


semantic_segmentation_logger_config = SemanticSegmentationLoggerConfig()
