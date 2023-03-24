from enum import Enum

from dataquality.loggers.logger_config.base_logger_config import BaseLoggerConfig


class BoxFormat(str, Enum):
    """Format for bounding box representation"""

    xyxy = "xyxy"  # xmin, ymin, xmax, ymax
    tlxywh = "tlxywh"  # top left x, top left y, width, height
    cxywh = "cxywh"  # center x, center y, width, height


class ObjectDetectionLoggerConfig(BaseLoggerConfig):
    image_cloud_path: str = ""
    box_format: str = BoxFormat.xyxy


object_detection_logger_config = ObjectDetectionLoggerConfig()
