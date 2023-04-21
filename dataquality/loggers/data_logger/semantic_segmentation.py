from enum import Enum
from typing import Any, List, Optional, Union

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
    BaseGalileoDataLogger,
    DataSet,
    MetasType,
)
from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.schemas.split import Split

# smaller than ITER_CHUNK_SIZE from base_data_logger because very large chunks
# containing image data often won't fit in memory

ITER_CHUNK_SIZE_IMAGES = 10000


class SemSegCols(str, Enum):
    image_path = "image_path"
    mask_path = "mask_path"
    id = "id"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging


class SemanticSegmentationDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "semantic_segmentation"
    logger_config: SemanticSegmentationLoggerConfig = (
        semantic_segmentation_logger_config
    )

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        labels: Optional[List[int]] = None,
        ids: Optional[List[int]] = None,
        split: Optional[str] = None,
        meta: Optional[MetasType] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            meta=meta,
        )

    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        text: Union[str, int] = "text",
        id: Union[str, int] = "id",
        split: Optional[Split] = None,
        meta: Optional[List[Union[str, int]]] = None,
        **kwargs: Any,
    ) -> None:
        raise GalileoException(
            "Semantic Segmentation does not support log_dataset. "
            "Use watch(model, [dataloaders])"
        )
