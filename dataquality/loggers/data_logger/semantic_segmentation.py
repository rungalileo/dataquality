import os
from typing import Any, List, Optional, Union

import vaex

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
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
from dataquality.utils.vaex import get_output_df

# smaller than ITER_CHUNK_SIZE from base_data_logger because very large chunks
# containing image data often won't fit in memory

ITER_CHUNK_SIZE_IMAGES = 10000


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

    def upload_split(
        self,
        location: str,
        split: str,
        object_store: ObjectStore,
        last_epoch: Optional[int],
        create_data_embs: bool,
    ) -> None:
        split_loc = f"{location}/{split}"
        output_logged = os.path.exists(split_loc)
        if not output_logged:
            return
        dir_name = f"{split_loc}/0"
        out_frame = get_output_df(
            dir_name,
            prob_only=False,
            split=split,
            epoch_or_inf=0,  # For SemSeg we only have one epoch, the final pass
        )
        if "id" not in out_frame.get_column_names():
            out_frame["id"] = vaex.vrange(0, len(out_frame), dtype="int")
        minio_file = f"{self.proj_run}/{split}/data/data.hdf5"
        object_store.create_project_run_object_from_df(
            df=out_frame,
            object_name=minio_file,
            bucket_name=config.results_bucket_name,
        )
