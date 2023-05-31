from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import vaex

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    BaseGalileoDataLogger,
    DataSet,
    MetasType,
)
from dataquality.loggers.logger_config.semantic_segmentation import (
    SemanticSegmentationLoggerConfig,
    semantic_segmentation_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.semantic_segmentation import SemSegCols
from dataquality.schemas.split import Split
from dataquality.utils.thread_pool import lock
from dataquality.utils.vaex import get_output_df

# smaller than ITER_CHUNK_SIZE from base_data_logger because very large chunks
# containing image data often won't fit in memory

ITER_CHUNK_SIZE_IMAGES = 1000


class SemanticSegmentationDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "semantic_segmentation"
    logger_config: SemanticSegmentationLoggerConfig = (
        semantic_segmentation_logger_config
    )

    INPUT_DATA_FILE_EXT = "hdf5"

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
        batch_size: int = ITER_CHUNK_SIZE_IMAGES,
        text: Union[str, int] = "text",
        image: Union[str, int] = SemSegCols.image.value,
        id: Union[str, int] = SemSegCols.id.value,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Union[List[str], List[int], None] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(dataset, dict):
            raise GalileoException(
                f"Dataset must be a dict for Semantic Segmentation, "
                f"but got type {type(dataset)}"
            )

        column_map = {
            id: SemSegCols.id.value,
            image: SemSegCols.image.value,
        }
        for old_key, new_key in column_map.items():
            if old_key in dataset:
                dataset[new_key] = dataset.pop(old_key)

        metas = defaultdict(list)
        for meta_col in meta or []:
            metas[meta_col].extend(dataset[meta_col])

        self._log_dict(dataset, meta=metas, split=split, inference_name=inference_name)

    def _log_dict(
        self,
        d: Dict,
        meta: Dict,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        self.log_image_samples(
            images=d[SemSegCols.image.value],
            ids=d[SemSegCols.id.value],
            split=split,
            meta=meta,
            inference_name=inference_name,
        )

    def log_image_samples(
        self,
        *,
        images: List[str],
        ids: List[int],
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[MetasType] = None,
        **kwargs: Any,  # For typing
    ) -> None:
        """Log input samples for semseg"""
        self.validate_kwargs(kwargs)
        self.images = images
        self.ids = ids
        self.split = split
        self.inference_name = inference_name
        self.meta = meta or {}
        self.log()

    def export_df(self, df: vaex.DataFrame) -> None:
        """Export the dataframe and increment the input_data_logged
        in this helper in order to allow for overrides in child classes.

        For instance semseg needs to do this in a multithreaded way and
        add locks to avoid threading issues
        """
        with lock:
            super().export_df(df)

    def _get_input_df(self) -> vaex.DataFrame:
        df_len = len(self.ids)
        inp = dict(
            id=self.ids,
            image=self.images,
            split=[Split(self.split).value] * df_len,
            data_schema_version=[__data_schema_version__] * df_len,
            **self.meta,
        )
        if self.split == Split.inference:
            inp["inference_name"] = [self.inference_name] * df_len

        return vaex.from_dict(inp)

    @classmethod
    def upload_split_from_in_frame(
        cls,
        object_store: ObjectStore,
        in_frame: vaex.DataFrame,
        split: str,
        split_loc: str,
        last_epoch: Optional[int] = None,
        create_data_embs: bool = False,
    ) -> None:
        """Upload image df and polygon df to Minio root bucket

        For SemSeg we only have one epoch, the final pass. So for now
        we hard code 0 in place of last_epoch.
        """
        proj_run = f"{config.current_project_id}/{config.current_run_id}"
        minio_file = f"{proj_run}/{split}/0/data/data.hdf5"
        cls._handle_numpy_types(df=in_frame)
        object_store.create_project_run_object_from_df(
            df=in_frame,
            object_name=minio_file,
            bucket_name=config.root_bucket_name,
        )

        dir_name = f"{split_loc}/0"
        out_frame = get_output_df(
            dir_name,
            prob_only=False,
            split=split,
            epoch_or_inf=0,
        )
        if "id" not in out_frame.get_column_names():
            out_frame["id"] = vaex.vrange(0, len(out_frame), dtype="int32")

        polygon_minio_file = f"{proj_run}/{split}/0/prob/prob.hdf5"
        cls._handle_numpy_types(df=out_frame)
        object_store.create_project_run_object_from_df(
            df=out_frame,
            object_name=polygon_minio_file,
            bucket_name=config.root_bucket_name,
        )
