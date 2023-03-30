from typing import Any, List, Optional, Union

import pandas as pd
import vaex
from pandas import DataFrame

from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
    BaseGalileoDataLogger,
    DataSet,
    MetasType,
)
from dataquality.loggers.logger_config.object_detection import (
    ObjectDetectionLoggerConfig,
    object_detection_logger_config,
)
from dataquality.schemas import __data_schema_version__
from dataquality.schemas.split import Split


class ObjectDetectionDataLogger(BaseGalileoDataLogger):
    __logger_name__ = "object_detection"
    logger_config: ObjectDetectionLoggerConfig = object_detection_logger_config
    ids: List
    file_names: List
    bbox: List
    cls: List

    def __init__(
        self,
        texts: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        ids: Optional[List[int]] = None,
        split: Optional[str] = None,
        meta: Optional[MetasType] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            meta=meta,
        )
        self.ids = []
        self.file_names = []
        self.bbox = []
        self.cls = []

    def _get_input_df(self) -> DataFrame:
        df_len = len(self.ids)
        inp = dict(
            id=self.ids,
            file_names=self.file_names,
            # bbox=self.bbox,
            # cls=self.cls,
            split=[Split(self.split).value] * df_len,
            data_schema_version=[__data_schema_version__] * df_len,
            **self.meta,
        )
        print(pd.DataFrame(inp))
        return vaex.from_pandas(pd.DataFrame(inp))

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
        self.split = split

        for img in dataset:
            self.ids.append(img["id"])
            self.file_names.append(img["file_name"])
            self.bbox.append(img["bbox"])
            self.cls.append(img["cls"])
        self.log()

    # @classmethod
    # def process_in_out_frames(
    #     cls,
    #     in_frame: DataFrame,
    #     out_frame: DataFrame,
    #     prob_only: bool,
    #     epoch_or_inf_name: str,
    #     split: str,
    # ) -> BaseLoggerDataFrames:
    #     pass
