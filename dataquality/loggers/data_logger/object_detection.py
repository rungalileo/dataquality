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
from dataquality.schemas.dataframe import BaseLoggerDataFrames, DFVar
from dataquality.schemas.split import Split
from dataquality.utils.vaex import add_pca_to_df


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

    def convert_large_string(self, df: DataFrame) -> DataFrame:
        """We skip this step because there is no 'text' field"""
        return df

    @classmethod
    def prob_only(
        cls,
        epochs: List[str],
        split: str,
        epoch_or_inf_name: Union[int, str],
        last_epoch: Optional[int],
    ) -> bool:
        """In OD, theres only 1 epoch, so we want to upload it all, not just probs"""
        return False

    @classmethod
    def create_and_upload_data_embs(
        cls, df: DataFrame, split: str, epoch_or_inf: str
    ) -> None:
        """Data embeddings not yet supported for any CV task"""

    @classmethod
    def process_in_out_frames(
        cls,
        in_frame: DataFrame,
        out_frame: DataFrame,
        prob_only: bool,
        epoch_or_inf_name: str,
        split: str,
    ) -> BaseLoggerDataFrames:
        """Process the logged input data and output data into uploadable files

        In OD (like NER), we have 2 different 'levels' of data: image and box. The
        in_frame is everything at the image level (image url, metadata etc). The
        out_frame is the embeddings, probabilities, bboxes, label, etc (see
        `dq.loggers.model_logger._get_data_dict` for more details).

        We want to upload the image data as the `data` field, the probabilities and
        boxes as the `probs` field, and embeddings as the `emb` field.

        It's possible that PCA/UMAP have already been applied, so we will check for
        those columns (via CUDA). If they have, we uplod the PCA embeddings as well
        as the x, y coordinates of UMAP.

        If not, we will preemptively apply PCA to the embeddings before uploading.
        This is because the embeddings in OD are very large (> 2000dim), and it's not
        scalable. Additionally, we run our algorithms based off of the PCA embeddings
        in the server. So we will always drop the raw embeddings for PCA.
        """
        out_frame["id"] = vaex.vrange(0, len(out_frame), dtype="int32")
        out_cols = out_frame.get_column_names()

        emb_cols = ["id", "emb_pca"]
        if "emb_pca" not in out_cols:
            out_frame = add_pca_to_df(out_frame, chunk_size=25_000)
        if "x" in out_cols and "y" in out_cols:
            emb_cols.extend(["x", "y"])
        emb_df = out_frame[emb_cols]
        prob_cols = [i for i in out_cols if i not in emb_cols and i != "emb"]
        prob_df = out_frame[prob_cols]
        # Epoch is always 0 because we only do 1 pass over the data
        # see `dq.loggers.data_logger.base_data_logger.upload_in_out_frames`
        prob_df.set_variable(DFVar.progress_name, "0")

        return BaseLoggerDataFrames(
            data=in_frame,
            emb=emb_df,
            prob=prob_df,
        )
