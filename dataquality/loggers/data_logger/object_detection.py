from collections import defaultdict
from enum import Enum, unique
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
import vaex
from pandas import DataFrame

from dataquality.exceptions import GalileoException
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
from dataquality.utils.vaex import add_pca_to_df, rename_df


@unique
class GalileoDataLoggerAttributes(str, Enum):
    image = "image"
    ids = "ids"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging

    @staticmethod
    def get_valid() -> List[str]:
        return list(map(lambda x: x.value, GalileoDataLoggerAttributes))


class ODCols(str, Enum):
    image = "image"
    bbox = "bbox"
    gold_cls = "gold_cls"
    id = "id"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"  # Metadata columns for logging
    width = "width"
    height = "height"


class ObjectDetectionDataLogger(BaseGalileoDataLogger):
    """
    Class for logging input data/data of Object Detection models to Galileo.
    """

    __logger_name__ = "object_detection"
    logger_config: ObjectDetectionLoggerConfig = object_detection_logger_config

    def __init__(
        self,
        images: Optional[List[str]] = None,
        ids: Optional[List[int]] = None,
        split: Optional[str] = None,
        meta: Optional[MetasType] = None,
        inference_name: Optional[str] = None,
        width: Optional[List[float]] = None,
        height: Optional[List[float]] = None,
    ) -> None:
        super().__init__(meta)
        self.images = images if images is not None else []
        self.ids = ids if ids is not None else []
        self.width = width if width is not None else []
        self.height = height if height is not None else []
        self.split = split
        self.inference_name = inference_name

    @staticmethod
    def get_valid_attributes() -> List[str]:
        """
        Returns a list of valid attributes that GalileoModelConfig accepts
        :return: List[str]
        """
        return GalileoDataLoggerAttributes.get_valid()

    def _get_input_df(self) -> DataFrame:
        df_len = len(self.ids)
        inp = dict(
            id=self.ids,
            image=self.images,
            split=[Split(self.split).value] * df_len,
            data_schema_version=[__data_schema_version__] * df_len,
            width=self.width,
            height=self.height,
            **self.meta,
        )
        if self.split == Split.inference:
            inp["inference_name"] = [self.inference_name] * df_len
        return vaex.from_pandas(pd.DataFrame(inp))

    def log_dataset(
        self,
        dataset: DataSet,
        *,
        batch_size: int = ITER_CHUNK_SIZE,
        image: Union[str, int] = ODCols.image.value,
        id: Union[str, int] = ODCols.id.value,
        width: Union[str, int] = ODCols.width.value,
        height: Union[str, int] = ODCols.height.value,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Union[List[str], List[int], None] = None,
        **kwargs: Any,
    ) -> None:
        """Log a dataset of input samples for OD"""
        self.validate_kwargs(kwargs)
        self.split = split
        self.inference_name = inference_name
        column_map = {
            id: ODCols.id,
            image: ODCols.image,
            width: ODCols.width,
            height: ODCols.height,
        }
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.rename(columns=column_map)
            self._log_df(dataset, meta)
        elif isinstance(dataset, DataFrame):
            for chunk in range(0, len(dataset), batch_size):
                chunk_df = dataset[chunk : chunk + batch_size]
                chunk_df = rename_df(chunk_df, column_map)
                self._log_df(chunk_df, meta)
        elif isinstance(dataset, Iterable):
            self._log_iterator(
                dataset,
                batch_size,
                image,
                id,
                width,
                height,
                meta,
                split,
                inference_name,
            )
        else:
            raise GalileoException(
                f"Dataset must be one of pandas, vaex, HF, or Iterable, "
                f"but got {type(dataset)}"
            )

    def _log_iterator(
        self,
        dataset: Iterable,
        batch_size: int,
        image: Union[str, int],
        id: Union[str, int],
        width: Union[str, int],
        height: Union[str, int],
        meta: Union[List[str], List[int], None] = None,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        batches = defaultdict(list)
        metas = defaultdict(list)
        for chunk in dataset:
            batches[ODCols.image].append(chunk[image])
            batches[ODCols.id].append(chunk[id])
            batches[ODCols.width].append(chunk[width])
            batches[ODCols.height].append(chunk[height])

            for meta_col in meta or []:
                metas[meta_col].append(self._convert_tensor_to_py(chunk[meta_col]))

            if len(batches[ODCols.image]) >= batch_size:
                self._log_dict(batches, metas, split)
                batches.clear()
                metas.clear()

        # in case there are any left
        if batches:
            self._log_dict(batches, metas, split, inference_name)

    def _log_dict(
        self,
        d: Dict,
        meta: Dict,
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
    ) -> None:
        self.log_image_samples(
            images=d[ODCols.image],
            ids=d[ODCols.id],
            width=d[ODCols.width],
            height=d[ODCols.height],
            split=split,
            meta=meta,
            inference_name=inference_name,
        )

    def _log_df(
        self,
        df: Union[pd.DataFrame, DataFrame],
        meta: Union[List[str], List[int], None] = None,
    ) -> None:
        """Helper to log a pandas or vaex df"""
        self.images = df[ODCols.image].tolist()
        self.ids = df[ODCols.id].tolist()
        self.width = df[ODCols.width].tolist()
        self.height = df[ODCols.height].tolist()
        for meta_col in meta or []:
            self.meta[str(meta_col)] = df[meta_col].tolist()
        self.log()

    def log_image_samples(
        self,
        *,
        images: List[str],
        ids: List[int],
        width: List[float],
        height: List[float],
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[MetasType] = None,
        **kwargs: Any,  # For typing
    ) -> None:
        """Log input samples for OD"""
        self.validate_kwargs(kwargs)
        self.images = images
        self.ids = ids
        self.split = split
        self.inference_name = inference_name
        self.meta = meta or {}
        self.width = width
        self.height = height
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
        prob_cols = [i for i in out_cols if i not in emb_cols and i != "emb"] + ["id"]
        prob_df = out_frame[prob_cols]
        # Epoch is always 0 because we only do 1 pass over the data
        # see `dq.loggers.data_logger.base_data_logger.upload_in_out_frames`
        prob_df.set_variable(DFVar.progress_name, "0")

        return BaseLoggerDataFrames(
            data=in_frame,
            emb=emb_df,
            prob=prob_df,
        )
