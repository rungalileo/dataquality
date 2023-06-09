from __future__ import annotations

import glob
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import DataSet, MetasType
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)
from dataquality.schemas.split import Split
from dataquality.utils.upload import chunk_load_then_upload_df

# smaller than ITER_CHUNK_SIZE from base_data_logger because very large chunks
# containing image data often won't fit in memory

ITER_CHUNK_SIZE_IMAGES = 10000


class ImageClassificationDataLogger(TextClassificationDataLogger):
    __logger_name__ = "image_classification"
    logger_config: ImageClassificationLoggerConfig = image_classification_logger_config

    DATA_FOLDER_EXTENSION = {"emb": "hdf5", "prob": "hdf5", "data": "arrow"}

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
            texts=texts,
            labels=labels,
            ids=ids,
            split=split,
            meta=meta,
            inference_name=inference_name,
        )

    @property
    def support_data_embs(self) -> bool:
        """Coming soon via CLIP"""
        return False

    def log_image_dataset(
        self,
        dataset: Union[DataSet, "ImageFolder"],  # type: ignore # noqa: F821
        *,
        imgs_colname: Optional[str] = None,
        imgs_location_colname: Optional[str] = None,
        imgs_remote_location: Optional[str] = None,
        batch_size: int = ITER_CHUNK_SIZE_IMAGES,
        id: str = "id",
        label: Union[str, int] = "label",
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Union[List[str], List[int], None] = None,
        column_map: Optional[Dict[str, str]] = None,
        parallel: bool = False,
    ) -> Any:
        """
        For docstring see top level method located in core/log.py
        """
        if type(dataset).__name__ == "ImageFolder":
            dataset = self._prepare_df_from_ImageFolder(dataset, imgs_remote_location)
            imgs_location_colname = "text"

        if imgs_colname is None and imgs_location_colname is None:
            raise GalileoException(
                "Must provide one of imgs_colname or imgs_location_colname."
            )
        column_map = column_map or {id: "id"}
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.rename(columns=column_map)
            if self._dataset_requires_upload_prep(
                dataset=dataset, imgs_location_colname=imgs_location_colname
            ):
                dataset = self._prepare_content(
                    dataset=dataset,
                    imgs_location_colname=imgs_location_colname,
                    parallel=parallel,
                )
        elif self.is_hf_dataset(dataset):
            dataset = self._prepare_hf(
                dataset,
                imgs_colname=imgs_colname,
                imgs_location_colname=imgs_location_colname,
                id_=id,
            )
        else:
            raise GalileoException(
                f"Dataset must be one of pandas or HF, but got {type(dataset)}"
            )
        self.log_dataset(
            dataset=dataset,
            batch_size=batch_size,
            text="text",
            id=id,
            label=label,
            split=split,
            inference_name=inference_name,
            meta=meta,
        )

    def _dataset_requires_upload_prep(
        self, dataset: pd.DataFrame, imgs_location_colname: Optional[str] = None
    ) -> bool:
        if imgs_location_colname is None:
            raise GalileoException(
                "Must provide imgs_location_colname in order to upload content."
            )
        return os.path.isfile(dataset[imgs_location_colname].iloc[0])

    def _prepare_content(
        self,
        dataset: pd.DataFrame,
        imgs_location_colname: Optional[str],
        parallel: bool = False,
    ) -> pd.DataFrame:
        file_list = dataset[imgs_location_colname].tolist()
        project_id = config.current_project_id

        with tempfile.TemporaryDirectory() as temp_dir:
            export_format = "arrow"
            chunk_load_then_upload_df(
                file_list=file_list,
                export_cols=["data", "object_path"],
                project_id=project_id,
                temp_dir=temp_dir,
                parallel=parallel,
                export_format=export_format,
                use_data_md5_hash=True,
                object_path=str(project_id),
            )
            df = vaex.open(f"{temp_dir}/*.arrow")
        df = df.to_pandas_df()
        # df has columns "file_path", "object_path" we merge with original dataset
        # on imgs_location_colname and rename "object_path" to "text"
        dataset = dataset.merge(df, left_on=imgs_location_colname, right_on="file_path")
        dataset["text"] = dataset["object_path"]
        for f in glob.glob(f"{temp_dir}/*.arrow"):
            os.remove(f)

        return dataset

    def convert_large_string(self, df: DataFrame) -> DataFrame:
        """We override to avoid doing the computation to check if the text is over 2GB

        Because this is CV, almost certainly the text will be over the limit, and for
        really big ones, the computation gets very long (and seems to actually use
        some memory). We just assume it's over the limit (which is safe) and export
        """
        df["text"] = df['astype(text, "large_string")']
        return df

    def _prepare_hf(
        self,
        dataset: DataSet,
        imgs_colname: Optional[str],
        imgs_location_colname: Optional[str],
        id_: str,
        # returns HF dataset, hard to express in mypy without
        # importing the datasets package
    ) -> Any:
        import datasets

        assert isinstance(dataset, datasets.Dataset)

        # Find the id column, or create it.
        if id_ not in dataset.column_names:
            dataset = dataset.add_column(name=id_, column=list(range(len(dataset))))

        if imgs_colname is not None:
            # HF datasets Image feature
            from dataquality.utils.hf_images import process_hf_image_feature_for_logging

            prepared = process_hf_image_feature_for_logging(dataset, imgs_colname)
        elif imgs_location_colname is not None:
            # file paths
            from dataquality.utils.hf_images import process_hf_image_paths_for_logging

            prepared = process_hf_image_paths_for_logging(
                dataset, imgs_location_colname
            )
        else:
            raise GalileoException(
                "Must provide one of imgs_colname or imgs_location_colname."
            )

        return prepared

    def _prepare_df_from_ImageFolder(
        self,
        dataset: "ImageFolder",  # type: ignore # noqa: F821
        imgs_remote_location: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create a dataframe containing the ids, labels and paths of the images
        coming from an ImageFolder dataset.
        """
        df = pd.DataFrame(columns=["text", "label"], data=dataset.imgs)
        label_idx_to_label = {
            label_idx: label for label, label_idx in dataset.class_to_idx.items()
        }
        df["label"] = df.label.map(label_idx_to_label)
        df = df.reset_index().rename(columns={"index": "id"})

        # Replace the paths with the remote one, if a remote location is specified
        if imgs_remote_location is not None:
            df["text"] = df["text"].str.replace(dataset.root, imgs_remote_location)

        return df
