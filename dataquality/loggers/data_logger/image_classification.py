from __future__ import annotations

import glob
import os
import tempfile
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
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
from dataquality.schemas.dataframe import BaseLoggerDataFrames
from dataquality.schemas.split import Split
from dataquality.utils.cv_smart_features import generate_smart_features
from dataquality.utils.upload import chunk_load_then_upload_df

# smaller than ITER_CHUNK_SIZE from base_data_logger because very large chunks
# containing image data often won't fit in memory
from dataquality.utils.vaex import validate_unique_ids

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
        self.imgs_local_colname: Optional[str] = None
        self.imgs_remote_colname: Optional[str] = None

    @property
    def support_data_embs(self) -> bool:
        """Coming soon via CLIP"""
        return False

    def log_image_dataset(
        self,
        dataset: Union[DataSet, "ImageFolder"],  # type: ignore # noqa: F821
        *,
        imgs_local: Optional[str] = None,
        imgs_remote: Optional[str] = None,
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
        For main docstring see top level method located in core/log.py.

        In addition we set the variables self.imgs_local_colname and
        self.imgs_remote_location.
        """
        if type(dataset).__name__ == "ImageFolder":
            # TODO: add support for also providing a df with metadata
            # imgs_local ignored as it is not necessary
            dataset = self._prepare_df_from_ImageFolder(
                dataset=dataset, imgs_remote_location=imgs_remote, split=split
            )
            column_map = None  # make sure we don't accidentally use it (no df provided)
        else:
            if imgs_local is None and imgs_remote is None:
                raise GalileoException(
                    "Must provide either imgs_local or imgs_remote when using a df"
                )
            self.imgs_local_colname = imgs_local
            self.imgs_remote_colname = imgs_remote

        # Get the column mapping and rename imgs_local and imgs_remote if required
        # Only add id -> "id" when !="id", since we accept dfs without the id column
        column_map = column_map or ({id: "id"} if id != "id" else {})
        imgs_local = (
            None if imgs_local is None else column_map.get(imgs_local, imgs_local)
        )
        imgs_remote = (
            None if imgs_remote is None else column_map.get(imgs_remote, imgs_remote)
        )

        # If no remote paths are found, upload to the local images to the objectstore
        if isinstance(dataset, pd.DataFrame):
            dataset = self._prepare_content(
                dataset=dataset, column_map=column_map, parallel=parallel
            )
        elif self.is_hf_dataset(dataset):
            dataset = self._prepare_hf(dataset, id_=id, column_map=column_map)
        else:
            raise GalileoException(
                f"Dataset must be one of pandas or HF, but got {type(dataset)}"
            )

        # Log the local images paths as non-metadata if they are provided + they are not
        # already logged as metadata
        non_meta = []
        if (self.imgs_local_colname is not None) and (
            meta is None or self.imgs_local_colname not in meta
        ):
            non_meta.append(self.imgs_local_colname)

        self.log_dataset(
            dataset=dataset,
            batch_size=batch_size,
            text=self.imgs_remote_colname or "text",  # colname for remote images
            id=id,
            label=label,
            split=split,
            inference_name=inference_name,
            meta=meta,
            non_meta=non_meta,
        )

    def _has_remote_images(self, dataset: DataSet) -> bool:
        """Check if the dataset contains a column containing remote images"""
        # TODO: should we check that we can reach one of the images ?
        if isinstance(dataset, pd.DataFrame):
            columns = dataset.columns
        elif self.is_hf_dataset(dataset):
            columns = dataset.column_names  # type: ignore # noqa: F821

        return (
            self.imgs_remote_colname is not None and self.imgs_remote_colname in columns
        )

    def _prepare_content(
        self,
        dataset: pd.DataFrame,
        column_map: dict,
        parallel: bool = False,
    ) -> pd.DataFrame:
        """
        If remote paths already exist in the df, do nothing.

        If not, upload the images to the objectstore and add their paths in the df in
        the column self.imgs_remote_colname := "gal_remote_images_paths".
        """
        # Rename text -> text_original if "text" exists (as its used internally)
        column_map.update(
            {"text": "text_original"} if "text" in dataset.columns else {}
        )
        dataset = dataset.rename(columns=column_map)

        # No need to upload data if we already have access to remote images
        if self._has_remote_images(dataset):
            return dataset
        self.imgs_remote_colname = "gal_remote_images_paths"

        file_list = dataset[self.imgs_local_colname].tolist()
        project_id = config.current_project_id

        with tempfile.TemporaryDirectory() as temp_dir:
            export_format = "arrow"
            chunk_load_then_upload_df(
                file_list=file_list,
                export_cols=["data", "object_path"],
                project_id=project_id,
                temp_dir=temp_dir,
                bucket=config.images_bucket_name,
                parallel=parallel,
                export_format=export_format,
                use_data_md5_hash=True,
                object_path=str(project_id),
            )
            df = vaex.open(f"{temp_dir}/*.arrow")
        df = df.to_pandas_df()
        # df has columns "file_path", "object_path" we merge with original dataset
        # on imgs_location_colname and rename "object_path" to self.imgs_remote_colname
        dataset = dataset.merge(
            df, left_on=self.imgs_local_colname, right_on="file_path"
        )
        dataset.rename(columns={"object_path": self.imgs_remote_colname}, inplace=True)
        dataset.drop(columns=["file_path"], inplace=True)
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
        id_: str,
        column_map: dict,
        # returns HF dataset, hard to express in mypy without
        # importing the datasets package
    ) -> Any:
        import datasets

        assert isinstance(dataset, datasets.Dataset)

        # Rename text -> text_original if "text" exists (as its used internally)
        column_map.update(
            {"text": "text_original"} if "text" in dataset.column_names else {}
        )
        for old_col, new_col in column_map.items():
            dataset = dataset.rename_column(old_col, new_col)

        # Find the id column, or create it.
        if id_ not in dataset.column_names:
            dataset = dataset.add_column(name=id_, column=list(range(len(dataset))))

        # No need to upload data if we already have access to remote images
        if self._has_remote_images(dataset):
            return dataset
        self.imgs_remote_colname = "gal_remote_images_paths"
        assert (
            self.imgs_local_colname is not None
        ), "imgs_local has to be specified since imgs_remote is not in the df"

        if dataset.features[self.imgs_local_colname].dtype == "string":
            # Case where the column contains paths to the images
            from dataquality.utils.hf_images import process_hf_image_paths_for_logging

            prepared = process_hf_image_paths_for_logging(
                dataset, self.imgs_local_colname, self.imgs_remote_colname
            )
        elif isinstance(dataset.features[self.imgs_local_colname], datasets.Image):
            # Case where the column contains Image feature
            # We will not have local paths in this case
            from dataquality.utils.hf_images import process_hf_image_feature_for_logging

            prepared = process_hf_image_feature_for_logging(
                dataset, self.imgs_local_colname, self.imgs_remote_colname
            )
            # We don't want to log the local_col since it does not contain local paths
            self.imgs_local_colname = None
        else:
            raise GalileoException(
                f"The argument imgs_local={repr(self.imgs_local_colname)} doesn't point"
                "to a column containing local paths or images. Pass a valid column name"
            )

        return prepared

    def _prepare_df_from_ImageFolder(
        self,
        dataset: "ImageFolder",  # type: ignore # noqa: F821
        imgs_remote_location: Optional[str] = None,
        split: Optional[Split] = None,
    ) -> pd.DataFrame:
        """
        Create a dataframe containing the ids, labels and paths of the images
        coming from an ImageFolder dataset.
        """
        # Extract the local paths from the ImageFolder dataset and add them to the df
        self.imgs_local_colname = "gal_local_images_paths"

        if split == Split.inference:
            df = pd.DataFrame(
                columns=[self.imgs_local_colname], data=[img[0] for img in dataset.imgs]
            )
        else:
            df = pd.DataFrame(
                columns=[self.imgs_local_colname, "label"], data=dataset.imgs
            )
            label_idx_to_label = {
                label_idx: label for label, label_idx in dataset.class_to_idx.items()
            }
            df["label"] = df.label.map(label_idx_to_label)

        df = df.reset_index().rename(columns={"index": "id"})

        # Also add remote paths, if a remote location is specified
        if imgs_remote_location is not None:
            self.imgs_remote_colname = "text"  # getting renamed to "text" later anyways
            df[self.imgs_remote_colname] = df[self.imgs_local_colname].str.replace(
                dataset.root, imgs_remote_location
            )

        return df

    @classmethod
    def process_in_out_frames(
        cls,
        in_frame: DataFrame,
        out_frame: DataFrame,
        prob_only: bool,
        epoch_or_inf_name: str,
        split: str,
    ) -> BaseLoggerDataFrames:
        """We have to be careful with joins in the image datasets because of the string
        encoded images. They are too long and cause arrow offsets

        Override base to handle very large strings (the encoded images)

        There are a number of bugs (and open PRs) around this issue. PyArrow as a
        fundamental issue around strings over 2GB in size. They have a special datatype
        `large_string` for them, but that type is not robust.
        See https://issues.apache.org/jira/browse/ARROW-9773
        and https://issues.apache.org/jira/browse/ARROW-17828

        One such issue is the use of `.take` with arrays of large_strings. .take is
        both _not_ memory safe, and causes an ArrayOffSetOverFlow error
        (`pyarrow.lib.ArrowInvalid: offset overflow while concatenating arrays`)
        See https://github.com/vaexio/vaex/issues/2335
        and https://github.com/huggingface/datasets/issues/615

        The solution is to use `.slice` instead of `.take` - this creates a zero-memory
        copy, and does not cause the overflow.
        See https://github.com/huggingface/datasets/pull/645

        The issue is that vaex currently uses `.take` (because this didn't used to be
        an issue) when performing `join` operations. Because the join in vaex is lazy,
        the issue doesn't materialize until exporting. The true solution is for vaex to
        stop using take (I made a pr: https://github.com/vaexio/vaex/pull/2336)

        So we are careful to only join on the columns we need
        emb: "id", "emb"
        prob: "id", "gold", "prob"
        data: "id", "pred" + all the other cols not in emb or prob
        """
        validate_unique_ids(out_frame, epoch_or_inf_name)
        allow_missing_in_df_ids = cls.logger_config.dataloader_random_sampling
        filter_ids: Set[int] = set()
        if allow_missing_in_df_ids:
            observed_ids = image_classification_logger_config.observed_ids
            keys = [k for k in observed_ids.keys() if split in k]
            if len(keys):
                filter_ids = set(observed_ids[keys[0]])
            for k in keys:
                filter_ids = filter_ids.intersection(observed_ids[k])

        emb_cols = ["id"] if prob_only else ["id", "emb"]
        emb_df = out_frame[emb_cols]
        if allow_missing_in_df_ids:
            filter_ids_arr: np.ndarray = np.array(list(filter_ids))
            del filter_ids
            in_frame = in_frame[in_frame["id"].isin(filter_ids_arr)]
            out_frame = out_frame[out_frame["id"].isin(filter_ids_arr)]

        # The in_frame has gold, so we join with the out_frame to get the probabilities
        prob_df = out_frame.join(in_frame[["id", "gold"]], on="id")[
            cls._get_prob_cols()
        ]

        if prob_only:
            emb_df = out_frame[["id"]]
            data_df = out_frame[["id"]]
        else:
            emb_df = out_frame[["id", "emb"]]
            remove_cols = emb_df.get_column_names() + prob_df.get_column_names()

            # The data df needs pred, which is in the prob_df, so we join just on that
            # col
            # TODO: We should update runner processing so it can grab the pred from the
            #  prob_df on the server. This is confusing code
            data_cols = in_frame.get_column_names() + ["pred"]
            data_cols = ["id"] + [c for c in data_cols if c not in remove_cols]

            data_df = in_frame.join(out_frame[["id", "pred"]], on="id")[data_cols]

        dataframes = BaseLoggerDataFrames(prob=prob_df, emb=emb_df, data=data_df)

        # These df vars will be used in upload_in_out_frames
        dataframes.emb.set_variable("skip_upload", prob_only)
        dataframes.data.set_variable("skip_upload", prob_only)
        epoch_inf_val = out_frame[[epoch_or_inf_name]][0][0]
        dataframes.prob.set_variable("progress_name", str(epoch_inf_val))

        return dataframes

    def add_cv_smart_features(self, in_frame_split: DataFrame) -> DataFrame:
        """
        Add smart features on images (blurriness, contrast, etc) to the dataframe.
        Overwriting the base method in the base_data_logger.
        """
        in_frame_split = generate_smart_features(in_frame_split)

        return in_frame_split
