from __future__ import annotations

import glob
import os
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import vaex
from vaex.dataframe import DataFrame

from dataquality import config
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import DataSet, MetasType
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)
from dataquality.schemas.cv import GAL_LOCAL_IMAGES_PATHS
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

    @property
    def support_data_embs(self) -> bool:
        """Coming soon via CLIP"""
        return False

    def log_image_dataset(
        self,
        dataset: Union[DataSet, "ImageFolder"],  # type: ignore # noqa: F821
        *,
        imgs_local_colname: Optional[str] = None,
        imgs_remote: Optional[str] = None,
        batch_size: int = ITER_CHUNK_SIZE_IMAGES,
        id: str = "id",
        label: str = "label",
        split: Optional[Split] = None,
        inference_name: Optional[str] = None,
        meta: Optional[List[str]] = None,
        column_map: Optional[Dict[str, str]] = None,
        parallel: bool = False,
    ) -> Any:
        """For main docstring see top level method located in core/log.py."""
        if type(dataset).__name__ == "ImageFolder":
            # For ImageFolder we ignore imgs_local_colname since not dataframe was
            # passed in and we infer it from the ImageFolder
            dataset = self._prepare_df_from_ImageFolder(
                dataset=dataset, imgs_remote_location=imgs_remote, split=split
            )
            # In _prepare_df_from_ImageFolder, we set a column GAL_LOCAL_IMAGES_PATHS
            # that maps to the local image path
            imgs_local_colname = GAL_LOCAL_IMAGES_PATHS

        if imgs_local_colname is None and imgs_remote is None:
            raise GalileoException(
                "Must provide imgs_local_colname or imgs_remote when using a df"
            )
        elif imgs_local_colname is None:
            warnings.warn(
                "Smart Features won't be calculated since no local paths to images"
                "were provided"
            )
        elif imgs_remote is None:
            warnings.warn(
                "The images will be uploaded to a remote object store (can be"
                "slow). Provide remote paths to images to avoid this upload"
            )

        # Get the column mapping and rename imgs_local and imgs_remote if required
        column_map = column_map or {id: "id"}
        if imgs_local_colname is not None:
            column_map[imgs_local_colname] = GAL_LOCAL_IMAGES_PATHS
        # Rename the col with remote path to "text" (it would be renamed to "text" later
        # anyways since IC inherits the logging methods from TC which uses "text")
        if imgs_remote is not None:
            column_map[imgs_remote] = "text"

        dataset = self.apply_column_map(dataset, column_map)
        # If no remote paths are found, upload to the local images to the objectstore
        if isinstance(dataset, pd.DataFrame):
            dataset, has_local_paths = self._prepare_content(dataset, parallel)
        elif self.is_hf_dataset(dataset):
            dataset, has_local_paths = self._prepare_hf(dataset, id)
        else:
            raise GalileoException(
                f"Dataset must be one of pandas or HF, but got {type(dataset)}"
            )

        meta = meta or []
        if has_local_paths and GAL_LOCAL_IMAGES_PATHS not in meta:
            meta.append(GAL_LOCAL_IMAGES_PATHS)

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

    def _has_remote_images(self, dataset: DataSet) -> bool:
        """Check if the dataset contains a column containing remote images"""
        if isinstance(dataset, pd.DataFrame):
            columns = dataset.columns
        elif self.is_hf_dataset(dataset):
            columns = dataset.column_names  # type: ignore # noqa: F821

        return "text" in columns

    def _prepare_content(
        self,
        dataset: pd.DataFrame,
        parallel: bool = False,
    ) -> Tuple[pd.DataFrame, bool]:
        """
        Uploads local images to ObjectStore and adds remote paths to the df in a column
        called "text".

        NOTE: If the dataset already contains remote paths, this function does nothing
        """
        has_local_paths = GAL_LOCAL_IMAGES_PATHS in dataset.columns

        # No need to upload data if we already have access to remote images
        if self._has_remote_images(dataset):
            return dataset, has_local_paths

        # If it doesn't have remote images, it necessarily has local images
        file_list = dataset[GAL_LOCAL_IMAGES_PATHS].tolist()
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
        # on GAL_LOCAL_IMAGES_PATHS and rename "object_path" to imgs_remote_colname
        dataset = dataset.merge(
            df, left_on=GAL_LOCAL_IMAGES_PATHS, right_on="file_path"
        )
        dataset.rename(columns={"object_path": "text"}, inplace=True)
        dataset.drop(columns=["file_path"], inplace=True)
        for f in glob.glob(f"{temp_dir}/*.arrow"):
            os.remove(f)

        return dataset, has_local_paths

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
        # returns HF dataset, hard to express in mypy without
        # importing the datasets package
    ) -> Tuple[DataSet, bool]:
        """
        If remote paths already exist in the df, do nothing.

        If not, upload the images to the objectstore and add their paths in the df in
        the column imgs_remote_colname := GAL_REMOTE_IMAGES_PATHS
        """
        import datasets

        assert isinstance(dataset, datasets.Dataset)

        # Find the id column, or create it.
        if id_ not in dataset.column_names:
            dataset = dataset.add_column(name=id_, column=list(range(len(dataset))))

        # Check if the data in the local_col are string (paths) and not bytes
        has_local_paths = (GAL_LOCAL_IMAGES_PATHS in dataset.column_names) and (
            dataset.features[GAL_LOCAL_IMAGES_PATHS].dtype == "string"
        )

        # No need to upload data if we already have access to remote images
        if self._has_remote_images(dataset):
            return dataset, has_local_paths

        if dataset.features[GAL_LOCAL_IMAGES_PATHS].dtype == "string":
            # Case where the column contains paths to the images
            from dataquality.utils.hf_images import process_hf_image_paths_for_logging

            prepared = process_hf_image_paths_for_logging(
                dataset, GAL_LOCAL_IMAGES_PATHS
            )
        elif isinstance(dataset.features[GAL_LOCAL_IMAGES_PATHS], datasets.Image):
            # Case where the column contains Image feature
            # We will not have local paths in this case
            from dataquality.utils.hf_images import process_hf_image_feature_for_logging

            prepared = process_hf_image_feature_for_logging(
                dataset, GAL_LOCAL_IMAGES_PATHS
            )
        else:
            raise GalileoException(
                f"The argument imgs_local={GAL_LOCAL_IMAGES_PATHS} doesn't point"
                "to a column containing local paths or images. Pass a valid column name"
            )

        return prepared, has_local_paths

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
        if split == Split.inference:
            df = pd.DataFrame(
                columns=[GAL_LOCAL_IMAGES_PATHS], data=[img[0] for img in dataset.imgs]
            )
        else:
            df = pd.DataFrame(
                columns=[GAL_LOCAL_IMAGES_PATHS, "label"], data=dataset.imgs
            )
            label_idx_to_label = {
                label_idx: label for label, label_idx in dataset.class_to_idx.items()
            }
            df["label"] = df.label.map(label_idx_to_label)

        df = df.reset_index().rename(columns={"index": "id"})

        # Also add remote paths, if a remote location is specified
        if imgs_remote_location is not None:
            df["text"] = df[GAL_LOCAL_IMAGES_PATHS].str.replace(
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
            # We also don't want to upload the local image paths to ObjectStore
            # for processing
            remove_cols.append(GAL_LOCAL_IMAGES_PATHS)

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

    def upload_split_from_in_frame(
        self,
        object_store: ObjectStore,
        in_frame: DataFrame,
        split: str,
        split_loc: str,
        last_epoch: Optional[int],
        create_data_embs: bool,
        data_embs_col: str,
    ) -> None:
        in_frame = self.add_cv_smart_features(in_frame, split)
        super().upload_split_from_in_frame(
            object_store=object_store,
            in_frame=in_frame,
            split=split,
            split_loc=split_loc,
            last_epoch=last_epoch,
            create_data_embs=create_data_embs,
            data_embs_col=data_embs_col,
        )

    @classmethod
    def add_cv_smart_features(cls, in_frame: DataFrame, split: str) -> DataFrame:
        """
        Calculate and add smart features on images (blurriness, contrast, etc) to the
        dataframe.

        The in_frame df only requires the column containing the paths to local images
        GAL_LOCAL_IMAGES_PATHS for this method to run.
        """
        if GAL_LOCAL_IMAGES_PATHS not in in_frame.get_column_names():
            return in_frame

        print(
            f"🔲 Calculating Smart Features for split {split} (can take a few minutes "
            "depending on the size of your dataset)"
        )
        in_frame = generate_smart_features(in_frame)
        return in_frame
