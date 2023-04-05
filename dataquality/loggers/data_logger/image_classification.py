from typing import Any, Dict, List, Optional, Union

import pandas as pd
from vaex.dataframe import DataFrame

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
        dataset: DataSet,
        *,
        imgs_colname: Optional[str] = None,
        imgs_location_colname: Optional[str] = None,
        batch_size: int = ITER_CHUNK_SIZE_IMAGES,
        id: str = "id",
        label: Union[str, int] = "label",
        split: Optional[Split] = None,
        meta: Optional[List[Union[str, int]]] = None,
        column_map: Optional[Dict[str, str]] = None,
    ) -> Any:
        if imgs_colname is None and imgs_location_colname is None:
            raise GalileoException(
                "Must provide one of imgs_colname or imgs_location_colname."
            )
        column_map = column_map or {id: "id"}
        if isinstance(dataset, pd.DataFrame):
            dataset = dataset.rename(columns=column_map)
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
            meta=meta,
        )

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

        emb_cols = ["id"] if prob_only else ["id", "emb"]
        emb_df = out_frame[emb_cols]
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
