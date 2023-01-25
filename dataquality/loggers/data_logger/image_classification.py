import os
from typing import List, Optional, Union

import pandas as pd
from PIL.Image import Image

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    DataSet,
    MetasType,
)
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)
from dataquality.schemas.split import Split
from dataquality.utils.cv import (
    _bytes_to_b64_str,
    _img_path_to_b64_str,
    _img_to_b64_str,
)

# smaller than ITER_CHUNK_SIZE from base_data_logger because very large chunks
# containing image data often won't fit in memory
ITER_CHUNK_SIZE_IMAGES = 10000


class ImageClassificationDataLogger(TextClassificationDataLogger):
    __logger_name__ = "image_classification"
    logger_config: ImageClassificationLoggerConfig = image_classification_logger_config

    def __init__(
        self,
        texts: List[str] = None,
        labels: List[str] = None,
        ids: List[int] = None,
        split: str = None,
        meta: MetasType = None,
        inference_name: str = None,
    ) -> None:
        super().__init__(
            texts=texts,
            labels=labels,
            ids=ids,
            split=split,
            meta=meta,
            inference_name=inference_name,
        )

    def _prepare_pandas(self,
                        dataset: pd.DataFrame,
                        imgs_location_colname: Optional[str],
                        imgs_colname: Optional[str],
                        imgs_dir: Optional[str],
                        ) -> pd.DataFrame:
        imgs_dir = imgs_dir or ""
        imgs_dir: str

        if imgs_location_colname is not None:
            # image paths
            dataset["text"] = dataset[imgs_location_colname].apply(
                lambda x: _img_path_to_b64_str(img_path=os.path.join(imgs_dir, x))
            )
        else:
            # PIL images in a DataFrame column - weird, but we'll allow it
            example = dataset[imgs_location_colname].values[0]
            if not isinstance(example, Image):
                raise GalileoException(
                    f"Got imgs_colname={repr(imgs_colname)}, but that "
                    "dataset column does not contain images. If you have "
                    "image paths, pass imgs_location_colname instead."
                )

            dataset["text"] = dataset[imgs_colname].apply(_img_to_b64_str)

        return dataset

    def _prepare_hf(self,
                        dataset: DataSet,
                        imgs_colname: Optional[str],
                        imgs_location_colname: Optional[str],
                        id_: str,
                        ) -> DataSet:
        import datasets
        dataset: datasets.Dataset

        # Find the id column, or create it.
        if id_ not in dataset.column_names:
            dataset = dataset.add_column(
                name=id_, column=list(range(len(dataset)))
            )

        if imgs_colname is not None:
            # HF datasets Image feature
            import datasets

            dataset = dataset.cast_column(
                imgs_colname, datasets.Image(decode=False)
            )

            def hf_map_image_feature(example):
                image = example[imgs_colname]

                if image.dtype != "PIL.Image.Image":
                    raise GalileoException(
                        f"Got imgs_colname={repr(imgs_colname)}, but that "
                        "dataset feature does not contain images. If you have "
                        "image paths, pass imgs_location_colname instead."
                    )

                if image["bytes"] is None:
                    # sometimes the Image feature only contains a path
                    # example: beans dataset
                    example["text"] = _img_path_to_b64_str(
                        # assume abs paths for HF
                        img_path=image["path"],
                    )
                else:
                    example["text"] = _bytes_to_b64_str(
                        # assume abs paths for HF
                        img_bytes=image["bytes"],
                        img_path=image["path"],
                    )
                return example

            dataset = dataset.map(hf_map_image_feature)
        else:
            # file paths
            def hf_map_file_path(example):
                example["text"] = _img_path_to_b64_str(
                    # assume abs paths for HF
                    example[imgs_location_colname]
                )
                return example

            dataset = dataset.map(hf_map_file_path)
        return dataset

    def log_image_dataset(
        self,
        dataset: DataSet,
        *,
        imgs_colname: Optional[str] = None,
        imgs_location_colname: Optional[str] = None,
        imgs_dir: Optional[str] = None,
        batch_size: int = ITER_CHUNK_SIZE_IMAGES,
        id: Union[str, int] = "id",
        label: Union[str, int] = "label",
        split: Optional[Split] = None,
        meta: Optional[List[Union[str, int]]] = None,
    ) -> None:
        if imgs_colname is None and imgs_location_colname is None:
            raise GalileoException(
                "Must provide one of imgs_colname or imgs_location_colname."
            )
        if self.is_hf_dataset(dataset):
            dataset = self._prepare_hf(dataset,
                                       imgs_colname=imgs_colname,
                                       imgs_location_colname=imgs_location_colname,
                                       id_=id)
        elif isinstance(dataset, pd.DataFrame):
            dataset = self._prepare_pandas(dataset,
                                           imgs_colname=imgs_colname,
                                           imgs_location_colname=imgs_location_colname,
                                           imgs_dir=imgs_dir)
        else:
            raise GalileoException(
                f"Dataset must be one of pandas or HF, " f"but got {type(dataset)}"
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
