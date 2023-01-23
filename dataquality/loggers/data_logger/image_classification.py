import os
from enum import Enum, unique
from typing import List, Optional, Union

import pandas as pd
from PIL.Image import Image

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import (
    ITER_CHUNK_SIZE,
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


@unique
class ImageFieldType(str, Enum):
    file_path = "file_path"
    pil_image = "pil_image"
    hf_image_feature = "hf_image_feature"
    unknown = "unknown"


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
                        imgs_location_colname: str,
                        imgs_dir: str,
                        ) -> pd.DataFrame:
        example = dataset[imgs_location_colname].values[0]

        image_field_type = ImageFieldType.unknown
        if isinstance(example, Image):
            image_field_type = image_field_type.pil_image
        elif isinstance(example, str):
            image_field_type = image_field_type.file_path
        elif isinstance(example, dict) and set(example.keys()) == {'bytes', 'path'}:
            image_field_type = image_field_type.hf_image_feature

        if image_field_type == image_field_type.file_path:
            dataset["text"] = dataset[imgs_location_colname].apply(
                lambda x: _img_path_to_b64_str(img_path=os.path.join(imgs_dir, x))
            )
        elif image_field_type == image_field_type.pil_image:
            dataset["text"] = dataset[imgs_location_colname].apply(_img_to_b64_str)
        elif image_field_type == image_field_type.hf_image_feature:
            # TODO
        else:
            raise GalileoException(
                f"Could not interpret column {imgs_location_colname} as either images"
                "or image paths."
            )
        return dataset

    def _prepare_hf(self,
                        dataset: DataSet,
                        imgs_location_colname: str,
                        id: str,
                        ) -> DataSet:
        import datasets
        dataset: datasets.Dataset

        # Find the id column, or create it.
        if id not in dataset.column_names:
            dataset = dataset.add_column(
                name=id, column=list(range(len(dataset)))
            )

        image_feature = dataset.features[imgs_location_colname]

        image_field_type = ImageFieldType.unknown
        if image_feature.dtype == "PIL.Image.Image":
            image_field_type = image_field_type.hf_image_feature
        elif image_feature.dtype == "string":
            image_field_type = image_field_type.file_path

        if image_field_type == image_field_type.hf_image_feature:
            import datasets

            dataset = dataset.cast_column(
                imgs_location_colname, datasets.Image(decode=False)
            )

            def hf_map_image_feature(example):
                image = example[imgs_location_colname]

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
        elif image_field_type == image_field_type.file_path:

            def hf_map_file_path(example):
                example["text"] = _img_path_to_b64_str(
                    # assume abs paths for HF
                    example[imgs_location_colname]
                )
                return example

            dataset = dataset.map(hf_map_file_path)
        else:
            raise GalileoException(
                f"Could not interpret column {imgs_location_colname} as either"
                " images or image paths."
            )
        return dataset

    def log_image_dataset(
        self,
        dataset: DataSet,
        imgs_dir: str,
        *,
        imgs_location_colname: Optional[str] = "relpath",
        batch_size: int = ITER_CHUNK_SIZE,
        id: Union[str, int] = "id",
        label: Union[str, int] = "label",
        split: Optional[Split] = None,
        meta: Optional[List[Union[str, int]]] = None,
    ) -> None:
        if self.is_hf_dataset(dataset):
            dataset = self._prepare_hf(dataset,
                                       imgs_location_colname=imgs_location_colname,
                                       id=id)
        elif isinstance(dataset, pd.DataFrame):
            dataset = self._prepare_pandas(dataset,
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
