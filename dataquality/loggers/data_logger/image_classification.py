import os
from typing import List, Optional, Union, Any
from enum import Enum, unique

import pandas as pd
from PIL import Image

from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.base_data_logger import ITER_CHUNK_SIZE, MetasType, DataSet
from dataquality.loggers.data_logger.text_classification import (
    TextClassificationDataLogger,
)
from dataquality.loggers.logger_config.image_classification import (
    ImageClassificationLoggerConfig,
    image_classification_logger_config,
)
from dataquality.schemas.split import Split
from dataquality.utils.cv import _img_path_to_b64_str, _img_to_b64_str


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

    @staticmethod
    def _infer_image_field_type(example: Union[Image, dict, str]) -> ImageFieldType:
        if isinstance(example, Image):
            return ImageFieldType.pil_image
        if isinstance(example, str):
            return ImageFieldType.file_path
        if isinstance(example, dict) and all(k in example for k in ['bytes', 'path']):
            return ImageFieldType.hf_image_feature
        return ImageFieldType.unknown

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
            # TODO: cast to non decode
            example = dataset[0][imgs_location_colname]
            image_field_type = self._infer_image_field_type(example)

            if image_field_type == image_field_type.hf_image_feature:
                def hf_map_file_path(example):
                    example["text"] = _img_path_to_b64_str(example[imgs_location_colname])
                    return example
                dataset["text"] = dataset.map(hf_map_file_path)
            elif image_field_type == image_field_type.file_path:
                def hf_map_file_path(example):
                    example["text"] = _img_path_to_b64_str(example[imgs_location_colname])
                    return example
                dataset["text"] = dataset.map(hf_map_file_path)
            else:
                raise GalileoException(
                    f"Could not interpret column {imgs_location_colname} as either images"
                    "or image paths."
                )
        elif isinstance(dataset, pd.DataFrame):
            example = dataset[imgs_location_colname].values[0]
            image_field_type = self._infer_image_field_type(example)

            if image_field_type == image_field_type.file_path:
                dataset["text"] = dataset[imgs_location_colname].apply(
                    lambda x: _img_path_to_b64_str(img_path=os.path.join(imgs_dir, x))
                )
            elif image_field_type == image_field_type.pil_image:
                dataset["text"] = dataset[imgs_location_colname].apply(
                    _img_to_b64_str
                )
            else:
                raise GalileoException(
                    f"Could not interpret column {imgs_location_colname} as either images"
                    "or image paths."
                )
        else:
            raise GalileoException(
                f"Dataset must be one of pandas or HF, "
                f"but got {type(dataset)}"
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
