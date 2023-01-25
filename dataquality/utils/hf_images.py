import datasets

from dataquality.exceptions import GalileoException
from dataquality.utils.cv import _bytes_to_b64_str, _img_path_to_b64_str


def _hf_map_image_feature(example: dict, imgs_colname: str) -> dict:
    image = example[imgs_colname]

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


def process_hf_image_feature_for_logging(
    dataset: datasets.Dataset, imgs_colname: str
) -> datasets.Dataset:
    if not isinstance(dataset.features[imgs_colname], datasets.Image):
        raise GalileoException(
            f"Got imgs_colname={repr(imgs_colname)}, but that "
            "dataset feature does not contain images. If your dataset has "
            "image paths, pass imgs_location_colname instead."
        )

    dataset = dataset.cast_column(imgs_colname, datasets.Image(decode=False))

    dataset = dataset.map(
        _hf_map_image_feature, fn_kwargs=dict(imgs_colname=imgs_colname)
    )
    return dataset


def _hf_map_image_file_path(example: dict, imgs_location_colname: str) -> dict:
    example["text"] = _img_path_to_b64_str(
        # assume abs paths for HF
        example[imgs_location_colname]
    )
    return example


def process_hf_image_paths_for_logging(
    dataset: datasets.Dataset, imgs_location_colname: str
) -> datasets.Dataset:
    if dataset.features[imgs_location_colname].dtype != "string":
        raise GalileoException(
            f"Got imgs_location_colname={repr(imgs_location_colname)}, but that "
            "dataset feature does not contain strings. If your dataset uses "
            "datasets.Image features, pass imgs_colname instead."
        )

    dataset = dataset.map(
        _hf_map_image_file_path,
        fn_kwargs=dict(imgs_location_colname=imgs_location_colname),
    )
    return dataset
