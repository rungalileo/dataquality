import datasets

from dataquality.utils.cv import _bytes_to_img, _write_image_bytes_to_objectstore


def _hf_map_image_feature(example: dict, imgs_colname: str) -> dict:
    image = example[imgs_colname]

    if image["bytes"] is None:
        # sometimes the Image feature only contains a path
        # example: beans dataset
        # assume abs path for hf
        example["text"] = _write_image_bytes_to_objectstore(
            img_path=image["path"],
        )
    else:
        example["text"] = _write_image_bytes_to_objectstore(
            img=_bytes_to_img(image["bytes"]),
        )

    return example


def process_hf_image_feature_for_logging(
    dataset: datasets.Dataset, imgs_colname: str
) -> datasets.Dataset:
    dataset = dataset.cast_column(imgs_colname, datasets.Image(decode=False))

    dataset = dataset.map(
        _hf_map_image_feature,
        fn_kwargs=dict(imgs_colname=imgs_colname),
    )
    return dataset


def _hf_map_image_file_path(example: dict, imgs_location_colname: str) -> dict:
    example["text"] = _write_image_bytes_to_objectstore(
        # assume abs paths for HF
        img_path=example[imgs_location_colname]
    )
    return example


def process_hf_image_paths_for_logging(
    dataset: datasets.Dataset, imgs_location_colname: str
) -> datasets.Dataset:
    dataset = dataset.map(
        _hf_map_image_file_path,
        fn_kwargs=dict(imgs_location_colname=imgs_location_colname),
    )
    return dataset
