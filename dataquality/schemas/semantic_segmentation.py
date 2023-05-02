from enum import Enum


class SemSegCols(str, Enum):
    id = "id"
    image_path = "image_path"
    mask_path = "mask_path"
    # mixin restriction on str (due to "str".split(...))
    split = "split"  # type: ignore
    meta = "meta"
