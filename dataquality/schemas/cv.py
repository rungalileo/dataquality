from dataclasses import dataclass
from enum import Enum

GAL_LOCAL_IMAGES_PATHS = "gal_local_images_paths"


@dataclass
class CVSmartFeatureColumn(str, Enum):
    """
    A class holding the column names appearing with the smart feature methods.
    When updated, also need to update the coresponding schema in rungalileo.
    """

    image_path: str = "sf_image_path"
    height: str = "sf_height"
    width: str = "sf_width"

    channels: str = "sf_channels"
    hash: str = "sf_hash"
    contrast: str = "sf_contrast"
    overexp: str = "sf_overexposed"
    underexp: str = "sf_underexposed"
    blur: str = "sf_blur"
    lowcontent: str = "sf_content"

    # Only the columns below will be displayed in the console as SmartFeatures
    outlier_size: str = "has_odd_size"
    outlier_ratio: str = "has_odd_ratio"
    outlier_near_duplicate_id: str = "near_duplicate_id"
    outlier_near_dup: str = "is_near_duplicate"
    outlier_channels: str = "has_odd_channels"
    outlier_low_contrast: str = "has_low_contrast"
    outlier_overexposed: str = "is_overexposed"
    outlier_underexposed: str = "is_underexposed"
    outlier_low_content: str = "has_low_content"
    outlier_blurry: str = "is_blurry"
