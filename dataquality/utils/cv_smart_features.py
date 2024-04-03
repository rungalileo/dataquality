import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import vaex
from imagededup.methods import PHash
from multiprocess import Pool, cpu_count
from PIL import ImageFilter, ImageStat
from PIL.Image import Image
from PIL.Image import open as Image_open
from vaex.dataframe import DataFrame

from dataquality.exceptions import GalileoWarning
from dataquality.schemas.cv import GAL_LOCAL_IMAGES_PATHS
from dataquality.schemas.cv import CVSmartFeatureColumn as CVSF

"""
CONSTANTS: the constants for all these methods were chosen conservatively with the goal
of minimizing the number of False Positives that they return.

METHODS: the methods for detecting smart feature all follow the same architecture with
a first method to quantify the anomaly (as a real number) followed by a method to
threshold it and return a boolean (qualitive).
For example to detect if a method is blurry we call
    blurriness = _blurry_laplace(image_gray)
    is_blurry = _is_blurry_laplace(blurriness)
These methods are kept separate to allow easy generalization to the use-case where we
compute the quantitative value first (the real-valued number, say via _blurry_laplace),
store it, and compute the qualitive value (boolean) by thresholding later at different
values (for example if we compute it at different prec/rec in the console).
"""

BLURRY_THREHSOLD = 110

LOW_CONTRAST_RANGE_THRESHOLD = 0.35
LOW_CONTRAST_MIN_Q = 0.02
LOW_CONTRAST_MAX_Q = 0.98

OVER_EXPOSED_MIN_THRESHOLD = 0.45
OVER_EXPOSED_MIN_Q = 0.02

UNDER_EXPOSED_MAX_THRESHOLD = 0.4
UNDER_EXPOSED_MAX_Q = 0.9

LOW_ENTROPY_THRESHOLD = 6

CHANNELS_TYPE_TO_MODE = {
    "BW": ["1"],
    "Gray": ["L", "LA", "La"],
    "Pallet": ["P"],
    "Color": ["RGB", "RGBA", "RGBa" "RGBX", "CMYK", "I", "F", "LAB", "YCbCr", "HSV"],
}
CHANNELS_DICT = {
    val: key for key, vals in CHANNELS_TYPE_TO_MODE.items() for val in vals
}

MEDIAN_RES_FACTOR_SMALL = 6
MEDIAN_RES_FACTOR_LARGE = MEDIAN_RES_FACTOR_SMALL

WIDE_OUTLIER_RATIO_STD = 4
TALL_OUTLIER_RATIO_STD = WIDE_OUTLIER_RATIO_STD

CHANNEL_RATIO_OUTLIER_THRESHOLD = 0.01

PHASH_NEAR_DUPLICATE_THRESHOLD = 7  # reduce based on the food101 dataset


def _has_odd_channels(df: DataFrame) -> np.ndarray:
    """
    Thresholding method to find outlier images that have odd channels.
    If a channel has less than CHANNEL_RATIO_OUTLIER_THRESHOLD% images, we consider
    all the images with that channel to be outliers.
    """
    dfg = df.groupby(CVSF.channels.value, agg=vaex.agg.count(CVSF.channels.value))
    channel_outliers = dfg[
        dfg[f"{CVSF.channels}_count"] < len(df) * CHANNEL_RATIO_OUTLIER_THRESHOLD
    ][CVSF.channels.value].tolist()
    outliers_channels = df.func.where(
        df[CVSF.channels.value].isin(channel_outliers), True, False
    ).to_numpy()
    return outliers_channels


def _has_odd_ratio(df: DataFrame) -> np.ndarray:
    """
    Thresholding method to find outlier images that have odd ratio.

    To find outliers that are very wide, we look at the images that are in landscape
    position and compute the mean and std of all width/height ratios. The images with
    width/height ratio > mean + WIDE_OUTLIER_RATIO_STD * std are considered outliers.

    We then repeat the same process to find very tall outliers.

    Finally we return as outliers all the images that are either wide outliers or tall
    outliers.
    """
    # If some images have height=0, we add a small epsilon to avoid division by zero
    if int((df[CVSF.height.value] < 1e-5).sum()) > 0:
        warnings.warn("Some images have height=0", GalileoWarning)
        df[CVSF.height.value] = df.func.where(
            df[CVSF.height.value] == 0, 1e-5, df[CVSF.height.value]
        )
    df["ratio_wh"] = df[CVSF.width.value] / df[CVSF.height.value]

    df_hor = df[df["ratio_wh"] >= 1]
    # If df_hor is empty that means that no images have bigger width than height, so we
    # can set this threshold at 1 and no images will have ratio above that
    wide_outlier_min_ratio = 1
    if len(df_hor) > 0:
        wide_outlier_min_ratio = (
            float(df_hor["ratio_wh"].mean())
            + WIDE_OUTLIER_RATIO_STD * df_hor["ratio_wh"].std()
        )

    df_vert = df[df["ratio_wh"] <= 1]
    # If df_vert is empty that means that no images have bigger height than width, so we
    # can set this threshold at 1 and no images will have ratio below that
    tall_outlier_max_ratio = 1
    if len(df_vert) > 0:
        tall_outlier_max_ratio = (
            float(df_vert["ratio_wh"].mean())
            - TALL_OUTLIER_RATIO_STD * df_vert["ratio_wh"].std()
        )

    outliers_ratio = df.func.where(
        (df["ratio_wh"] < tall_outlier_max_ratio)
        | (df["ratio_wh"] > wide_outlier_min_ratio),
        True,
        False,
    ).to_numpy()
    return outliers_ratio


def _has_odd_size(df: DataFrame) -> np.ndarray:
    """
    Thresholding method to find outlier images that have odd size.

    We compute the median resolution of all images, and consider all images with
        - resolution > mean * MEDIAN_RES_FACTOR_LARGE as a large outlier
        - resolution < mean * MEDIAN_RES_FACTOR_SMALL as a small outlier

    We return as outliers all images that are either large outliers or small outliers.
    """
    df["resolution"] = df[CVSF.height.value] * df[CVSF.width.value]
    # Vaex has no good method for computing the median (only an approximate or so),
    # we can bring it in memory and use numpy since it's a small df anyways
    median_resolution = np.median(df["resolution"].to_numpy())
    max_resolution = median_resolution * MEDIAN_RES_FACTOR_LARGE
    min_resolution = median_resolution / MEDIAN_RES_FACTOR_SMALL
    outliers_size = df.func.where(
        (df["resolution"] >= max_resolution) | (df["resolution"] <= min_resolution),
        True,
        False,
    ).to_numpy()
    return outliers_size


def _low_contrast_ranges(np_gray: np.ndarray) -> Tuple[float, float, float]:
    """
    Manual method to detect
    - over exposed images (too bright, histogram of intensities right-skewed)
    - under exposed images (too dark, histogram of intensities left-skewed)
    - low contrast images (histogram of intensities doesn't range too many values).
    We assuume that the image is gray scale with integer intensities in [0, 255].
    To try to reduce noise we only consider some quantiles of values, for example for
    detecting low contrast we remove low_contrast_min of low values and low_contrast_max
    of high values.

    Low contrast is detected when the resulting histogram does not span enough intensity
    values, i.e., if it spans less than a certain threshold % of the values.
    - for example if all pixels intesities are in the range [100, 150], then the
      histogram only spans 50 / 255 ~ 20% of the values. That is low contrast.

    Over exposed is detected when there aren't pixels with low intensity values, i.e.,
    if the min intensity of all pixesl is > some threshold.

    Under exposed is detected when there aren't pixels with high intensity values, i.e.,
    if the max intensity of all pixels is < some threshold.

    Note: First creating the histogram and then manually removing the edges seems faster
    than finding the quantile with np.quantile (since it's done multiple times).
    """
    MAX_VAL = 255  # already checked that the pixels are encoded as uint8 integers

    np_hist = np.histogram(
        np_gray, bins=MAX_VAL + 1, range=(0, MAX_VAL + 1), density=True
    )[0]

    q_min, q_max = _hist_keep_quantile(
        np_hist, q_min=LOW_CONTRAST_MIN_Q, q_max=LOW_CONTRAST_MAX_Q
    )
    contrast_range = (q_max - q_min) / MAX_VAL

    _, q_max_under = _hist_keep_quantile(np_hist, q_max=UNDER_EXPOSED_MAX_Q)
    q_max_under_f = float(q_max_under) / MAX_VAL
    q_min_over, _ = _hist_keep_quantile(np_hist, q_min=OVER_EXPOSED_MIN_Q)
    q_min_over_f = float(q_min_over) / MAX_VAL

    return contrast_range, q_max_under_f, q_min_over_f


def _hist_keep_quantile(
    np_hist: np.ndarray, q_min: float = 0.0, q_max: float = 1.0
) -> Tuple[int, int]:
    """
    Given a numpy array interpreted as a histogram, return the indices i_min and i_max
    containing the quantiles (q_min, q_max).

    So for example for a uniform distribution between 0-100 where the histogram is
    [0.01, 0.01, 0.01, ... , 0.01] and quantiles q_min = 0.2 and q_max = 0.75, this
    method returns i_min=19 and i_max=75 since the interval [20, 76] contains the
    quantile 20-75.
    """
    total_val = np.sum(np_hist)
    cumsum = 0
    for i, val in enumerate(np_hist):
        cumsum += val
        if cumsum > total_val * q_min:
            break
    i_min = i

    cumsum = 0
    for i, val in enumerate(np_hist[::-1]):
        cumsum += val
        if cumsum > total_val * (1 - q_max):
            break
    i_max = len(np_hist) - 1 - i
    return i_min, i_max


def _is_low_contrast(
    contrast_exp: np.ndarray,
    low_contrast_range: float = LOW_CONTRAST_RANGE_THRESHOLD,
) -> np.ndarray:
    """
    Thresholding method associated to low_contrast_ranges for low contrast
    """
    return contrast_exp <= low_contrast_range


def _is_over_exposed(
    q_min_over: np.ndarray,
    over_exposed_min_thresh: float = OVER_EXPOSED_MIN_THRESHOLD,
) -> np.ndarray:
    """
    Thresholding method associated to low_contrast_ranges for over exposure
    """
    return q_min_over >= over_exposed_min_thresh


def _is_under_exposed(
    q_max_over: np.ndarray,
    under_exposed_max_thresh: float = UNDER_EXPOSED_MAX_THRESHOLD,
) -> np.ndarray:
    """
    Thresholding method associated to low_contrast_ranges
    """
    return q_max_over <= under_exposed_max_thresh


def _blurry_laplace(image_gray: Image) -> float:
    """
    Bluriness detector method where we compute the Variance of the Laplacian.
    We use PIL to estimate the Laplacian via the 3x3 convolution filter
    -1, -1, -1
    -1,  8, -1
    -1, -1, -1.
    A LARGE variance indicates a lot of edges and thus a sharp image, whereas a SMALL
    variance indicates a blurry image without many edges found.
    There is no global threshold that will work for all images/datasets, experimentation
    suggests
    - conservative threshold without many FPs = 50
    - generous threshold containing more FPs = 350
    """
    edges = image_gray.filter(ImageFilter.FIND_EDGES)
    blurriness = ImageStat.Stat(edges).var[0]
    return blurriness


def _is_blurry_laplace(
    blurriness: np.ndarray, blurry_thresh: int = BLURRY_THREHSOLD
) -> np.ndarray:
    """
    Thresholding method associated to blurry_laplace
    """
    return blurriness < blurry_thresh


def _image_content_entropy(image: Image) -> float:
    """
    Returns the entropy of the pixels on the image. A high entropy means a more complex
    image with lots of variation in the pixel values (histogram closer to being uniform)
    whereas a low entropy means a simpler image with histogram more concentrated at a
    few specific intensities.

    There is no global threshold that will work for all images/datasets, experimentation
    suggests
      - conservative threshold without many FPs = 6
      - generous threshold containing more FPs = 7-8
    """
    return image.entropy()


def _is_low_content_entropy(
    image_entropy: np.ndarray, low_entropy_threshold: int = LOW_ENTROPY_THRESHOLD
) -> np.ndarray:
    """
    Thresholding method associated to image_content_entropy
    """
    return image_entropy < low_entropy_threshold


def _get_phash_encoding(hasher: PHash, np_gray: np.ndarray) -> str:
    """
    Get Perceptual hashes for every image. This hash will then be used to find near
    duplicated images: images with almost identical hashes (in terms of Hamming
    distance) will be near duplicates.
    """
    phash_str = ""
    if hasher is not None:
        phash_str = hasher.encode_image(image_array=np_gray)
    return phash_str


def _compute_near_duplicate_id(
    hasher: PHash,
    path_to_encodings: Dict[str, str],
    dist_thresh: int = PHASH_NEAR_DUPLICATE_THRESHOLD,
) -> Dict[str, int]:
    """
    Compute the Hamming distances pairwise between the encoded hashes, and group
    together images with distance < threshold.
    Returns:
        - a dictionary with keys=image_path, value=near_duplicate_id
    """
    # Mute the loggers from imagededup
    from imagededup.handlers.search.retrieval import logger as imagededup_handler_logger
    from imagededup.methods.hashing import logger as imagededup_hashing_logger

    imagededup_hashing_logger.disabled = True
    imagededup_handler_logger.disabled = True

    # Catch and hide warnings from imagededup
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # find_duplicates returns a dict with paths as keys and the values are a list of
        # images whose Hamming distance on the hash is < thresh (empty list when none).
        duplicates = hasher.find_duplicates(
            encoding_map=path_to_encodings,
            max_distance_threshold=dist_thresh,
        )

    # convert to a dict containing only the images with a near duplicate as keys, and a
    # near_duplicate_id as values. Near duplicate images have the same near_duplicate_id
    # Reserve the number 0 for images that are not near duplicate from any other image
    path_to_dup_id = {}
    group_id = 1
    for path, dups in duplicates.items():
        # The path could already have a group id assigned (by being the duplicate of an
        # earlier image already seen in the for loop)
        if path in path_to_dup_id:
            continue

        # Taking care of non duplicates later
        if not dups:
            continue

        # Add the path + its duplicates in the dict
        path_to_dup_id[path] = group_id
        for path_dup in dups:
            path_to_dup_id[path_dup] = group_id
        group_id += 1

    return path_to_dup_id


def _near_duplicate_id(
    df: DataFrame, hasher: PHash, images_paths: List[str]
) -> np.ndarray:
    """
    For every image
    """
    path_to_enc = {
        row[CVSF.image_path.value]: row[CVSF.hash.value]
        for row in df.to_records(column_names=[CVSF.image_path.value, CVSF.hash.value])
    }
    path_to_dup_id = _compute_near_duplicate_id(hasher, path_to_enc)
    np_near_duplicate_id = np.array(
        [path_to_dup_id.get(image_path, 0) for image_path in images_paths]
    )
    return np_near_duplicate_id


def _is_near_duplicate(in_frame: DataFrame) -> np.ndarray:
    np_is_near_duplicate = in_frame.func.where(
        in_frame[CVSF.outlier_near_duplicate_id.value] == 0, False, True
    ).to_numpy()
    return np_is_near_duplicate


def _open_and_resize(image_path: str) -> Tuple[Image, Image, np.ndarray]:
    """
    Open the image at the given path and return a triple with
    - the original image opened with PIL
    - the image converted to grey-scale with PIL
    - the numpy array of pixel intensities of the gray-scale image (as uint8)

    If any of the sides of the image is larger than 2**11, resize the image so that the
    largest side is now 2**11.
    """
    image = Image_open(image_path)
    image_gray = image.convert(
        "L"
    )  # TODO: check if image already grey, faster to skip that ?

    # Resize for both blurriness detector and increasing speed for processing big images
    w, h = image_gray.size
    if min(w, h) > 2**11:
        resize_factor = max(w, h) / 2**11  # resize the largest side to 2**11=2048
        image_gray = image_gray.resize((int(w / resize_factor), int(h / resize_factor)))
    np_gray = np.asarray(image_gray)

    if np_gray.dtype != "uint8":
        # I don't think this can ever happen, it's just an extra safety just in case
        warnings.warn("The images are not opened in uint8 format.", GalileoWarning)

    return image, image_gray, np_gray


def analyze_image_smart_features(
    image_path: str, hasher: Optional[PHash] = None
) -> Dict[str, Any]:
    """
    Evaluate if the image contains anomalies (too blurry, over exposed, etc) and return
    a dictionary storing a quantitative value for every such feature. These values have
    to be further thresholded (in the next method) to return a boolean.

    If no hasher is passed, the hash will be set to the empty string "".
    """
    image_data: Dict[str, Any] = {}

    image, image_gray, np_gray = _open_and_resize(image_path)

    # Near Dupliocates: compute hash encoding for every image
    image_data[CVSF.hash.value] = _get_phash_encoding(hasher, np_gray)

    # Odd Channels / Size / Ratio: gather image stats
    image_data.update(
        {
            CVSF.image_path.value: image_path,
            CVSF.width.value: image.width,
            CVSF.height.value: image.height,
            CVSF.channels.value: CHANNELS_DICT.get(image.mode),
        }
    )

    # Blurriness: compute variance of Laplacian
    image_data[CVSF.blur.value] = _blurry_laplace(image_gray)

    # Low contrast: compute range of all / low / high pixel intensities
    contrast_range, q_max_under, q_min_over = _low_contrast_ranges(np_gray)
    image_data.update(
        {
            CVSF.contrast.value: contrast_range,
            CVSF.underexp.value: q_max_under,
            CVSF.overexp.value: q_min_over,
        }
    )

    # Low content: compute image entropy
    image_data[CVSF.lowcontent.value] = _image_content_entropy(image)

    return image_data


def analyze_image_smart_features_wrapper(hasher: Optional[PHash] = None) -> Callable:
    """
    Wrapper around analyze_image_smart_features to allow calling with only the path
    argument and making it easier to parallelize.
    """

    def analyze_image_smart_features_only_image_path(image_path: str) -> Dict[str, Any]:
        return analyze_image_smart_features(image_path, hasher)

    return analyze_image_smart_features_only_image_path


def generate_smart_features(in_frame: DataFrame, n_cores: int = -1) -> DataFrame:
    """
    Create and return a dataframe containing the  smart features on images (blurriness,
    contrast, etc).

    Can run in parallel if n_cores is specified and different than 1. To use all
    available cores set n_cores = -1.
    """
    hasher = PHash()
    images_data: List[dict] = []

    # Collect the quantitative features to find anomalies (bluriness, num_channels,
    # width, etc). Called serially or in parallel depending on n_cores
    images_paths = in_frame[GAL_LOCAL_IMAGES_PATHS].tolist()
    if n_cores == 1:
        for image_path in images_paths:
            image_data = analyze_image_smart_features(image_path, hasher)
            images_data.append(image_data)
    else:
        p = Pool(n_cores if n_cores != -1 else cpu_count())
        analyze_image_smart_features_par = analyze_image_smart_features_wrapper(hasher)
        images_data = p.map(analyze_image_smart_features_par, images_paths)
    df = vaex.from_records(images_data)

    # Add smart features to the dataframe
    in_frame[CVSF.outlier_channels.value] = _has_odd_channels(df)
    in_frame[CVSF.outlier_ratio.value] = _has_odd_ratio(df)
    in_frame[CVSF.outlier_size.value] = _has_odd_size(df)
    in_frame[CVSF.outlier_blurry.value] = _is_blurry_laplace(
        df[CVSF.blur.value].to_numpy()
    )
    in_frame[CVSF.outlier_low_contrast.value] = _is_low_contrast(
        df[CVSF.contrast.value].to_numpy()
    )
    in_frame[CVSF.outlier_overexposed.value] = _is_over_exposed(
        df[CVSF.overexp.value].to_numpy()
    )
    in_frame[CVSF.outlier_underexposed.value] = _is_under_exposed(
        df[CVSF.underexp.value].to_numpy()
    )
    in_frame[CVSF.outlier_low_content.value] = _is_low_content_entropy(
        df[CVSF.lowcontent.value].to_numpy()
    )
    in_frame[CVSF.outlier_near_duplicate_id.value] = _near_duplicate_id(
        df, hasher, images_paths
    )
    in_frame[CVSF.outlier_near_dup.value] = _is_near_duplicate(in_frame)

    return in_frame
