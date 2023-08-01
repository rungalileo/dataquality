from glob import glob
from pathlib import Path

from dataquality.utils.cv_smart_features import (
    _blurry_laplace,
    _compute_near_duplicate_id,
    _get_phash_encoding,
    _image_content_entropy,
    _is_blurry_laplace,
    _is_low_content_entropy,
    _is_low_contrast,
    _is_over_exposed,
    _is_under_exposed,
    _low_contrast_ranges,
    _open_and_resize,
    analyze_image_smart_features,
    generate_smart_features,
)
from tests.assets.constants import TEST_ASSETS_SMART_FEATS_DIR

low_contrast_imagename = "lowcontrast.jpeg"
over_exposed_imagename = "overexposed.jpeg"
under_exposed_imagename = "underexposed.jpeg"
blurry_imagename = "blurry.jpeg"
lowcontent_imagename = "lowcontent.jpeg"
lowcontent_neardup_imagename = "lowcontent2.jpeg"
lowcontent_neardup_imagename2 = "lowcontent3.jpeg"


def test_low_contrast() -> None:
    """Test the method detecting low contrast images on a test image"""
    image_path = Path(TEST_ASSETS_SMART_FEATS_DIR) / low_contrast_imagename
    _, _, np_gray = _open_and_resize(image_path)
    contrast_range, _, _ = _low_contrast_ranges(np_gray)
    is_low_contrast = _is_low_contrast(contrast_range)
    assert is_low_contrast


def test_over_exposed() -> None:
    """Test the method detecting over exposed images on a test image"""
    image_path = Path(TEST_ASSETS_SMART_FEATS_DIR) / over_exposed_imagename
    _, _, np_gray = _open_and_resize(image_path)
    _, _, q_min_over_f = _low_contrast_ranges(np_gray)
    assert _is_over_exposed(q_min_over_f)


def test_under_exposed() -> None:
    """Test the method detecting under exposed images on a test image"""
    image_path = Path(TEST_ASSETS_SMART_FEATS_DIR) / under_exposed_imagename
    _, _, np_gray = _open_and_resize(image_path)
    _, q_max_under_f, _ = _low_contrast_ranges(np_gray)
    assert _is_under_exposed(q_max_under_f)


def test_blurry() -> None:
    """Test the method detecting blurry images on a test image"""
    image_path = Path(TEST_ASSETS_SMART_FEATS_DIR) / blurry_imagename
    _, image_gray, _ = _open_and_resize(image_path)
    blurriness = _blurry_laplace(image_gray)
    assert _is_blurry_laplace(blurriness)


def test_low_content() -> None:
    """Test the method detecting low content images on a test image"""
    image_path = Path(TEST_ASSETS_SMART_FEATS_DIR) / lowcontent_imagename
    image, _, _ = _open_and_resize(image_path)
    image_entropy = _image_content_entropy(image)
    assert _is_low_content_entropy(image_entropy)


def test_near_duplicates() -> None:
    """Test the method detecting near duplicate images on a test set"""
    from imagededup.methods import PHash

    hasher = PHash()
    name_to_hash = {}

    for image_path in glob(TEST_ASSETS_SMART_FEATS_DIR + "/*.jpeg"):
        _, _, np_gray = _open_and_resize(image_path)
        image_name = image_path.split("/")[-1]
        name_to_hash[image_name] = _get_phash_encoding(hasher, np_gray)

    path_to_dup_id = _compute_near_duplicate_id(hasher, name_to_hash)

    # assert that the only duplicates are the two low content images (dup to each other)
    assert path_to_dup_id == {
        lowcontent_imagename: 1,
        lowcontent_neardup_imagename: 1,
        lowcontent_neardup_imagename2: 1,
    }


def test_near_duplicate_id() -> None:
    """Test the method creating near duplicate group ids on a test set"""
    images_names = [
        low_contrast_imagename,
        over_exposed_imagename,
        under_exposed_imagename,
        blurry_imagename,
        lowcontent_imagename,
        lowcontent_neardup_imagename,
        lowcontent_neardup_imagename2,
    ]
    images_paths = [
        f"{TEST_ASSETS_SMART_FEATS_DIR}/{image_name}" for image_name in images_names
    ]
    df = generate_smart_features(images_paths)

    assert len(df) == len(images_names)
    outlier_cols = {
        "is_near_duplicate",
        "near_duplicate_id",
        "is_blurry",
        "is_underexposed",
        "is_overexposed",
        "is_low_contrast",
        "has_low_content",
        "has_odd_size",
        "has_odd_ratio",
        "has_odd_channels",
    }
    assert outlier_cols.issubset(df.columns)


def test_analyze_image_smart_features() -> None:
    """Test the entire workflow of checking image anomalies on one specific image"""

    image_path = Path(TEST_ASSETS_SMART_FEATS_DIR) / blurry_imagename
    image_data = analyze_image_smart_features(image_path)

    assert image_data["sf_hash"] == ""  # since we didn't pass a hasher
    assert str(image_data["sf_image_path"]) == str(image_path)  # check correct image
    assert image_data["sf_width"] == 678  # Get width from image viewer
    assert image_data["sf_height"] == 446  # Get height from image viewer
    assert image_data["sf_channels"] == "Color"  # RGB image
    assert _is_blurry_laplace(image_data["sf_blur"])
    assert not _is_low_contrast(image_data["sf_contrast"])
    assert not _is_under_exposed(image_data["sf_underexp"])
    assert not _is_over_exposed(image_data["sf_overexp"])
    assert not _is_low_content_entropy(image_data["sf_content"])


def test_generate_smart_features():
    """Check the method creating smart features on multiple images"""
    images_names = [
        low_contrast_imagename,
        over_exposed_imagename,
        under_exposed_imagename,
        blurry_imagename,
        lowcontent_imagename,
        lowcontent_neardup_imagename,
        lowcontent_neardup_imagename2,
    ]
    images_paths = [
        f"{TEST_ASSETS_SMART_FEATS_DIR}/{image_name}" for image_name in images_names
    ]
    df = generate_smart_features(images_paths)

    assert len(df) == len(images_names)
    outlier_cols = {
        "is_near_duplicate",
        "near_duplicate_id",
        "is_blurry",
        "is_underexposed",
        "is_overexposed",
        "is_low_contrast",
        "has_low_content",
        "has_odd_size",
        "has_odd_ratio",
        "has_odd_channels",
    }
    assert outlier_cols.issubset(df.columns)
