import os
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np
import torch
from PIL import ImageColor
from PIL.Image import Image, fromarray

from dataquality import config
from dataquality.clients.objectstore import ObjectStore
from dataquality.schemas.semantic_segmentation import (
    Polygon,
    SemSegMetricData,
    SemSegMetricType,
)
from dataquality.utils.semantic_segmentation.polygons import draw_polygon

object_store = ObjectStore()

MAX_DEP_HEATMAP_SIZE = 128


def calculate_and_upload_dep(
    probs: torch.Tensor,
    gold_masks: torch.Tensor,
    image_ids: List[int],
    local_folder_path: str,
) -> torch.Tensor:
    """Calculates the Data Error Potential (DEP) for each image in the batch

    Uploads the heatmap to Minio as a png.
    Returns the dep_heatmaps
    """
    dep_heatmaps = calculate_dep_heatmaps(probs, gold_masks)
    write_dep_to_disk(dep_heatmaps, image_ids, local_folder_path)
    return dep_heatmaps


def colorize_dep_heatmap(image: Image, dep_mean: int) -> Image:
    """Recolors a grayscale image to a color image based on our dep mapping"""
    color_1 = ImageColor.getrgb("#9bc33f")  # Red
    color_2 = ImageColor.getrgb("#ece113")  # Yellow
    color_3 = ImageColor.getrgb("#ba3612")  # Green

    # Convert the image to RGB mode if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Create a new image with the same size and mode as the original
    image_np = np.array(image)
    height, width, _ = image_np.shape
    colorized_image = np.zeros((height, width, 3))
    threshold_mask = image_np[:, :, 0] <= dep_mean

    ratio = (image_np / dep_mean)[threshold_mask][:, 0]
    colorized_image[threshold_mask, 0] = (1 - ratio) * color_1[0] + ratio * color_2[0]
    colorized_image[threshold_mask, 1] = (1 - ratio) * color_1[1] + ratio * color_2[1]
    colorized_image[threshold_mask, 2] = (1 - ratio) * color_1[2] + ratio * color_2[2]

    ratio = ((image_np - dep_mean) / (255 - dep_mean))[~threshold_mask][:, 0]
    colorized_image[~threshold_mask, 0] = (1 - ratio) * color_2[0] + ratio * color_3[0]
    colorized_image[~threshold_mask, 1] = (1 - ratio) * color_2[1] + ratio * color_3[1]
    colorized_image[~threshold_mask, 2] = (1 - ratio) * color_2[2] + ratio * color_3[2]

    return fromarray(colorized_image.astype(np.uint8))


def calculate_dep_heatmaps(
    probs: torch.Tensor, gold_masks: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Data Error Potential (DEP) for each image in the batch

    :param probs: np array of floats, size = (bs, height, width, n_classes)
    :param gold_masks: np array of gold masks as ints, size = (bs, height, width)
    :return: (bs, height, width)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = probs.clone().to(device)
    gold_masks = gold_masks.to(device)
    n_classes = probs.shape[-1]
    bs = probs.shape[0]
    # flatten the height and width dimensions
    probs = probs.view(bs, -1, n_classes)  # (bs, n_pixels, n_classes)
    mask_size = gold_masks.shape
    gold_masks = gold_masks.view(bs, -1, 1)  # (bs, n_pixels, 1)
    gold_indices = (
        gold_masks.reshape((bs, -1, 1)).expand(-1, -1, probs.shape[2]).type(torch.int64)
    )  # (bs, n_pixels, n_classes)
    value_at_gold = torch.gather(probs, 2, gold_indices)[:, :, 0]  # (bs, n_pixels)

    next_highest = probs
    # Takes GT indices and puts 0 at that index so we don't use it as next highest value
    next_highest.scatter_(2, gold_indices, 0)
    next_highest = next_highest.max(dim=2).values
    margin = value_at_gold - next_highest
    # Since margin is between -1 and 1, we normalize it to be between 0 and 1
    normalized_margin = (1 + margin) / 2
    dep_masks = 1 - normalized_margin
    dep_masks = dep_masks.view(mask_size)
    gold_masks = gold_masks.cpu()
    return dep_masks.cpu()


def write_dep_to_disk(
    dep_heatmaps: torch.Tensor,
    image_ids: List[int],
    local_folder_path: str,
) -> None:
    """Writes dep to disk as a png locally

    Args:
        dep_heatmaps (torch.Tensor): bs x height x width dep heatmaps
        image_ids (List[int]): image id for each image in the batch
        local_folder_path (str): folder path to store the dep heatmaps
    """
    os.makedirs(local_folder_path, exist_ok=True)
    for i, image_id in enumerate(image_ids):
        dep_heatmap = dep_heatmaps[i].numpy()
        obj_name = f"{local_folder_path}/{image_id}.png"
        with open(obj_name, "wb") as f:
            img = dep_heatmap_to_img(dep_heatmap)
            img = colorize_dep_heatmap(img, 128)
            img.save(f, "PNG")


def upload_dep_heatmaps(
    dep_heatmaps: torch.Tensor,
    image_ids: List[int],
    obj_prefix: str,
) -> None:
    """Uploads dep heatmap to Minio for each image in the batch

    :param dep_heatmaps: DEP heatmap for each image in the batch
        shape = (bs, height, width)
    """
    for i, image_id in enumerate(image_ids):
        dep_heatmap = dep_heatmaps[i].numpy()
        obj_name = f"{obj_prefix}/{image_id}.png"
        with NamedTemporaryFile(suffix=".png", mode="w+") as f:
            img = dep_heatmap_to_img(dep_heatmap)
            img = colorize_dep_heatmap(img, 128)
            img.save(f.name)

            object_store.create_object(
                object_name=obj_name,
                file_path=f.name,
                content_type="image/png",
                progress=False,
                bucket_name=config.results_bucket_name,
            )


def dep_heatmap_to_img(dep_heatmap: np.ndarray) -> Image:
    """Converts DEP heatmap to PIL Image
    We cast the heatmap to a 1-channel PIL Image object in grey scale
    and store it as a PNG file in Minio in order to compress the file size
    as much as possible.

    To keep DEP heatmaps small, the maximum heatmap size is 64x64 pixels.

    :param dep_heatmap: DEP heatmap for each image in the batch
        shape = (height, width)
    :return: PIL Image object
    """
    # Scale the array values to the range [0, 255]
    dep_heatmap = (dep_heatmap * 255).astype(np.uint8)
    # Create a PIL Image object from the numpy array as grey-scale
    img = fromarray(dep_heatmap, mode="L")
    if img.size[0] > MAX_DEP_HEATMAP_SIZE or img.size[1] > MAX_DEP_HEATMAP_SIZE:
        img = img.resize((MAX_DEP_HEATMAP_SIZE, MAX_DEP_HEATMAP_SIZE))
    return img


def compute_metric(
    pred_mask: np.ndarray,
    gold_mask: np.ndarray,
    metric_type: SemSegMetricType,
    num_labels: int,
) -> SemSegMetricData:
    """Computes the SemSeg metric data for a single image

    See 'calculate_batch_metric' for more details on the metrics

    Args:
        pred_mask (np.ndarray): argmax of the probability predictions
        gold_mask (np.ndarray): ground truth mask
        metric_type (SemSegMetricType): type of metric to compute
        num_labels (int): number of classes

    Returns:
        SemSegMetricData: metric data for the image
    """
    intersection_bool = pred_mask == gold_mask

    intersection_pixels = np.histogram(
        pred_mask[intersection_bool], bins=num_labels, range=(0, num_labels)
    )[0]
    pred_pixels = np.histogram(pred_mask, bins=num_labels, range=(0, num_labels))[0]
    gold_pixels = np.histogram(gold_mask, bins=num_labels, range=(0, num_labels))[0]

    if metric_type == SemSegMetricType.dice:
        pixels_per_class = pred_pixels + gold_pixels
        metric_per_class = (2 * intersection_pixels) / pixels_per_class
    else:
        pixels_per_class = pred_pixels + gold_pixels - intersection_pixels
        metric_per_class = intersection_pixels / pixels_per_class

    # fill the nans with 0s
    pixels_per_class = np.nan_to_num(pixels_per_class)
    data = SemSegMetricData(
        metric=metric_type,
        value=np.nanmean(metric_per_class),
        value_per_class=metric_per_class.tolist(),
        area_per_class=pixels_per_class.tolist(),
    )

    return data


def calculate_batch_metric(
    pred_masks: torch.Tensor,
    gold_masks: torch.Tensor,
    metric_type: SemSegMetricType,
    num_labels: int,
) -> List[SemSegMetricData]:
    """Function to calcuate semseg metrics for each image in a batch

    SemSeg metrics can be one of:
        - Mean IoU
        - Boundary IoU
        - Dice coefficient

    Dice score is the intersection over the total number of pixels per class.
    We take the 'macro' average where we weight each class's dice score by the
    amount of pixels in that class, and then after computing each class's dice
    score we average them together.

    IoU is simply the intersection over union, where boundary IoU is the IoU
    computed on the boundary masks.

    Args:
        pred_masks (torch.Tensor): argmax of predicted probabilities
        gold_masks (torch.Tensor): ground truth masks
        metric_type (SemSegMetricType): type of metric to compute
        num_labels (int): number of classes

    Returns:
        List[SemSegMetricData]: A metric data object for each image in the batch
    """
    pred_masks_np = pred_masks.numpy()
    gold_masks_np = gold_masks.numpy()

    metric_data = []
    for i in range(len(pred_masks)):
        metric_data.append(
            compute_metric(
                pred_masks_np[i : i + 1],
                gold_masks_np[i : i + 1],
                metric_type,
                num_labels,
            )
        )
    return metric_data


def calculate_polygon_area(
    polygon: Polygon,
    height: int,
    width: int,
) -> int:
    """Calculates the area for a polygon
    Args:
        polygon (Polygon): polygon to calculate area for
        height (int)
        width (int)

    Returns:
        int: area of the polygon
    """
    polygon_img = draw_polygon(polygon, (height, width))
    return (polygon_img != 0).sum()


def add_area_to_polygons(
    polygons: List[Polygon],
    height: int,
    width: int,
) -> None:
    """Adds the area of each polygon in an image to the obj

    Args:
        polygons (List[Polygon]): list of each images polygons
        height (int)
        width (int)
    """
    for polygon in polygons:
        polygon.area = calculate_polygon_area(polygon, height, width)


def add_area_to_polygons_batch(
    polygon_batch: List[List[Polygon]],
    heights: List[int],
    widths: List[int],
) -> None:
    """Calculates the area for every polygon in a btach
    Args:
        polygon_batch (List[List[Polygon]]): list of each images polygons
        size (Tuple[int, int]): shape to draw the polygons onto
    """
    for idx in range(len(polygon_batch)):
        add_area_to_polygons(polygon_batch[idx], heights[idx], widths[idx])
