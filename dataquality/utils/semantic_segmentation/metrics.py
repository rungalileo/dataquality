from tempfile import NamedTemporaryFile
from typing import List, Tuple

import evaluate
import numpy as np
import torch
from PIL import Image

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import GALILEO_DEFAULT_RESULT_BUCKET_NAME
from dataquality.schemas.semantic_segmentation import IouData, IoUType, Polygon
from dataquality.utils.semantic_segmentation.polygons import draw_polygon

object_store = ObjectStore()

MAX_DEP_HEATMAP_SIZE = 64


def calculate_and_upload_dep(
    probs: torch.Tensor,
    gold_masks: torch.Tensor,
    image_ids: List[int],
    obj_prefix: str,
) -> Tuple[List[float], torch.Tensor]:
    """Calculates the Data Error Potential (DEP) for each image in the batch

    Uploads the heatmap to Minio as a png.
    Returns the image DEP for each image in the batch. As well as the dep_heatmaps.
        Image dep is calculated by the average pixel dep.
    """
    dep_heatmaps = calculate_dep_heatmaps(probs, gold_masks)
    upload_dep_heatmaps(dep_heatmaps, image_ids, obj_prefix)
    return calculate_image_dep(dep_heatmaps), dep_heatmaps


def calculate_dep_heatmaps(
    probs: torch.Tensor, gold_masks: torch.Tensor
) -> torch.Tensor:
    """
    Calculates the Data Error Potential (DEP) for each image in the batch

    :param probs: np array of floats, size = (bs, height, width, n_classes)
    :param gold_masks: np array of gold masks as ints, size = (bs, height, width)
    :return: (bs, height, width)
    """
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

    next_highest = probs.clone()
    # Takes GT indices and puts 0 at that index so we don't use it as next highest value
    next_highest.scatter_(2, gold_indices, 0)
    next_highest = next_highest.max(dim=2).values

    margin = value_at_gold - next_highest
    # Since margin is between -1 and 1, we normalize it to be between 0 and 1
    normalized_margin = (1 + margin) / 2
    dep_masks = 1 - normalized_margin
    dep_masks = dep_masks.view(mask_size)
    return dep_masks


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
            img.save(f.name)

            object_store.create_object(
                object_name=obj_name,
                file_path=f.name,
                content_type="image/png",
                progress=False,
                bucket_name=GALILEO_DEFAULT_RESULT_BUCKET_NAME,
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
    img = Image.fromarray(dep_heatmap, mode="L")
    if img.size[0] > MAX_DEP_HEATMAP_SIZE or img.size[1] > MAX_DEP_HEATMAP_SIZE:
        img.resize((MAX_DEP_HEATMAP_SIZE, MAX_DEP_HEATMAP_SIZE))
    return img


def calculate_image_dep(dep_heatmap: torch.Tensor) -> List[float]:
    """Calculates the Data Error Potential (DEP) for each image in the batch

    :param dep_heatmap: DEP heatmap for each image in the batch
        size = (bs, height, width)
    :return: list of DEP values for each image in the batch
        size = (bs,)
    """
    return dep_heatmap.mean(dim=(1, 2)).tolist()


def calculate_union_area(
    pred_mask: torch.Tensor, gold_mask: torch.Tensor, nc: int
) -> List[int]:
    """Calculates the union area for each class in the batch

    Where the union area is the area of the union of the predicted mask and the
    ground truth mask.

    :param pred_mask: argmax of the prediction probabilities
         shape = (height, width)
    :param gold_mask: ground truth masks
            shape = (height, width)
    :param nc: number of classes

    returns: List (int) of union area values for each class in the batch
    """
    per_class_union_area = []
    for i in range(nc):
        current_pred_mask = np.where(pred_mask == i, 1, 0)
        current_gold_mask = np.where(gold_mask == i, 1, 0)
        # take the elementwise and
        union_mask = np.logical_and(current_pred_mask, current_gold_mask)
        # calculate the area
        union_area = np.sum(union_mask)
        all_area = np.sum(current_pred_mask) + np.sum(current_gold_mask)
        per_class_union_area.append(all_area - union_area)
    return per_class_union_area


def calculate_mean_iou(
    pred_masks: torch.Tensor, gold_masks: torch.Tensor, iou_type: IoUType, nc: int
) -> List[IouData]:
    """Calculates the Mean Intersection Over Union (mIoU) for each image in the batch

    If boundary masks are passed into this function, we return the
    boundary IoU (bIoU).

    :param pred_masks: argmax of the prediction probabilities
       shape = (bs, height, width)
    :param gold_masks: ground truth masks
       shape = (bs, height, width)
    :param iou_type: mean or boundary
    :param nc: number of classes

    returns: List[IouData], where IouData contains:
      - iou value for the image
      - list of iou values per class
      - list of areas per class (unioned over pred and gold masks)
    """
    metric = evaluate.load("mean_iou")
    iou_data = []

    # for iou need shape (bs, 1, height, width) to get per mask iou
    for i in range(len(pred_masks)):
        iou = metric._compute(
            pred_masks[i : i + 1],  # tensor (1, height, width)
            gold_masks[i : i + 1],  # tensor (1, height, width)
            num_labels=nc,
            ignore_index=255,
        )
        iou_data.append(
            IouData(
                iou=iou["mean_iou"].item(),
                iou_per_class=iou["per_category_iou"].tolist(),
                area_per_class=calculate_union_area(pred_masks[i], gold_masks[i], nc),
                iou_type=iou_type,
            )
        )
    return iou_data


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
