from tempfile import NamedTemporaryFile
from typing import List, Tuple

import evaluate
import numpy as np
import torch
from PIL import Image

from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import GALILEO_DEFAULT_RESULT_BUCKET_NAME

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


def calculate_mean_iou(
    pred_masks: torch.Tensor, gold_masks: torch.Tensor, nc: int = 21
) -> Tuple[List[float], List[np.ndarray]]:
    """Calculates the Mean Intersection Over Union (mIoU) for each image in the batch

    :param pred_masks: argmax of the prediction probabilities
       shape = (bs, height, width)
    :param gold_masks: ground truth masks
       shape = (bs, height, width)
    :param nc: number of classes

    returns: Tuple:
      - List of mIoU values for each image in the batch
      - List of mIoU values for each image per class in the batch
    """
    metric = evaluate.load("mean_iou")
    mean_ious = []
    per_class_ious = []

    # for iou need shape (bs, 1, height, width) to get per mask iou
    for i in range(len(pred_masks)):
        iou = metric._compute(
            pred_masks[i : i + 1],  # tensor (1, height, width)
            gold_masks[i : i + 1],  # tensor (1, height, width)
            num_labels=nc,
            ignore_index=255,
        )
        mean_ious.append(iou["mean_iou"].item())
        per_class_ious.append(iou["per_category_iou"])
    return mean_ious, per_class_ious
