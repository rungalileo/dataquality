import os
from tempfile import NamedTemporaryFile
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageColor

from dataquality import config
from dataquality.clients.objectstore import ObjectStore
from dataquality.schemas.semantic_segmentation import IouData, Polygon
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


def colorize_dep_heatmap(image: Image.Image, dep_mean: int) -> Image.Image:
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

    return Image.fromarray(colorized_image.astype(np.uint8))


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
    img = Image.fromarray(dep_heatmap, mode="L")
    if img.size[0] > MAX_DEP_HEATMAP_SIZE or img.size[1] > MAX_DEP_HEATMAP_SIZE:
        img = img.resize((MAX_DEP_HEATMAP_SIZE, MAX_DEP_HEATMAP_SIZE))
    return img


def calculate_batch_iou(
    pred_masks: torch.Tensor, gold_masks: torch.Tensor, iou_type: str, nc: int
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
    :return: list of IoU data for each image in the batch
       shape = (bs,)
    """
    pred_masks_np = pred_masks.numpy()
    gold_masks_np = gold_masks.numpy()
    iou_data = []

    # for iou need shape (bs, 1, height, width) to get per mask iou
    for i in range(len(pred_masks_np)):
        iou, area_per_class = compute_iou(
            pred_masks_np[i : i + 1],  # tensor (1, height, width)
            gold_masks_np[i : i + 1],  # tensor (1, height, width)
            num_labels=nc,
        )

        iou_data.append(
            IouData(
                iou=np.nanmean(iou),
                iou_per_class=iou.tolist(),
                area_per_class=area_per_class.tolist(),
                iou_type=iou_type,
            )
        )
    return iou_data


def compute_iou(
    pred_mask: np.ndarray,
    gold_mask: np.ndarray,
    num_labels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes the intersection over union for a single image

    Computes the iou for a single image as well as returning the total union area
    per class

    Args:
        pred_mask (np.ndarray): (h, w) pred mask
        gold_mask (np.ndarray): (h, w) gold mask
        num_labels (int): total number of labels including background

    Returns:
        Tuple[np.ndarray, np.ndarray]: the iou per class and the union area per class
            in numpy array
    """
    intersection_bool = pred_mask == gold_mask

    intersection_pixels = np.histogram(
        pred_mask[intersection_bool], bins=num_labels, range=(0, num_labels)
    )[0]
    pred_pixels = np.histogram(pred_mask, bins=num_labels, range=(0, num_labels))[0]
    gold_pixels = np.histogram(gold_mask, bins=num_labels, range=(0, num_labels))[0]

    union_pixels_per_class = pred_pixels + gold_pixels - intersection_pixels
    iou_per_class = intersection_pixels / union_pixels_per_class

    # fill the nans with 0s
    union_pixels_per_class = np.nan_to_num(union_pixels_per_class)

    return iou_per_class, union_pixels_per_class


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
