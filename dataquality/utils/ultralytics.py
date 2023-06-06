import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import yaml
from torchvision.ops.boxes import box_convert, box_iou
from ultralytics.yolo.utils.plotting import Colors

from dataquality.schemas.split import Split

colors = Colors()

# helper functions for ploting predictions


def box_label(
    image: Any,
    box: Any,
    label: str = "",
    color: Tuple[int, int, int] = (128, 128, 128),
    txt_color: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw a labeled bounding box on an image.

    :param image: The image to draw on.
    :param box: The bounding box to draw. The format is (x1, y1, x2, y2).
    :param label: The label to draw.
    :param color: The color of the box and label.
    :param txt_color: The color of the text.
    """
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[
            0
        ]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            lw / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def plot_bboxes(
    image: Any,
    boxes: Any,
    labels: List = [],
    score: bool = True,
    conf: Optional[float] = None,
) -> None:
    """Plot bounding boxes on image, with or without confidence score.

    :param image: The image to draw on.
    :param boxes: The bounding boxes to draw.
    The format is (x1, y1, x2, y2, class, score).
    :param labels: The labels to draw.
    :param score: Whether to draw the score.
    :param conf: The confidence threshold.
    """
    # plot each boxes
    for box in boxes:
        # add score in label if score=True
        if score:
            label = (
                labels[int(box[-1])] + " " + str(round(100 * float(box[-2]), 1)) + "%"
            )
        else:
            label = labels[int(box[-1])]
        # filter every box under conf threshold if conf threshold setted
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors(int(box[-1]))
            box_label(image, box, label, color)
    # show image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        cv2.imshow(image)  # if used in Python
    except Exception:
        from google.colab.patches import cv2_imshow

        cv2_imshow(image)  # if used in Colab


# Modified to keep logits
def non_max_suppression(
    prediction: Any,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: Any = (),
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    nm: Optional[int] = None,  # number of masks (optional)
) -> List[torch.Tensor]:
    """
    Perform non-maximum suppression (NMS) on a set of boxes,
    with support for masks and multiple labels per box.

    Arguments:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_boxes,
            num_classes + 4 + num_masks)
            containing the predicted boxes, classes, and masks.
            The tensor should be in the format output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will
            be filtered out. Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered
            out during NMS. Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None,
            all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes,
            and all classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists,
            where each inner list contains the apriori labels for a given image.
            The list should be in the format output by a dataloader,
            with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): (optional) The number of classes output by the model.
            Any indices after this will be considered masks.

    Returns:
        (List[torch.Tensor]): A list of length batch_size,
            where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"""Invalid Confidence threshold {conf_thres}, \
        valid values are between 0.0 and 1.0"""
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    if nm is None:
        nm = prediction.shape[1] - nc - 4
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x.transpose(0, -1)[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)
        # xywh2xyxy
        box = box_convert(
            box, "cxcywh", "xyxy"
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        if multi_label:
            i, j = (cls > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask, cls), 1)[
                conf.view(-1) > conf_thres
            ]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[
            x[:, 4].argsort(descending=True)[:max_nms]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return output


def denorm(in_images: Any) -> np.ndarray:
    """Denormalize images.

    :param in_images: Input images"""
    if isinstance(in_images, torch.Tensor):
        images = in_images.clone().float()
    else:
        raise ValueError("Input must be tensor")
    if images[0].max() <= 1:
        images *= 255  # de-normalise (optional)
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("Input tensor must have shape (n, 3, w, h).")
    return images.permute(0, 2, 3, 1).to(torch.int8).cpu().numpy().astype(np.uint8)


def process_batch_data(batch: Dict) -> Dict[int, Any]:
    """Convert batch data to a dictionary of image index to image data.
    Also denormalizes the images.

    :param batch: Batch data"""
    # Get the unique image indices and the count of bounding boxes per image
    unique_indices, counts = torch.unique(batch["batch_idx"], return_counts=True)
    # Split the bboxes tensor based on the counts of bounding boxes per image
    bboxes_split = torch.split(batch["bboxes"], counts.tolist())
    labels_split = torch.split(batch["cls"], counts.tolist())
    denormed_imgs = denorm(batch["img"])
    label_per_image: Dict[int, Any] = {
        i: {"img": img} for i, img in enumerate(denormed_imgs)
    }
    for idx, bboxes, labels in zip(unique_indices, bboxes_split, labels_split):
        i = int(idx.item())
        label_dict = label_per_image[i]
        label_dict["bboxes"] = bboxes
        label_dict["labels"] = labels.squeeze(-1)
    return label_per_image


# step names from ultralytics and their corresponding split
ultralytics_split_mapping = split_mapping = {
    Split.test: "test",
    Split.training: "train",
    Split.validation: "val",
}


def _read_config(path: str) -> Dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def temporary_cfg_for_val(cfg: Dict, split: Split, ds_path: str = "") -> str:
    """Creates a temporary config file with the split set to the given split.

    :param cfg_path: Path to the config file
    :param split: Split to set the config to"""
    cfg_copy = {**cfg}
    if not cfg.get(ultralytics_split_mapping[split]):
        return ""
    new_value = cfg.get(ultralytics_split_mapping[split])
    for csplit in ["train", "val", "test"]:
        cfg_copy[csplit] = new_value
    pre_path = Path(ds_path).resolve()
    # if pre_path is a file then get its parent
    if pre_path.is_file():
        pre_path = pre_path.parent

    if "path" in cfg_copy:
        cfg_path = Path(cfg_copy["path"])
        # check if cfg path is relative or absolute with pathlib
        # if relative then make it absolute
        # if absolute then leave it as it is
        # if cfg_path is relative then it is relative to ds_path
        # which can be relative to pwd
        if not cfg_path.is_absolute():
            cfg_path = pre_path / cfg_path
            # Check if the folder cfg_path exists
            if cfg_path.is_dir():
                cfg_copy["path"] = str(cfg_path)
            else:
                print("cfg_path does not exist")
    else:
        print("cfg_path not found in config")
        cfg_copy["path"] = str(pre_path)

    tmp = NamedTemporaryFile("w", delete=False, suffix=".yaml")
    yaml.safe_dump(cfg_copy, tmp)
    tmp.close()
    return tmp.name
