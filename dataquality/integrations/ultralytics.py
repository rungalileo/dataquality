from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import torch
import ultralytics
from torchvision.ops.boxes import box_convert
from ultralytics import YOLO
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.utils.ops import scale_boxes
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

import dataquality as dq
from dataquality import get_data_logger
from dataquality.exceptions import GalileoException
from dataquality.loggers.data_logger.object_detection import (
    ObjectDetectionDataLogger,
    ODCols,
)
from dataquality.schemas.split import Split
from dataquality.schemas.task_type import TaskType
from dataquality.utils.dqyolo import CONF_DEFAULT, IOU_DEFAULT
from dataquality.utils.ultralytics import non_max_suppression, process_batch_data

ultralytics.checks()

Coordinates = Union[Tuple, List]


def find_midpoint(
    box: Coordinates, shape: Coordinates, resized_shape: Coordinates
) -> Tuple[int, int, int, int]:
    """Finds the midpoint of a box in xyxy format

    :param box: box in xyxy format
    :param shape: shape of the image
    :param resized_shape: shape of the resized image

    :return: midpoint of the box
    """
    # function to find the midpoint
    # based off a box in xyxy format
    x1, y1, x2, y2 = box[:4]
    x1 = int(x1 * resized_shape[1] / shape[1])
    x2 = int(x2 * resized_shape[1] / shape[1])
    y1 = int(y1 * resized_shape[0] / shape[0])
    y2 = int(y2 * resized_shape[0] / shape[0])
    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2),
        int((x1 + x2) / 2),
        int((y1 + y2) / 2),
    )


def create_embedding(
    features: List, box: List, size: Tuple[int, int] = (640, 640)
) -> torch.Tensor:
    """Creates an embedding from a feature map

    :param features: feature map
    :param box: box in xyxy format
    :param size: size of the image

    :return: embedding
    """
    # creates embeddings, features is a feature dict
    # with feature maps, boxes are xyxy format
    out = []
    for i in range(len(features)):
        embedding = features[i]
        box_ = find_midpoint(box, size, (embedding.shape[1], embedding.shape[2]))
        out.append(embedding[:, box_[1], box_[0]].reshape(-1))

    return torch.cat(out, dim=0)


def embedding_fn(features: List, boxes: Any, size: Any) -> torch.Tensor:
    """Creates embeddings for all boxes

    :param features: feature map
    :param boxes: boxes in xyxy format
    :param size: size of the image

    :return: embeddings
    """

    # function to create all the embeddings
    embeddings = []
    for box in boxes:
        embeddings.append(create_embedding(features, box, size))
    return torch.stack(embeddings) if len(boxes) > 0 else torch.empty((0, 512))


class StoreHook:
    """Generic Hook class to store model input and output"""

    h: Any = None

    def __init__(self, on_finish_func: Optional[Callable] = None) -> None:
        """Initializes the hook

        :param on_finish_func: function to be called when the hook is finished
        """
        if on_finish_func is not None:
            self.on_finish = on_finish_func

    def hook(self, model: Any, model_input: Any, model_output: Any) -> None:
        """Hook function to store model input and output

        :param model: model
        :param model_input: model input
        :param model_output: model output
        """

        self.model = model
        self.model_input = model_input
        self.model_output = model_output
        if hasattr(self, "on_finish"):
            self.on_finish()

    def store_hook(self, h: Any) -> None:
        """Stores hook for later removal

        :param h: hook
        """

        self.h = h


class BatchLogger:
    """Batch Logger class to store batches for later logging"""

    def __init__(self, old_function: Callable) -> None:
        """Store the batch by overwriting the given method

        :param old_function: method that is wrapped"""
        self.old_function = old_function

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Stores the batch and returns it"""
        self.batch = self.old_function(*args, **kwargs)
        return self.batch


class Callback:
    """Callback class that is used to log batches, embeddings and predictions"""

    split: Optional[Split]
    file_map: Dict

    def __init__(
        self,
        nms_fn: Optional[Callable] = None,
        bucket: str = "",
        relative_img_path: str = "",
        labels: List = [],
        iou_thresh: float = IOU_DEFAULT,
        conf_thresh: float = CONF_DEFAULT,
    ) -> None:
        """Initializes the callback

        :param nms_fn: non-maximum suppression function
        """
        self.step_embs = StoreHook()  # To log embeddings
        self.step_pred = StoreHook(self._after_pred_step)  # To log predictions
        self.nms_fn = nms_fn
        self.hooked = False
        self.split = None
        # bucket needs to start with // and not end with /
        bucket = bucket if bucket[-1] != "/" else bucket[:-1]
        assert "://" in bucket, "bucket needs to start with s3:// or gs://"
        assert bucket.count("/") == 2, "bucket should only contain 2 slashes"
        self.bucket = bucket

        self.relative_img_path = relative_img_path
        self.labels = labels
        self.file_map = {}  # maps file names to ids
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

    def postprocess(self, batch: torch.Tensor) -> Any:
        """Postprocesses the batch for a training step. Taken from ultralytics.
        Might be removed in the future.

        :param batch: batch to be postprocessed
        """
        ref_model = self.step_embs.model
        shape = batch[0].shape  # height, width
        if ref_model.dynamic or ref_model.shape != shape:
            ref_model.anchors, ref_model.strides = (
                x.transpose(0, 1) for x in make_anchors(batch, ref_model.stride, 0.5)
            )
            ref_model.shape = shape

        x_cat = torch.cat([xi.view(shape[0], ref_model.no, -1) for xi in batch], 2)
        if ref_model.export and ref_model.format in (
            "saved_model",
            "pb",
            "tflite",
            "edgetpu",
            "tfjs",
        ):  # avoid TF FlexSplitV ops
            box = x_cat[:, : ref_model.reg_max * 4]
            cls = x_cat[:, ref_model.reg_max * 4 :]
        else:
            box, cls = x_cat.split((ref_model.reg_max * 4, ref_model.nc), 1)
        dbox = (
            dist2bbox(
                ref_model.dfl(box), ref_model.anchors.unsqueeze(0), xywh=True, dim=1
            )
            * ref_model.strides
        )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if ref_model.export else (y, batch)

    def _after_pred_step(self, *args: Any, **kwargs: Any) -> None:
        """Called after a prediction step. Logs the predictions and embeddings"""
        if self.split not in [Split.validation]:
            return
        with torch.no_grad():
            preds = self.step_pred.model_output
            logging_data = process_batch_data(self.bl.batch)
            if not self.nms_fn:
                raise Exception("NMS function not found")
            postprocess = (
                lambda x: x if self.split == Split.validation else self.postprocess
            )
            preds = postprocess(preds)
            nms = self.nms_fn(
                preds, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
            )
            self.nms = nms

            batch = self.bl.batch

            for i in range(len(nms)):
                logging_data[i]["fn"] = batch["im_file"][i]
                logging_data[i]["id"] = self.file_map[batch["im_file"][i]]
                ratio_pad = batch["ratio_pad"][i]
                shape = batch["ori_shape"][i]
                batch_img_shape = batch["img"][i].shape[1:]
                pred = nms[i].detach().cpu()
                bbox = pred[:, :4].float()
                features = [
                    self.step_embs.model_input[0][0][i],
                    self.step_embs.model_input[0][1][i],
                    self.step_embs.model_input[0][2][i],
                ]
                logging_data[i]["pred_embs"] = embedding_fn(
                    features, bbox, batch_img_shape
                )
                # Box rescaling taken from ultralytics source code
                logging_data[i]["bbox_pred"] = scale_boxes(
                    batch_img_shape, bbox, shape, ratio_pad=ratio_pad
                )
                logging_data[i]["probs"] = pred[:, 6:].numpy()
                # if there are no gt boxes then bboxes will not be in the logging data
                if "bboxes" in logging_data[i].keys():
                    # iterate on the ground truth boxes (given in cxcywh format)
                    bbox = logging_data[i]["bboxes"]
                    height, width = batch_img_shape
                    tbox = box_convert(bbox, "cxcywh", "xyxy") * torch.tensor(
                        (width, height, width, height), device=bbox.device
                    )
                    # Box rescaling taken from ultralytics source code
                    logging_data[i]["gt_embs"] = embedding_fn(
                        features, tbox, batch_img_shape
                    )
                    logging_data[i]["bbox_gold"] = scale_boxes(
                        batch_img_shape, tbox, shape, ratio_pad=ratio_pad
                    )
                else:
                    logging_data[i]["bbox_gold"] = torch.Tensor([])
                    logging_data[i]["gt_embs"] = torch.Tensor([])
                    logging_data[i]["labels"] = torch.Tensor([])
            self.logging_data = logging_data

        # create what I feed to the dataquality client
        pred_boxes = []
        gold_boxes = []
        labels = []
        pred_embs = []
        gold_embs = []
        probs = []
        ids = []
        for i in range(len(self.logging_data)):
            pred_boxes.append(self.logging_data[i]["bbox_pred"].cpu().numpy())
            gold_boxes.append(self.logging_data[i]["bbox_gold"].cpu().numpy())
            labels.append(self.logging_data[i]["labels"].cpu().numpy())
            pred_embs.append(self.logging_data[i]["pred_embs"].cpu().numpy())
            gold_embs.append(self.logging_data[i]["gt_embs"].cpu().numpy())
            probs.append(self.logging_data[i]["probs"])
            ids.append(self.logging_data[i]["id"])

        # TODO: replace properly
        split = get_data_logger().logger_config.cur_split
        assert split
        dq.core.log.log_od_model_outputs(
            pred_boxes=pred_boxes,
            gold_boxes=gold_boxes,
            labels=labels,
            pred_embs=pred_embs,
            gold_embs=gold_embs,
            image_size=None,
            probs=probs,
            logits=probs,
            ids=ids,
            split=split,
            epoch=0,
        )

    def register_hooks(self, model: Any) -> None:
        """Register hooks to the model to log predictions and embeddings

        :param model: the model to hook"""
        h = model.register_forward_hook(self.step_pred.hook)
        self.step_pred.store_hook(h)
        # Take last layer for the embeddings of model.model
        h = model.model[-1].register_forward_hook(self.step_embs.hook)
        self.step_embs.store_hook(h)
        self.hooked = True

    def init_run(self) -> None:
        """Initialize the run"""
        if not len(self.labels):
            labels = list(self.validator.data["names"].values())
        else:
            labels = self.labels
        dq.set_labels_for_run(labels)
        ds = self.convert_dataset(self.validator.dataloader.dataset)
        data_logger = get_data_logger()
        assert isinstance(data_logger, ObjectDetectionDataLogger), (
            "This method is only supported for image tasks. "
            "Please use dq.log_dataset for text tasks."
        )
        split = data_logger.logger_config.cur_split
        assert split
        data_logger.log_dataset(ds, split=split)

    def convert_dataset(self, dataset: Any) -> List:
        """Convert the dataset to the format expected by the dataquality client"""
        assert len(dataset) > 0
        processed_dataset = []
        for i, image in enumerate(dataset):
            self.file_map[image["im_file"]] = i
            batch_img_shape = image["img"].shape[-2:]
            shape = image["ori_shape"]
            image_height, image_width = shape
            bbox = image["bboxes"].clone()
            # Batch width and height. The max width/height for this batch
            bch_height, bch_width = batch_img_shape
            tbox = box_convert(bbox, "cxcywh", "xyxy") * torch.tensor(
                (bch_width, bch_height, bch_width, bch_height), device=bbox.device
            )
            ratio_pad = image["ratio_pad"]
            bbox_gold = scale_boxes(batch_img_shape, tbox, shape, ratio_pad=ratio_pad)
            file_name = Path(image["im_file"]).name
            if self.relative_img_path[0] == "/":
                self.relative_img_path = self.relative_img_path[1:]
            if self.relative_img_path[-1] == "/":
                self.relative_img_path = self.relative_img_path[:-1]

            file_path = (
                self.bucket + "/" + urljoin(self.relative_img_path + "/", file_name)
            )
            processed_dataset.append(
                {
                    ODCols.id: i,
                    ODCols.image: str(file_path),
                    ODCols.bbox: bbox_gold,
                    ODCols.gold_cls: image["cls"],
                    ODCols.width: image_width,
                    ODCols.height: image_height,
                }
            )
        return processed_dataset

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Register hooks and preprocess batch function on train start

        :param trainer: the trainer
        """
        self.split = Split.training
        self.trainer = trainer
        self.register_hooks(trainer.model)
        self.bl = BatchLogger(trainer.preprocess_batch)
        trainer.preprocess_batch = self.bl

    def on_train_end(self, trainer: BaseTrainer) -> None:
        """Restore preprocess batch function on train end

        :param trainer: the trainer"""
        trainer.preprocess_batch = self.bl.old_function

    # -- Validator callbacks --

    def on_val_batch_start(self, validator: BaseValidator) -> None:
        """Register hooks and preprocess batch function on validation start

        :param validator: the validator"""
        self.split = Split.validation
        self.validator = validator
        if not self.hooked:
            self.register_hooks(validator.model.model)
            self.bl = BatchLogger(validator.preprocess)
            validator.preprocess = self.bl
            self.hooked = True
            self.init_run()

    # -- Predictor callbacks --
    def on_predict_start(self, predictor: BasePredictor) -> None:
        """Register hooks on prediction start
        Note: prediction is not perfect as the model is not in eval mode.
        May be removed

        :param predictor: the predictor"""
        self.split = Split.inference
        self.predictor = predictor
        if not self.hooked:
            self.register_hooks(predictor.model.model)
            # Not implemnted self.bl = BatchLogger(lambda x: x)

    def on_predict_batch_end(self, predictor: BasePredictor) -> None:
        """Log predictions and embeddings on prediction batch end.
        Not functional yet
        """
        # TODO: make inference work: self.bl.batch = predictor.batch
        self._after_pred_step()


def add_callback(model: YOLO, cb: Callback) -> None:
    """Add the callback to the model

    :param model: the model
    :param cb: callback cls"""
    model.add_callback("on_train_start", cb.on_train_start)
    model.add_callback("on_train_end", cb.on_train_end)
    model.add_callback("on_predict_start", cb.on_predict_start)
    model.add_callback("on_predict_batch_end", cb.on_predict_batch_end)
    model.add_callback("on_val_batch_start", cb.on_val_batch_start)


def watch(
    model: YOLO,
    bucket: str,
    relative_img_path: str,
    labels: List,
    iou_thresh: float = IOU_DEFAULT,
    conf_thresh: float = CONF_DEFAULT,
) -> None:
    """Watch the model for predictions and embeddings logging.

    :param model: the model to watch"""
    assert dq.config.task_type == TaskType.object_detection, GalileoException(
        "dq client must be initialized for Object Detection. For example: "
        "dq.init('object_detection')"
    )
    cb = Callback(
        nms_fn=non_max_suppression,
        bucket=bucket,
        relative_img_path=relative_img_path,
        labels=labels,
        iou_thresh=iou_thresh,
        conf_thresh=conf_thresh,
    )
    add_callback(model, cb)
