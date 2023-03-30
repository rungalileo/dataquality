import torch
import numpy as np
import ultralytics
from ultralytics import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.engine.validator import BaseValidator
from ultralytics.yolo.engine.predictor import BasePredictor
from typing import Any, Callable, List, Optional, Tuple, Union
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
from ultralytics.yolo.utils.ops import scale_boxes
from torchvision.ops.boxes import box_convert
from dataquality.exceptions import GalileoException
from dataquality.schemas.split import Split
import dataquality as dq
from dataquality.utils.ultralytics import get_batch_data, non_max_suppression

ultralytics.checks()

Coordinates = Union[Tuple, List]


def find_midpoint(
    box: Coordinates, shape: Coordinates, resized_shape: Coordinates
) -> Tuple[int, int, int, int]:
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
    # creates embeddings, features is a feature dict
    # with feature maps, boxes are xyxy format
    out = []
    for i in range(len(features)):
        embedding = features[i]
        box_ = find_midpoint(box, size, (embedding.shape[1], embedding.shape[2]))
        out.append(embedding[:, box_[1], box_[0]].reshape(-1))

    return torch.cat(out, dim=0)


def embedding_fn(features: List, boxes: Any, size: Any) -> torch.Tensor:
    # function to create all the embeddings
    embeddings = []
    for box in boxes:
        embeddings.append(create_embedding(features, box, size))
    return torch.stack(embeddings) if len(boxes) > 0 else torch.empty((0, 512))


class StoreHook:
    h: Any = None

    def on_finish(*args: Any, **kwargs: Any) -> None:
        pass

    def hook(self, model: Any, model_input: Any, model_output: Any) -> None:
        self.model = model
        self.model_input = model_input
        self.model_output = model_output
        self.on_finish(model_input, model_output)

    def store_hook(self, h: Any) -> Any:
        self.h = h


class BatchLogger:
    def __init__(self, old_function: Callable) -> None:
        self.old_function = old_function

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        self.batch = self.old_function(*args, **kwargs)
        return self.batch


class Callback:
    split: Optional[Split] = None

    def __init__(self, nms_fn: Optional[Callable] = None) -> None:
        self.step_embs = StoreHook()
        self.step_pred = StoreHook()
        self.step_pred.on_finish = self._after_pred_step  # type: ignore
        self.nms_fn = nms_fn
        self.hooked = False
        self.split = None

    def postprocess(self, batch: torch.Tensor) -> Any:
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
        if self.split not in ["training", "validation"]:
            return
        with torch.no_grad():
            # Do what do we need to convert here?
            preds = (
                self.step_pred.model_output
            )  # tuple([pred for pred in self.step_pred.model_output])
            logging_data = get_batch_data(self.bl.batch)
            if not self.nms_fn:
                raise Exception("NMS function not found")
            postprocess = (
                lambda x: x if self.split == "validation" else self.postprocess
            )
            preds = postprocess(preds)
            nms = self.nms_fn(preds)
            self.nms = nms
            # these were from franz not sure the point of model input shape
            # model_input_shape = self.step_pred.model_input[0][0].shape[-2:]
            # batch_input_shape = self.bl.batch["img"].shape[-2:]
            # FIX: this need to be some sort of image idx instead of just i
            batch = self.bl.batch

            for i in range(len(nms)):
                logging_data[i]["fn"] = batch["im_file"][i]
                ratio_pad = batch["ratio_pad"][i]
                shape = batch["ori_shape"][i]
                batch_img_shape = batch["img"][i].shape[1:]
                pred = nms[i].detach().cpu()
                logging_data[i]["bbox_pred"] = pred.float()[:, :4]
                features = [
                    self.step_embs.model_input[0][0][i],
                    self.step_embs.model_input[0][1][i],
                    self.step_embs.model_input[0][2][i],
                ]
                logging_data[i]["pred_embs"] = (
                    embedding_fn(features, pred, batch_img_shape).cpu().numpy()
                )
                logging_data[i]["bbox_pred_scaled"] = scale_boxes(
                    batch_img_shape,
                    logging_data[i]["bbox_pred"].clone(),
                    shape,
                    ratio_pad=ratio_pad,
                )
                logging_data[i]["probs"] = pred[:, 6:].cpu().numpy()
                # if there are no gt boxes then bboxes will not be in the logging data
                if "bboxes" in logging_data[i].keys():
                    # iterate on the ground truth boxes
                    bbox = logging_data[i]["bboxes"].clone()
                    height, width = batch_img_shape
                    tbox = box_convert(bbox, "cxcywh", "xyxy") * torch.tensor(
                        (width, height, width, height)
                    )

                    logging_data[i]["bbox_gold"] = (
                        scale_boxes(batch_img_shape, tbox, shape, ratio_pad=ratio_pad)
                        .cpu()
                        .numpy()
                    )
                    logging_data[i]["gt_embs"] = (
                        embedding_fn(
                            features, logging_data[i]["bbox_gold"], batch_img_shape
                        )
                        .cpu()
                        .numpy()
                    )
                    logging_data[i]["labels"] = logging_data[i]["labels"].cpu().numpy()

                else:
                    logging_data[i]["bbox_gold"] = np.array([])
                    logging_data[i]["gt_embs"] = np.array([])
                    logging_data[i]["labels"] = np.array([])
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
            pred_boxes.append(self.logging_data[i]["bbox_pred"])
            gold_boxes.append(self.logging_data[i]["bbox_gold"])
            labels.append(self.logging_data[i]["labels"])
            pred_embs.append(self.logging_data[i]["pred_embs"])
            gold_embs.append(self.logging_data[i]["gt_embs"])
            probs.append(self.logging_data[i]["probs"])
            ids.append(self.logging_data[i]["fn"])

        # dq.log_model_output()

    def register_hooks(self, model: Any) -> None:
        h = model.register_forward_hook(self.step_pred.hook)
        self.step_pred.store_hook(h)
        h = model.model[-1].register_forward_hook(self.step_embs.hook)
        self.step_embs.store_hook(h)
        self.hooked = True

    def on_train_start(self, trainer: BaseTrainer) -> None:
        self.split = Split.training
        self.trainer = trainer
        self.register_hooks(trainer.model)
        self.bl = BatchLogger(trainer.preprocess_batch)
        trainer.preprocess_batch = self.bl

    def on_train_end(self, trainer: BaseTrainer) -> None:
        trainer.preprocess_batch = self.bl.old_function

    # -- Validator callbacks --
    def on_val_start(self, validator: BaseValidator) -> None:
        print("on val start")

    def on_val_batch_start(self, validator: BaseValidator) -> None:
        self.split = Split.validation
        print("val batch start")
        self.validator = validator
        if not self.hooked:
            self.register_hooks(validator.model.model)
            self.bl = BatchLogger(validator.preprocess)
            validator.preprocess = self.bl
            self.hooked = True

    # -- Predictor callbacks --
    def on_predict_start(self, predictor: BasePredictor) -> None:
        self.split = Split.inference
        self.predictor = predictor
        if not self.hooked:
            self.register_hooks(predictor.model.model)
            # Not implemnted self.bl = BatchLogger(lambda x: x)

    def on_predict_batch_start(predictor) -> None:
        pass

    def on_predict_batch_end(self, predictor: BasePredictor) -> None:
        print("on_predict_batch_end")
        # TODO: self.bl.batch = predictor.batch
        self._after_pred_step()

    def on_predict_postprocess_end(self, predictor: BasePredictor) -> None:
        pass

    def on_predict_end(self, predictor: BasePredictor) -> None:
        print("on_predict_end")


def add_callback(model: YOLO, cb: Callback) -> None:
    model.add_callback("on_train_start", cb.on_train_start)
    model.add_callback("on_train_end", cb.on_train_end)
    model.add_callback("on_predict_start", cb.on_predict_start)
    model.add_callback("on_predict_end", cb.on_predict_end)
    model.add_callback("on_predict_batch_end", cb.on_predict_batch_end)
    model.add_callback("on_val_start", cb.on_val_start)
    model.add_callback("on_val_batch_start", cb.on_val_batch_start)


def watch(model: YOLO) -> None:
    assert dq.config.task_type == "object_detection", GalileoException(
        "dq client must be initialized. " "For example: dq.init('text_classification')"
    )
    cb = Callback(nms_fn=non_max_suppression)
    add_callback(model, cb)
