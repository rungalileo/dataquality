import torch
import numpy as np
import ultralytics

from ultralytics.yolo.utils.tal import dist2bbox, make_anchors
from ultralytics.yolo.utils.ops import scale_boxes
from torchvision.ops.boxes import box_convert
ultralytics.checks()

def denorm(in_images):
    if isinstance(in_images, torch.Tensor):
        images = in_images.clone().float()
    else:
        raise ValueError("Input must be tensor")
    if images[0].max() <= 1:
        images *= 255  # de-normalise (optional)
    if images.dim() != 4 or images.size(1) != 3:
        raise ValueError("Input tensor must have shape (n, 3, w, h).")
    return images.permute(0, 2, 3, 1).to(torch.int8).cpu().numpy().astype(np.uint8)

def get_batch_data(batch):
    # Get the unique image indices and the count of bounding boxes per image
    unique_indices, counts = torch.unique(batch["batch_idx"], return_counts=True)
    # Split the bboxes tensor based on the counts of bounding boxes per image
    bboxes_split = torch.split(batch["bboxes"], counts.tolist())
    labels_split = torch.split(batch["cls"], counts.tolist())
    denormed_imgs = denorm(batch['img'])
    label_per_image = {i: {"img":img} for i,img in enumerate(denormed_imgs)}

    for idx, bboxes, labels in zip(unique_indices, bboxes_split, labels_split):
        i = int(idx.item())
        label_dict = label_per_image[i] 
        label_dict["bboxes"] = bboxes
        label_dict["labels"] = labels.squeeze(-1)
    return label_per_image

def convert_xywh_to_xyxy(bboxes):
  # converts xywh boxes to xyxy can be in either integer coords or 0-1
  if not isinstance(bboxes, torch.Tensor):
    return box_convert(torch.Tensor(bboxes),  "cxcywh","xyxy")
  return box_convert(bboxes,  "cxcywh","xyxy")

def find_midpoint(box, shape, resized_shape):
    # function to find the midpoint based off a box in xyxy format
    x1, y1, x2, y2 = box[:4]
    x1 = int(x1 * resized_shape[1] / shape[1])
    x2 = int(x2 * resized_shape[1] / shape[1])
    y1 = int(y1 * resized_shape[0] / shape[0])
    y2 = int(y2 * resized_shape[0] / shape[0])
    return int((x1 + x2)/2), int((y1 + y2)/2) , int((x1 + x2)/2), int((y1 + y2)/2)

def create_embedding(features, box, size=(640, 640)):
    # creates embeddings, features is a feature dict with feature maps, boxes are xyxy format
    out = []
    for i in range(len(features)):
        embedding = features[i]
        box_ = find_midpoint(box ,size, (embedding.shape[1], embedding.shape[2]))
        out.append(embedding[:,  box_[1], box_[0]].reshape(-1))

    return torch.cat(out, dim=0)

def embedding_fn(features, boxes, size):
    # function to create all the embeddings
    embeddings = []
    for box in boxes:
        embeddings.append(create_embedding(features, box, size))
    return torch.stack(embeddings) if len(boxes) > 0 else torch.empty((0, 512))
    

class StoreHook:
    def on_finish(*args,**kwargs):
        pass

    def hook(self,model, model_input, model_output):
        self.model = model
        self.model_input = model_input
        self.model_output = model_output
        self.on_finish(model_input,model_output)

class BatchLogger:
    def __init__(self, old_function):
        self.old_function = old_function

    def __call__(self, *args,**kwargs):
        self.batch = self.old_function(*args,**kwargs)
        return self.batch

class Callback:
    def __init__(self,nms_fn=None):
        self.step_embs = StoreHook()
        self.step_pred = StoreHook()
        self.step_pred.on_finish = self._after_pred_step
        self.nms_fn = nms_fn
        self.hooked = False
        self.split = None

    def postprocess(self, x):
        ref_model = self.step_embs.model
        shape = x[0].shape  # height, width
        if ref_model.dynamic or ref_model.shape != shape:
            ref_model.anchors, ref_model.strides = (x.transpose(0, 1) for x in make_anchors(x, ref_model.stride, 0.5))
            ref_model.shape = shape

        x_cat = torch.cat([xi.view(shape[0], ref_model.no, -1) for xi in x], 2)
        if ref_model.export and ref_model.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :ref_model.reg_max * 4]
            cls = x_cat[:, ref_model.reg_max * 4:]
        else:
            box, cls = x_cat.split((ref_model.reg_max * 4, ref_model.nc), 1)
        dbox = dist2bbox(ref_model.dfl(box), ref_model.anchors.unsqueeze(0), xywh=True, dim=1) * ref_model.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if ref_model.export else (y, x)

    def _after_pred_step(self, *args, **kwargs):
        if self.split not in ["training","validation"]:
          return
        with torch.no_grad():
            # Do what do we need to convert here?
            preds =  self.step_pred.model_output #tuple([pred for pred in self.step_pred.model_output])
            logging_data = get_batch_data(self.bl.batch)
            if not self.nms_fn:
              raise 'NMS function not found'
            postprocess =  lambda x: x if self.split == "validation" else self.postprocess
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
                ratio_pad = batch['ratio_pad'][i]
                shape = batch['ori_shape'][i]
                batch_img_shape = batch['img'][i].shape[1:]
                pred = nms[i].detach().cpu()
                logging_data[i]["bbox_pred"] = pred.float()[:, :4]
                features = [
                    self.step_embs.model_input[0][0][i],
                    self.step_embs.model_input[0][1][i],
                    self.step_embs.model_input[0][2][i]
                    ]
                logging_data[i]["pred_embs"] = embedding_fn(features,
                    pred,
                    batch_img_shape).cpu().numpy()
                logging_data[i]["bbox_pred_scaled"] = scale_boxes(
                    batch_img_shape,
                    logging_data[i]["bbox_pred"].clone(),
                    shape,
                    ratio_pad=ratio_pad
                    )
                logging_data[i]['probs'] = pred[:, 6:].cpu().numpy()
                # if there are no gt boxes then bboxes will not be in the logging data
                if 'bboxes' in logging_data[i].keys():
                    # iterate on the ground truth boxes
                    bbox = logging_data[i]['bboxes'].clone()
                    height, width = batch_img_shape
                    tbox = xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height))
                    
                    logging_data[i]['bbox_gold'] = scale_boxes(
                        batch_img_shape,
                        tbox,
                        shape,
                        ratio_pad=ratio_pad
                        ).cpu().numpy()
                    logging_data[i]['gt_embs'] = embedding_fn(
                        features
                        , 
                                                    logging_data[i]['bbox_gold'],
                                                    batch_img_shape).cpu().numpy()
                    logging_data[i]['labels'] = logging_data[i]['labels'].cpu().numpy()
                    
                else:
                    logging_data[i]['bbox_gold'] = np.array([])
                    logging_data[i]['gt_embs'] = np.array([])
                    logging_data[i]['labels'] = np.array([])
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
            pred_boxes.append(self.logging_data[i]['bbox_pred'])
            gold_boxes.append(self.logging_data[i]['bbox_gold'])
            labels.append(self.logging_data[i]['labels'])
            #pred_embs.append(self.logging_data[i]['pred_embs'])
            #gold_embs.append(self.logging_data[i]['gt_embs'])
            probs.append(self.logging_data[i]['probs'])
            ids.append(self.logging_data[i]['fn'])

        # dq.log_model_output()
    
    def register_hooks(self, model):
        self.step_embs.h = model.register_forward_hook(self.step_pred.hook)
        self.step_embs.h = model.model[-1].register_forward_hook(self.step_embs.hook)
        self.hooked = True

    def on_train_start(self, trainer):
        self.split = "training"
        self.trainer = trainer
        self.register_hooks(trainer.model)
        self.bl = BatchLogger(trainer.preprocess_batch)
        trainer.preprocess_batch = self.bl

    def on_train_end(self, trainer):
        trainer.preprocess_batch = self.bl.old_function

    # Validator callbacks --------------------------------------------------------------------------------------------------
    def on_val_start(self, validator):
        print("on val start")

    def on_val_batch_start(self, validator):
        self.split = "validation"
        print("val batch start")
        self.validator = validator
        if not self.hooked:
          self.register_hooks(validator.model.model)
          self.bl = BatchLogger(validator.preprocess)
          validator.preprocess = self.bl
          self.hooked = True


    def on_val_batch_end(self, validator):
        print("val batch end")


    def on_val_end(self, validator):
        pass
    # Predictor callbacks --------------------------------------------------------------------------------------------------
    def on_predict_start(self, predictor):
        self.split = "inference"
        self.predictor = predictor
        if not self.hooked:
          self.register_hooks(predictor.model.model)
          # Not implemnted self.bl = BatchLogger(lambda x: x)

    def on_predict_batch_start(predictor):
        pass


    def on_predict_batch_end(self, predictor):
        print("on_predict_batch_end")
        # TODO: self.bl.batch = predictor.batch
        self._after_pred_step()


    def on_predict_postprocess_end(self, predictor):
        pass

    def on_predict_end(self, predictor):
        print("on_predict_end")

 

def add_callback(model, cb):
    model.add_callback("on_train_start",cb.on_train_start)
    model.add_callback("on_train_end",cb.on_train_end)
    model.add_callback("on_predict_start",cb.on_predict_start)
    model.add_callback("on_predict_end",cb.on_predict_end)
    model.add_callback("on_predict_batch_end",cb.on_predict_batch_end)
    model.add_callback("on_val_start",cb.on_val_start)
    model.add_callback("on_val_batch_start",cb.on_val_batch_start)
    model.add_callback("on_val_batch_end",cb.on_val_batch_end)
