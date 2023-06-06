import glob
import os
import sys
from typing import Dict

from ultralytics import YOLO
from ultralytics.yolo.utils import get_settings

import dataquality as dq
from dataquality.integrations.ultralytics import watch
from dataquality.schemas.split import Split
from dataquality.utils.dqyolo import (
    find_last_run,
    get_conf_thres,
    get_dataset_path,
    get_iou_thres,
    get_model_path,
    validate_args,
)
from dataquality.utils.ultralytics import (
    _read_config,
    temporary_cfg_for_val,
    ultralytics_split_mapping,
)

# http://images.cocodataset.org.s3.amazonaws.com/train2017/000000000009.jpg
# http://images.cocodataset.org.s3.amazonaws.com/test2017/000000000001.jpg
# example yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01
# python dq-cli.py yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01


def main() -> None:
    """dqyolo is a wrapper around ultralytics yolo that will automatically
    run the model on the validation and test sets and provide data insights.
    """
    # 1. Take the original args to extract config path

    validate_args(sys.argv)
    original_cmd = sys.argv[1:]
    runs_dir = get_settings().get("runs_dir") or input(
        "Enter runs dir default. For example home/runs"
    )
    run_path_glob = str(runs_dir) + "/detect/train*"
    files_start = glob.glob(run_path_glob)
    bash_run = " ".join(original_cmd)
    if not bash_run.startswith("yolo"):
        bash_run = "yolo " + bash_run

    dataset_path = get_dataset_path(original_cmd)
    cfg = _read_config(dataset_path)
    bucket = cfg.get("bucket") or input(
        'Key "bucket" is missing in yaml, please enter path of files. '
        "For example s3://coco/coco128\n"
        "bucket: "
    )
    relative_img_paths: Dict[Split, str] = {}
    # Check each file
    for split in [Split.training, Split.validation, Split.test]:
        # Create a temporary config file for the training set that changes
        # the dataset path to the validation set so we can log the results
        tmp_cfg_path = cfg.get(ultralytics_split_mapping[split])
        if not tmp_cfg_path:
            continue
        relative_img_paths[split] = cfg[f"bucket_{ultralytics_split_mapping[split]}"]
    if not len(relative_img_paths):
        raise ValueError("No dataset paths found in config file.")

    # 2. Init galileo
    project_name = os.environ.get("GALILEO_PROJECT_NAME") or input("Project name: ")
    run_name = os.environ.get("GALILEO_RUN_NAME") or input("Run name: ")
    console_url = cfg.get("console_url", os.environ.get("GALILEO_CONSOLE_URL"))
    if console_url:
        dq.set_console_url(console_url)
    dq.init(task_type="object_detection", project_name=project_name, run_name=run_name)

    # Labels in the YAML files could either be a dict or a list
    labels = cfg.get("names", {})
    labels = labels if isinstance(labels, list) else list(labels.values())
    if not len(labels):
        raise ValueError("Labels not found in config file.")

    if " train " in bash_run:
        # 2. Run the original command
        os.system(bash_run)
        # 3. Once training is complete we can eval the data with galileo logging
        print("Run complete")
        # 4. Find the model path and load it
        files_end = glob.glob(run_path_glob)
        model_path = find_last_run(files_start, files_end)
        model_path = model_path + "/weights/best.pt"
    else:
        model_path = get_model_path(original_cmd)

    print("Loading trained model:", model_path)

    # Check each file
    for split in [Split.training, Split.validation, Split.test]:
        # Create a temporary config file for the training set that changes
        # the dataset path to the validation set so we can log the results
        tmp_cfg_path = temporary_cfg_for_val(cfg, split, dataset_path)
        if not tmp_cfg_path:
            continue
        relative_img_path = relative_img_paths[split]
        model = YOLO(model_path)
        watch(
            model,
            bucket=bucket,
            relative_img_path=relative_img_path,
            labels=labels,
            iou_thresh=get_iou_thres(original_cmd),
            conf_thresh=get_conf_thres(original_cmd),
        )  # This will automatically log the results to galileo
        dq.set_epoch(0)
        dq.set_split(split)
        model.val(data=tmp_cfg_path)
        # Remove the temporary config file after we are done
        os.remove(tmp_cfg_path)
    dq.finish()


if __name__ == "__main__":
    main()
