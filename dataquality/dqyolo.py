import glob
import os
import sys
from typing import List

from ultralytics import YOLO
from ultralytics.yolo.utils import get_settings

import dataquality as dq
from dataquality.integrations.ultralytics import watch
from dataquality.schemas.split import Split
from dataquality.utils.ultralytics import (
    _read_config,
    temporary_cfg_for_val,
    ultralytics_split_mapping,
)

# http://images.cocodataset.org.s3.amazonaws.com/train2017/000000000009.jpg
# http://images.cocodataset.org.s3.amazonaws.com/test2017/000000000001.jpg
# example yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01
# python dq-cli.py yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01


def get_dataset_path(arguments: list) -> str:
    """Extract the dataset path from the arguments of yolo.

    :param arguments: The arguments of ultralytics yolo.
    :return: The path to the dataset.
    """
    for arg in arguments:
        if arg.startswith("data="):
            return arg[5:]
    raise ValueError(
        "Dataset path not found in arguments."
        "Pass it in the following format data=coco.yaml"
    )


def get_model_path(arguments: list) -> str:
    """Extract the dataset path from the arguments of yolo.

    :param arguments: The arguments of ultralytics yolo.
    :return: The path to the dataset.
    """
    for arg in arguments:
        if arg.startswith("model="):
            return arg[6:]
    raise ValueError(
        "Model path not found in arguments."
        "Pass it in the following format model=./yolov8n.pt"
    )


def find_last_run(files_start: List, files_end: List) -> str:
    """Find the path of the last run. This assumes that the path is the only
    thing that has changed.

    :param files_start: The list of files before the run.
    :param files_end: The list of files after the run.
    :return: The path to the last run.
    """
    file_diff = set(files_end) - set(files_start)
    if not len(file_diff):
        raise ValueError("Model path could not be found.")
    path = list(file_diff)[0]
    return path


def main() -> None:
    """dq-yolo is a wrapper around ultralytics yolo that will automatically
    run the model on the validation and test sets and provide data insights.
    """
    # 1. Take the original args to extract config path
    original_cmd = sys.argv[1:]
    runs_dir = get_settings().get("runs_dir") or input(
        "Enter runs dir default. For example home/runs"
    )
    run_path_glob = str(runs_dir) + "/detect/train*"
    files_start = glob.glob(run_path_glob)
    bash_run = " ".join(original_cmd)
    if not bash_run.startswith("yolo"):
        bash_run = "yolo " + bash_run
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

    dataset_path = get_dataset_path(original_cmd)
    print("Loading trained model:", model_path)
    # 5. Init galileo
    project_name = os.environ.get("GALILEO_PROJECT_NAME") or input("Project name: ")
    run_name = os.environ.get("GALILEO_RUN_NAME") or input("Run name: ")
    console_url = os.environ.get("GALILEO_CONSOLE_URL")
    dq.set_console_url(console_url)
    dq.init(task_type="object_detection", project_name=project_name, run_name=run_name)
    # 6. Run the model on the available splits (train/val/test)
    cfg = _read_config(dataset_path)
    bucket = cfg.get("bucket") or input(
        'Key "bucket" is missing in yaml, please enter path of files. '
        "For example s3://coco/coco128.\n"
        "bucket: "
    )

    # Labels in the YAML files could either be a dict or a list
    labels = cfg.get("names", {})
    labels = labels if isinstance(labels, list) else list(labels.values())

    # Check each file
    for split in [Split.training, Split.validation, Split.test]:
        # Create a temporary config file for the training set that changes
        # the dataset path to the validation set so we can log the results
        tmp_cfg_path = temporary_cfg_for_val(cfg, split)
        if not tmp_cfg_path:
            continue
        relative_img_path = cfg[f"bucket_{ultralytics_split_mapping[split]}"]
        model = YOLO(model_path)
        watch(
            model, bucket=bucket, relative_img_path=relative_img_path, labels=labels
        )  # This will automatically log the results to galileo
        dq.set_epoch(0)
        dq.set_split(split)
        model.val(data=tmp_cfg_path)
        # Remove the temporary config file after we are done
        os.remove(tmp_cfg_path)
    dq.finish()


if __name__ == "__main__":
    main()
