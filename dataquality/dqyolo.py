import glob
import os
import sys
from typing import Dict, List

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


def validate(arguments: list) -> None:
    """Validate the arguments of the command line.

    :param arguments: The arguments of the command line.
    """
    data_valid = False
    model_valid = False
    for arg in arguments:
        if arg.startswith("data="):
            data_valid = True
        elif arg.startswith("model="):
            model_valid = True

    if not (data_valid and model_valid):
        raise ValueError(
            "You need to pass data and model.\n"
            "For example dqyolo data=coco128.yaml model=yolov8n.pt\n"
            "Or for training dqyolo train data=coco128.yaml model=yolov8n.pt "
            "epochs=1 lr0=0.01"
        )

    if len(arguments) < 2:
        raise ValueError(
            "\nYou need to pass at least two argument to the command line.\n"
            "For example dqyolo data=coco128.yaml model=yolov8n.pt\n"
            "Or for training dqyolo train data=coco128.yaml model=yolov8n.pt "
            "epochs=1 lr0=0.01"
        )


def main() -> None:
    """dqyolo is a wrapper around ultralytics yolo that will automatically
    run the model on the validation and test sets and provide data insights.
    """
    # 1. Take the original args to extract config path
    validate(sys.argv)

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
    try:
        bucket = cfg["bucket"]
    except KeyError:
        bucket = input(
            'Key "bucket" is missing in yaml, please enter path of files. '
            "For example s3://coco/coco128.\n"
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
    project_name = cfg.get("project_name", os.environ.get("DQ_PROJECT_NAME"))
    run_name = cfg.get("run_name", os.environ.get("DQ_RUN_NAME"))
    if not project_name:
        project_name = input("Project name: ")
    if not run_name:
        run_name = input("Run name: ")
    console_url = cfg.get("console_url", os.environ.get("GALILEO_CONSOLE_URL"))
    if console_url:
        dq.set_console_url("https://console.dev.rungalileo.io")
    dq.init(task_type="object_detection", project_name=project_name, run_name=run_name)

    labels = list(cfg.get("names", {}).values())
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
        tmp_cfg_path = temporary_cfg_for_val(cfg, split)
        if not tmp_cfg_path:
            continue
        relative_img_path = relative_img_paths[split]
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
