import glob
import os
import sys
from typing import List

from ultralytics import YOLO

import dataquality as dq
from dataquality.integrations.ultralytics import watch
from dataquality.schemas.split import Split
from dataquality.utils.ultralytics import temporary_cfg_for_val

# example yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01
# python dq-cli.py yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01


def get_dataset_path(arguments: list) -> str:
    for arg in arguments:
        if arg.startswith("data="):
            return arg[5:]
    raise ValueError("Dataset path not found in arguments.")


def find_last_run(files_start: List, files_end: List) -> str:
    file_diff = set(files_end) - set(files_start)
    if not len(file_diff):
        raise ValueError("Model path could not be found.")
    path = list(file_diff)[0]
    return path


def main() -> None:
    # Take the original args to extract config and then run the original comand
    original_cmd = sys.argv[1:]
    run_path_glob = "runs/detect/train*"
    files_start = glob.glob(run_path_glob)
    bash_run = " ".join(original_cmd)
    if not bash_run.startswith("yolo"):
        bash_run = "yolo " + bash_run
    os.system(bash_run)
    # Once training is complete we can init galileo
    print("Run complete")
    dataset_path = get_dataset_path(original_cmd)
    files_end = glob.glob(run_path_glob)
    model_path = find_last_run(files_start, files_end)
    print("Loading trained model:", model_path)

    # Init galileo
    project_name = input("Project name: ")
    run_name = input("Run name: ")
    dq.set_console_url("https://console.dev.rungalileo.io")
    dq.init(task_type="object_detection", project_name=project_name, run_name=run_name)

    # Check each file
    for split in [Split.training, Split.validation, Split.test]:
        tmp_cfg_path = temporary_cfg_for_val(dataset_path, split)
        if not tmp_cfg_path:
            continue
        model = YOLO(model_path + "/weights/best.pt")
        watch(model)
        dq.set_epoch(0)
        dq.set_split(split)
        model.val(data=tmp_cfg_path)
        os.remove(tmp_cfg_path)
    dq.finish()


if __name__ == "__main__":
    main()
