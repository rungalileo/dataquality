from typing import Dict
import sys
import os

from ultralytics import YOLO
import yaml
import dataquality as dq
from dataquality.integrations.ultralytics import watch
import glob

from dataquality.schemas.split import Split


def read_config(path: str) -> Dict:
    # Read the YAML file
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


def save_config(data: Dict, path: str) -> str:
    # Write the YAML file
    with open(path, "w") as file:
        yaml.safe_dump(data, file)
    return path


def get_modify_data(data: Dict, in_key: str, out_key: str) -> Dict:
    data_dict: Dict = {**data}
    # ["train", "val", "test"]
    assert in_key in ["train", "val", "test"]
    assert out_key in ["train", "val", "test"]
    data_dict[out_key] = data_dict[in_key]

    return data_dict


# yolo train data=coco128.yaml model=yolov8n.pt epochs=1 lr0=0.01


def main() -> None:
    original_cmd = sys.argv[1:]
    run_path_glob = "runs/detect/train*"
    files_start = glob.glob(run_path_glob)
    bash_run = " ".join(original_cmd)
    os.system(bash_run)
    print("Run complete")
    dataset_path = ""
    for arg in original_cmd:
        if arg.startswith("data="):
            dataset_path = arg[5:]

    project_name = input("Project name: ")
    run_name = input("Run name: ")
    files_end = glob.glob(run_path_glob)
    file_diff = set(files_end) - set(files_start)
    if not len(file_diff):
        return
    path = list(file_diff)[0]
    print("Loading trained model:", path)
    model = YOLO(path + "/weights/best.pt")
    cfg = read_config(dataset_path)
    dq.set_console_url("https://console.dev.rungalileo.io")
    dq.init(task_type="object_detection", project_name=project_name, run_name=run_name)
    split_mapping = {
        "training": Split.training,
        "train": Split.training,
        "val": Split.validation,
        "validation": Split.validation,
        "test": Split.test,
    }
    for split in ["train", "val", "test"]:
        dq.set_epoch(0)
        dq.set_split(split_mapping[split])
        cur_cfg: Dict = {**cfg}
        if cur_cfg.get(split):
            new_value = cur_cfg.get(split)
            for csplit in ["train", "val", "test"]:
                cur_cfg[csplit] = None
            cur_cfg["val"] = new_value
        save_config(cur_cfg, "tmp.yaml")
        model.val(data="tmp.yaml")
        watch(model)
    dq.finish()


if __name__ == "__main__":
    main()
