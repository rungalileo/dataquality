from typing import List


def extract_value(arguments: List[str], key: str) -> str:
    """Extract the value of the key from the arguments.

    :param arguments: The arguments of the command line.
    :param key: The key to extract.
    :return: The value of the key.
    """
    for arg in arguments:
        if arg.startswith(f"{key}="):
            return arg[len(key) + 1 :]
    raise ValueError(f"Could not find {key} in arguments.")


def get_dataset_path(arguments: list) -> str:
    """Extract the dataset path from the arguments of yolo.

    :param arguments: The arguments of ultralytics yolo.
    :return: The path to the dataset.
    """
    try:
        return extract_value(arguments, "data")
    except ValueError:
        raise ValueError(
            "Dataset path not found in arguments."
            "Pass it in the following format data=coco.yaml"
        )


def get_model_path(arguments: list) -> str:
    """Extract the dataset path from the arguments of yolo.

    :param arguments: The arguments of ultralytics yolo.
    :return: The path to the dataset.
    """
    try:
        return extract_value(arguments, "model")
    except ValueError:
        raise ValueError(
            "Model path not found in arguments."
            "Pass it in the following format model=./yolov8n.pt"
        )


def get_iou_thres(arguments: list) -> float:
    """Extract the iou threshold from the arguments of yolo.

    :param arguments: The arguments of ultralytics yolo.
    :return: The iou threshold.
    """
    try:
        return float(extract_value(arguments, "iou"))
    except ValueError:
        return 0.7


def get_conf_thres(arguments: list) -> float:
    """Extract the confidence threshold from the arguments of yolo.

    :param arguments: The arguments of ultralytics yolo.
    :return: The confidence threshold.
    """
    try:
        return float(extract_value(arguments, "conf"))
    except ValueError:
        return 0.25


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


def validate_args(arguments: list) -> None:
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
