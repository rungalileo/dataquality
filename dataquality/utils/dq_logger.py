import logging
import os
from typing import Any, Optional, Tuple

from pydantic import UUID4
from requests import HTTPError

from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.core._config import ConfigData, config
from dataquality.utils.helpers import check_noop

DQ_LOG_FILE_HOME = f"{ConfigData.DEFAULT_GALILEO_CONFIG_DIR}/out"
DQ_LOG_FILE = "out.log"


class CustomSplitAdapter(logging.LoggerAdapter):
    """
    This adapter appends the split to the message, if found. Otherwise, "None"

    Adapted from https://docs.python.org/3/howto/logging-cookbook.html (CustomAdapter)
    """

    def process(self, msg: str, kwargs: Any) -> Tuple[str, Any]:
        split = kwargs.pop("split", self.extra["split"])
        epoch = kwargs.pop("epoch", self.extra["epoch"])
        if epoch is not None:
            return "[%s] [epoch:%s]: %s" % (split, str(epoch), msg), kwargs
        return "[%s]: %s" % (split, msg), kwargs


log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s"
)


def get_dq_logger() -> CustomSplitAdapter:
    """Returns the dq logger for the current run_id"""
    logger = logging.getLogger(str(config.current_run_id))
    logger.setLevel(os.environ.get("GALILEO_LOG_LEVEL", "INFO").upper())
    # Avoid adding multiple handlers if one already exists
    if not logger.handlers:
        handler = logging.FileHandler(dq_log_file_path())
        handler.setFormatter(log_formatter)
        logger.addHandler(handler)
    adapter = CustomSplitAdapter(logger, {"split": None, "epoch": None})
    return adapter


def dq_log_file_path(run_id: Optional[UUID4] = None) -> str:
    rid = run_id or config.current_run_id
    return f"{DQ_LOG_FILE_HOME}/{rid}/{DQ_LOG_FILE}"


def remove_dq_log_file(run_id: Optional[UUID4] = None) -> None:
    file_path = dq_log_file_path(run_id)
    if os.path.isfile(file_path):
        os.remove(file_path)


def dq_log_object_name(project_id: UUID4, run_id: UUID4) -> str:
    """Returns the minio/s3 object name"""
    return f"{project_id}/{run_id}/out/{DQ_LOG_FILE}"


@check_noop
def upload_dq_log_file() -> None:
    # For typing
    assert config.current_project_id and config.current_run_id
    obj_store = ObjectStore()
    obj_name = dq_log_object_name(config.current_project_id, config.current_run_id)
    file_path = dq_log_file_path()
    if os.path.isfile(file_path):
        obj_store.create_project_run_object(
            object_name=obj_name,
            file_path=file_path,
            content_type="text/plain",
            progress=False,
        )
        remove_dq_log_file()


@check_noop
def get_dq_log_file(
    project_name: Optional[str] = None, run_name: Optional[str] = None
) -> Optional[str]:
    pid, rid = ApiClient()._get_project_run_id(
        project_name=project_name, run_name=run_name
    )
    log_object = dq_log_object_name(pid, rid)
    log_file_path = dq_log_file_path(rid)
    log_file_dir = os.path.split(log_file_path)[0]
    os.makedirs(log_file_dir, exist_ok=True)
    obj_store = ObjectStore()
    try:
        obj_store.download_file(object_name=log_object, file_path=log_file_path)
        print(f"Your logfile has been written to {log_file_path}")
        return log_file_path
    except HTTPError:
        print("No log file found for run")
        return None
