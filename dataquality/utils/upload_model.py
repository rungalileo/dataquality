import os
import tarfile
import tempfile
from typing import Any, Dict, Tuple

import requests
from transformers import PreTrainedTokenizer

from dataquality.clients.api import ApiClient
from dataquality.core._config import config
from dataquality.schemas.model import ModelUploadType

api_client = ApiClient()


def create_tar_archive(source_folder: str, output_filename: str) -> None:
    """
    Creates a tar archive from a folder / model.
    :param source_folder: The folder to archive.
    :param output_filename: The name of the output tar file.
    """
    with tarfile.open(output_filename, "w") as archive:
        for item in os.listdir(source_folder):
            full_path = os.path.join(source_folder, item)
            archive.add(full_path, arcname=item)


def upload_to_minio_using_presigned_url(presigned_url: str, file_path: str) -> Tuple:
    """
    Uploads a file to a presigned url.
    """
    with open(file_path, "rb") as f:
        response = requests.put(presigned_url, data=f)
        return response.status_code, response.text


def upload_model_to_dq(
    model: Any,
    model_parameters: Dict[str, Any],
    model_kind: ModelUploadType,
    tokenizer: PreTrainedTokenizer,
) -> None:
    """
    Uploads the model to the Galileo platform.

    :return: None
    """
    assert config.current_project_id, "Project id is required"
    assert config.current_run_id, "Run id is required"
    signed_url = api_client.get_presigned_url_for_model(
        project_id=config.current_project_id,
        run_id=config.current_run_id,
        model_kind=model_kind,
        model_parameters=model_parameters,
    )
    # save to temporary folder
    with tempfile.TemporaryDirectory() as tmpdirname:
        model.save_pretrained(f"{tmpdirname}/model_export")
        tokenizer.save_pretrained(f"{tmpdirname}/model_export")
        tar_path = f"{tmpdirname}/model.tar.gz"
        create_tar_archive(f"{tmpdirname}/model_export", tar_path)
        upload_to_minio_using_presigned_url(signed_url, tar_path)
    api_client.get_uploaded_model_info(
        project_id=config.current_project_id,
        run_id=config.current_run_id,
    )
