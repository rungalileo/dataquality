import hashlib
import os
from io import BytesIO
from typing import Any, Optional

from PIL import Image
from pydantic import UUID4

from dataquality import config
from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException

object_store = ObjectStore()
B64_CONTENT_TYPE_DELIMITER = ";base64,"

api_client = ApiClient()


def _bytes_to_img(b: bytes) -> Image:
    return Image.open(BytesIO(b))


def _write_img_bytes_to_file(
    img: Optional[Any] = None,
    img_path: Optional[str] = None,
    image_id: Optional[str] = None,
) -> str:
    img_bytes = BytesIO()
    if img_path is not None:
        with open(img_path, "rb") as f:
            img_bytes.write(f.read())
        _format = img_path.split(".")[-1].upper()
    elif img is not None:
        _format = img.format
        img.save(
            img_bytes,
            format=_format,
        )
    else:
        raise ValueError("img or img_path must be provided")

    if image_id is None:
        image_id = hashlib.md5(img_bytes.getvalue()).hexdigest()

    with open(
        f"{image_id}.{str(_format).lower()}",
        "wb",
    ) as f:
        f.write(img_bytes.getvalue())
    filepath = f"{image_id}.{str(_format).lower()}"
    return filepath


def _write_image_bytes_to_objectstore(
    project_id: Optional[UUID4] = None,
    img: Optional[Any] = None,
    img_path: Optional[str] = None,
    image_id: Optional[str] = None,
) -> str:
    if project_id is None:
        project_id = config.current_project_id
    if project_id is None:
        raise GalileoException(
            "project_id is not set in your config. Have you run dq.init()?"
        )
    file_path = _write_img_bytes_to_file(
        img=img,
        img_path=img_path,
        image_id=image_id,
    )
    object_name = f"{project_id}/{file_path}"
    object_store.create_object(
        object_name=object_name,
        file_path=file_path,
        bucket_name=config.images_bucket_name,
    )
    os.remove(file_path)
    return object_name
