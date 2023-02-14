import base64
import mimetypes
import os
from io import BytesIO
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from pydantic import UUID4

from dataquality import config
from dataquality.clients.objectstore import ObjectStore
from dataquality.exceptions import GalileoException

object_store = ObjectStore()
B64_CONTENT_TYPE_DELIMITER = ";base64,"


def _b64_image_data_prefix(mimetype: str) -> bytes:
    return f"data:{mimetype}{B64_CONTENT_TYPE_DELIMITER}".encode("utf-8")


def _bytes_to_img(b: bytes) -> Image:
    return Image.open(BytesIO(b))


def _img_to_b64(img: Image) -> bytes:
    img_bytes = BytesIO()
    img.save(img_bytes, format=img.format)
    return base64.b64encode(img_bytes.getvalue())


def _img_to_b64_str(img: Image) -> str:
    prefix = _b64_image_data_prefix(mimetype=img.get_format_mimetype())
    data = _img_to_b64(img=img)
    return (prefix + data).decode("utf-8")


def _bytes_to_b64_str(img_bytes: bytes, img_path: Optional[str] = None) -> Image:
    mimetype = None

    if img_path is not None:
        # try to guess from path without loading image
        mimetype, _ = mimetypes.guess_type(img_path)

    if mimetype is None:
        # slow path - load image and read mimetype
        mimetype = _bytes_to_img(img_bytes).get_format_mimetype()
    prefix = _b64_image_data_prefix(mimetype=mimetype)
    b64_data = base64.b64encode(img_bytes)
    return (prefix + b64_data).decode("utf-8")


def _img_path_to_b64_str(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return _bytes_to_b64_str(img_bytes=f.read(), img_path=img_path)


def _write_img_bytes_to_file(
    img: Optional[Any] = None,
    img_path: Optional[str] = None,
    image_id: Optional[UUID4] = None,
) -> str:
    if image_id is None:
        image_id = uuid4()

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
    image_id: Optional[UUID4] = None,
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
        bucket_name=object_store.IMAGES_BUCKET_NAME,
    )
    os.remove(file_path)
    return object_name
