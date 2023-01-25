import base64
import mimetypes
from io import BytesIO
from typing import Optional

from PIL import Image

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


def _bytes_to_b64_str(img_bytes: bytes, img_path: Optional[str] = None) -> str:
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
