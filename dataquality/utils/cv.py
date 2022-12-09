import base64
from io import BytesIO

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
