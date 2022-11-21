import base64
import zlib
from io import BytesIO

import numpy as np
import vaex as vx
from PIL import Image

RGB = "RGB"
B64_CONTENT_TYPE_DELIMITER = ";base64,"


def _b64_image_data_prefix(mimetype: str) -> bytes:
    return f"data:{mimetype}{B64_CONTENT_TYPE_DELIMITER}".encode("utf-8")


def _b64_data_without_prefix(data: bytes) -> bytes:
    return data.decode("utf-8").split(B64_CONTENT_TYPE_DELIMITER)[-1].encode("utf-8")


def _b64_to_img(b64_data: bytes) -> Image:
    return Image.open(BytesIO(base64.b64decode(b64_data)))


def _img_to_rgb(img: Image) -> Image:
    return img.convert(RGB)


def _img_to_b64(img: Image) -> bytes:
    img_bytes = BytesIO()
    img.save(img_bytes, format=img.format)
    return base64.b64encode(img_bytes.getvalue())


filepath = "tmp/docker.jpeg"
with Image.open(filepath) as img:
    # prefix = _b64_image_data_prefix(mimetype=img.get_format_mimetype())
    # data = _img_to_b64(img=img)
    # _data = prefix + data
    # compressed_data = zlib.compress(_data)
    rgb = _img_to_rgb(img=img)
    print(type(list(rgb.getdata())[0][0]))


# for n in [100, 1000, 10000, 100000, 200000]:
#     arrs = np.array([compressed_data] * n)
#     df = vx.from_arrays(arrs=arrs)
#     df.export_hdf5(f"tmp/test-arrs-b64-{n}.hdf5")
