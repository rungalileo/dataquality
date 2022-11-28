import base64
from io import BytesIO

from datasets import load_dataset
from PIL import Image

food = load_dataset("sasha/dog-food")


test = food["train"].select(range(2))


def pre_process(batch):
    # Image is in bytes and we use PIL to scale it to 128x128
    images = [Image.open(BytesIO(img)).resize((128, 128)) for img in batch["image"]]
    # We need to convert the image to base64 to display it in the frontend
    images_b64 = [base64.b64encode(img.tobytes()).decode("utf-8") for img in images]
    return {"img_thumbnail": images_b64}


test = test.map(pre_process, batched=True)
test[0]
