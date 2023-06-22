import os
from typing import Callable, Dict, Generator, Optional, Union
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

import dataquality as dq
from dataquality.clients.api import ApiClient
from dataquality.clients.objectstore import ObjectStore
from dataquality.integrations.torch_semantic_segmentation import watch
from tests.conftest import TestSessionVariables

dataset_path = os.path.abspath("tests/assets/testseg")

IMG_SIZE = 256
NC = 21  # Number of classes


class coco_hf_dataset_disk(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_path: str,
        relative_img_path: str,
        relative_mask_path: str,
        mask_transform: transforms = None,
        img_transform: transforms = None,
        size: int = 1024,
        binary: bool = False,
    ) -> None:
        """ "
        COCO val dataset from
        galileo-public-data/CV_datasets/COCO_seg_val_5000/all_images
        downloaded and located on disk.
        If no paths are provided we download the dataset from GCS and save it to disk.

        :param dataset_path: path to dataset
        :param relative_img_path: path to images relative to the dataset path
        :param relative_maks_path: path to masks relative to the dataset path
        :param mask_transform: transforms to apply to masks
        :param img_transform: transforms to apply to images
        :param size: size of image and mask
        """
        super(coco_hf_dataset_disk, self).__init__()

        self.dataset_path = dataset_path
        self.relative_img_path = relative_img_path
        self.relative_mask_path = relative_mask_path
        self.images = sorted(os.listdir(os.path.join(dataset_path, relative_img_path)))
        self.masks = sorted(os.listdir(os.path.join(dataset_path, relative_mask_path)))
        # remove .DS_Store
        if self.images[0] == ".DS_Store":
            self.images = self.images[1:]
        if self.masks[0] == ".DS_Store":
            self.masks = self.masks[1:]
        self.binary = binary

        num_images = len(self.images)
        num_masks = len(self.masks)
        print(f"Found dataset, there are {num_images} images and {num_masks} masks")

        # give default mask and image transforms
        if mask_transform is None:
            mask_transform = transforms.Compose(
                [
                    transforms.Resize((size, size), resample=Image.NEAREST),
                    transforms.ToTensor(),
                ]
            )
        if img_transform is None:
            img_transform = transforms.Compose(
                [
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
        self.mask_transform = mask_transform
        self.img_transform = img_transform

        self.class_dict = {
            "background": 0,
            "airplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "dining table": 11,
            "dog": 12,
            "horse": 13,
            "motorcycle": 14,
            "person": 15,
            "potted plant": 16,
            "sheep": 17,
            "couch": 18,
            "train": 19,
            "tv": 20,
        }

        self.int2str = {v: k for k, v in self.class_dict.items()}
        self.size = size

    def __len__(self) -> int:
        # returns length of the dataset
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, np.ndarray]]:
        # gets a single item from our dataset

        image_path = os.path.join(
            self.dataset_path, self.relative_img_path, self.images[idx]
        )
        mask_path = os.path.join(
            self.dataset_path, self.relative_mask_path, self.masks[idx]
        )
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # resize image and mask to given size
        unnormalized_image = image.copy().resize(
            (self.size, self.size), resample=Image.NEAREST
        )
        unnormalized_image = transforms.ToTensor()(unnormalized_image)
        unnormalized_image = expand_gray_channel()(unnormalized_image)
        unnormalized_image = np.array(unnormalized_image)

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.binary:
            mask_bool = mask > 0
            mask[mask_bool] = 1

        return {
            "image": image,
            "image_path": image_path,
            "mask_path": mask_path,
            "mask": mask,
            "idx": idx,
            "unnormalized_image": unnormalized_image,
        }


class expand_gray_channel:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # torch transform to expand gray channel to 3 channels
        if tensor.shape[0] > 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1:
            return tensor.expand(3, -1, -1)
        return tensor


img_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
        ),
        expand_gray_channel(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
mask_transforms = transforms.Compose(
    [
        transforms.PILToTensor(),
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST
        ),
    ]
)

train_image_path = "validation/images"
train_mask_path = "validation/masks"
validation_image_path = "validation/images"
validation_mask_path = "validation/masks"

train_dataset = coco_hf_dataset_disk(
    dataset_path=dataset_path,
    relative_img_path=train_image_path,
    relative_mask_path=train_mask_path,
    img_transform=img_transforms,
    mask_transform=mask_transforms,
    size=IMG_SIZE,
    binary=True,
)
validation_dataset = coco_hf_dataset_disk(
    dataset_path=dataset_path,
    relative_img_path=validation_image_path,
    relative_mask_path=validation_mask_path,
    img_transform=img_transforms,
    mask_transform=mask_transforms,
    size=IMG_SIZE,
    binary=True,
)
labels = ["background", "person"]
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=1)
validation_dataloader = DataLoader(
    validation_dataset, batch_size=6, shuffle=True, num_workers=1
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load(
    "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


@patch.object(dq.core.finish, "_version_check")
@patch.object(dq.core.finish, "_reset_run")
@patch.object(dq.core.finish, "upload_dq_log_file")
@patch.object(ApiClient, "make_request")
@patch.object(dq.core.finish, "wait_for_run")
@patch.object(ApiClient, "get_project_by_name")
@patch.object(ApiClient, "create_project")
@patch.object(ApiClient, "get_project_run_by_name", return_value={})
@patch.object(ObjectStore, "create_object")
@patch.object(ApiClient, "create_run")
@patch("dataquality.core.init._check_dq_version")
@patch.object(
    dq.clients.api.ApiClient,
    "get_healthcheck_dq",
    return_value={
        "bucket_names": {
            "images": "galileo-images",
            "results": "galileo-project-runs-results",
            "root": "galileo-project-runs",
        },
        "minio_fqdn": "127.0.0.1:9000",
    },
)
@patch.object(dq.core.init.ApiClient, "valid_current_user", return_value=True)
def test_semantic(
    mock_valid_user: MagicMock,
    mock_dq_healthcheck: MagicMock,
    mock_check_dq_version: MagicMock,
    mock_create_run: MagicMock,
    mock_create_object: MagicMock,
    mock_get_project_run_by_name: MagicMock,
    mock_create_project: MagicMock,
    mock_get_project_by_name: MagicMock,
    mock_wait_for_run: MagicMock,
    mock_make_request: MagicMock,
    mock_upload_log_file: MagicMock,
    mock_reset_run: MagicMock,
    mock_version_check: MagicMock,
    set_test_config: Callable,
    cleanup_after_use: Generator,
    test_session_vars: TestSessionVariables,
) -> None:
    mock_get_project_by_name.return_value = {"id": test_session_vars.DEFAULT_PROJECT_ID}
    mock_create_run.return_value = {"id": test_session_vars.DEFAULT_RUN_ID}
    set_test_config(current_project_id=None, current_run_id=None)
    # train for one epoch
    dq.init("semantic_segmentation", "Segmentation_Project", "COCO_dataset")

    watch(
        model,
        imgs_remote_location="gs://galileo-public-data/CV_datasets/Segmentation_Data",
        local_path_to_dataset_root=dataset_path,
        dataloaders={"training": train_dataloader, "validation": validation_dataloader},
    )
    dq.set_labels_for_run(labels)
    torch.cuda.amp.GradScaler()
    epochs = 1
    device = "cpu"
    with torch.autocast("cpu"):
        for epoch in range(epochs):
            dq.set_epoch_and_split(epoch, "training")
            for j, sample in enumerate(train_dataloader):
                imgs, masks = sample["image"], sample["mask"]
                out = model(imgs.to(device))

                # reshape to have loss for each pixel (bs * h * w, 21)\n",
                pred = out["out"].permute(0, 2, 3, 1).contiguous().view(-1, 21)
                masks = masks.long()
                msks_for_loss = masks.view(-1).to(device)

                loss = criterion(pred, msks_for_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    dq.finish()
