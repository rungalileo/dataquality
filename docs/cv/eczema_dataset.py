from typing import Optional, Dict, Union
from PIL import Image
import cv2
import torch
from torchvision import transforms
import os
import numpy as np

IMG_SIZE = 1024
NC = 2  # Number of classes

class dataset_disk(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,
                 relative_img_path: str,
                 relative_mask_path: str,
                 mask_transform: transforms=None,
                 img_transform: transforms=None,
                 size: int=1024,) -> None:
        """"

        :param dataset_path: path to dataset
        :param relative_img_path: path to images relative to the dataset path
        :param relative_masks_path: path to masks relative to the dataset path
        :param mask_transform: transforms to apply to masks
        :param img_transform: transforms to apply to images
        :param size: size of image and mask
        """
        super(dataset_disk, self).__init__()

        self.dataset_path = dataset_path
        self.relative_img_path = relative_img_path
        self.relative_mask_path = relative_mask_path
        self.images = sorted(os.listdir(os.path.join(dataset_path, relative_img_path)))[:100]
        self.masks = sorted(os.listdir(os.path.join(dataset_path, relative_mask_path)))[:100]
        try:
            self.images.remove('cl9xrwjqk3esh07xfale95f05.jpg')
            self.masks.remove('cl9xrwjqk3esh07xfale95f05_mask.jpg')
        except: 
            pass
        # remove .DS_Store
        if self.images[0] == '.DS_Store':
            self.images = self.images[1:]
        if self.masks[0] == '.DS_Store':
            self.masks = self.masks[1:]

        num_images = len(self.images)
        num_masks = len(self.masks)
        print(f"Found dataset, there are {num_images} images and {num_masks} masks")

        self.mask_transform = mask_transform
        self.img_transform = img_transform

        self.class_dict = { 'background': 0, 'eczema': 1}
        self.labels = [i for i in self.class_dict]

        self.int2str = {v: k for k, v in self.class_dict.items()}
        self.size = size

    def __len__(self) -> int:
        # returns length of the dataset
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, np.ndarray]]:
        # gets a single item from our dataset

        image_path = os.path.join(self.dataset_path, self.relative_img_path, self.images[idx])
        mask_path = os.path.join(self.dataset_path, self.relative_mask_path, self.masks[idx])
        image = Image.open(image_path).convert("RGB")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        try:
            mask[mask == 255] = 1
        except:
            print(mask_path)
            mask[mask == 255] = 1
        mask = Image.fromarray(mask)

        # resize image and mask to given size
        unnormalized_image = image.copy().resize((self.size, self.size), resample=Image.NEAREST)
        unnormalized_image = transforms.ToTensor()(unnormalized_image)
        unnormalized_image = expand_gray_channel()(unnormalized_image)
        unnormalized_image = np.array(unnormalized_image)


        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return {'image': image,
                'image_path': image_path,
                'mask_path': mask_path,
                'mask': mask,
                'idx': idx,
                'unnormalized_image': unnormalized_image}


class expand_gray_channel:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # torch transform to expand gray channel to 3 channels
        if tensor.shape[0] > 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1:
            return tensor.expand(3, -1, -1)
        return tensor
