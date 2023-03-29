import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class coco_hf_dataset(torch.utils.data.Dataset):
    def __init__(self, ds, mask_transform=None, img_transform=None, size: int=1024):
        super(coco_hf_dataset, self).__init__()
        self.ds = ds
        self.mask_transform = mask_transform
        self.img_transform = img_transform

        self.class_dict = { 'background': 0,
                            'airplane': 1,
                            'bicycle': 2,
                            'bird': 3,
                            'boat': 4,
                            'bottle': 5,
                            'bus': 6,
                            'car': 7,
                            'cat': 8,
                            'chair': 9,
                            'cow': 10,
                            'dining table': 11,
                            'dog': 12,
                            'horse': 13,
                            'motorcycle': 14,
                            'person': 15,
                            'potted plant': 16,
                            'sheep': 17,
                            'couch': 18,
                            'train': 19,
                            'tv': 20}
        self.int2str = {v: k for k, v in self.class_dict.items()}
        self.size = size

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = sample['image']
        mask = sample['mask']

        values = np.unique(mask)
        int2str_out = {v: self.int2str[v] for v in values}
        # resize image and mask to 640x640
        unnormalized_image = image.copy().resize((self.size, self.size), resample=Image.NEAREST)
        unnormalized_image = transforms.ToTensor()(unnormalized_image)
        unnormalized_image = expand_gray_channel()(unnormalized_image)
        unnormalized_image = np.array(unnormalized_image)
        

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return {'image': image,
                'mask': mask,
                'idx': idx,
                'unnormalized_image': unnormalized_image}


class expand_gray_channel:
    def __call__(self, tensor):
        if tensor.shape[0] > 3:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[0] == 1:
            return tensor.expand(3, -1, -1)
        return tensor
