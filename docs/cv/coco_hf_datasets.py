import os
from typing import Optional, Dict, Union

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from google.cloud import storage
from tqdm import tqdm
import datasets
    
class coco_hf_dataset_disk(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_path: str,
                 relative_img_path: Optional[str], 
                 relative_mask_path: Optional[str],
                 mask_transform: transforms=None, 
                 img_transform: transforms=None, 
                 size: int=1024,) -> None:
        """"
        COCO val dataset from galileo-public-data/CV_datasets/COCO_seg_val_5000/all_images
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

        if relative_img_path is None or relative_mask_path is None:
            dataset_path, relative_img_path, relative_mask_path = download_gcs_data()
        self.dataset_path = dataset_path
        self.relative_img_path = relative_img_path
        self.relative_mask_path = relative_mask_path
        self.images = sorted(os.listdir(os.path.join(dataset_path, relative_img_path)))
        self.masks = sorted(os.listdir(os.path.join(dataset_path, relative_mask_path)))
        # remove .DS_Store
        if self.images[0] == '.DS_Store':
            self.images = self.images[1:]
        if self.masks[0] == '.DS_Store':
            self.masks = self.masks[1:]

        num_images = len(self.images)
        num_masks = len(self.masks)
        print(f"Found dataset, there are {num_images} images and {num_masks} masks")

        # give default mask and image transforms
        if mask_transform is None:
            mask_transform = transforms.Compose([transforms.Resize((size, size), 
                                                                   resample=Image.NEAREST),
                                                 transforms.ToTensor()])
        if img_transform is None:
            img_transform = transforms.Compose([transforms.Resize((size, size)),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])
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

    def __len__(self) -> int:
        # returns length of the dataset
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, np.ndarray]]:
        # gets a single item from our dataset
        
        image_path = os.path.join(self.dataset_path, self.relative_img_path, self.images[idx])
        mask_path = os.path.join(self.dataset_path, self.relative_mask_path, self.masks[idx])
        image = Image.open(image_path)
        mask = Image.open(mask_path)

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
    

def download_gcs_data() -> None:
    # Initialize a client for Google Cloud Storage
    client = storage.Client()

    # Define the source bucket and folder paths
    bucket_name = 'galileo-public-data'
    dataset_path = '../../../'
    folder_paths = ['CV_datasets/COCO_seg_val_5000/all_images', 
                    'CV_datasets/COCO_seg_val_5000/all_masks']

    # Define the destination folder paths on disk
    destination_paths = [os.path.join(dataset_path, 'all_images'), 
                         os.path.join(dataset_path, 'all_masks')]

    if os.path.exists(os.path.join(dataset_path, folder_paths[0])):
        print(f'Found dataset in {os.path.abspath(dataset_path)}')
        num_images = len(os.listdir(os.path.join(dataset_path, folder_paths[0])))
        num_masks = len(os.listdir(os.path.join(dataset_path, folder_paths[1])))
        print(f'There are {num_images} images and {num_masks} masks')
        print('Skipping download...')
        return dataset_path, folder_paths[0], folder_paths[1]

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    # Loop over the folder paths and download the files
    for folder_path, destination_path in zip(folder_paths, destination_paths):
        # Create destination path if it doesn't exist
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        # Get a bucket object for the source bucket
        bucket = client.bucket(bucket_name)

        # Get a list of blobs in the folder
        blobs = bucket.list_blobs(prefix=folder_path)

        # Loop over the blobs and download each file with tqdm
        for blob in tqdm(blobs, desc=f"Downloading {folder_path}"):
            # Get the filename from the blob name
            filename = os.path.basename(blob.name)

            # Define the destination path for the file
            destination_file_path = os.path.join(destination_path, filename)

            # Download the file to the destination path
            blob.download_to_filename(destination_file_path)
    return dataset_path, folder_paths[0], folder_paths[1]