import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
#import torchvision transforms
from torchvision import transforms
import os

class BinarySegmentationDataset(Dataset):
    """
    BinarySegmentationDataset Class

    Args:
        img_dir (str): Directory containing the images.
        mask_dir (str): Directory containing the masks.
        transform (callable, optional): Transform to be applied on the images.
        mask_transform (callable, optional): Transform to be applied on the masks.

    """

    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(img_dir)

        #create image paths
        self.image_paths = [os.path.join(self.img_dir, image) for image in sorted(os.listdir(self.img_dir))]
        #create mask paths
        self.mask_paths = [os.path.join(self.mask_dir, mask) for mask in sorted(os.listdir(self.mask_dir))]
    
    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Total number of samples in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the image and the mask tensors.
        """

        img_path = self.image_paths[idx]
        #read image
        image = cv2.imread(img_path)
        #convert to RGB if not grayscale
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = np.expand_dims(image, axis=-1)

        image = image.transpose(2, 0, 1)
        #apply transforms
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)

        #read mask
        mask_path = self.mask_paths[idx]
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = torch.tensor(mask, dtype=torch.float32)
            mask = torch.clamp(mask, 0, 1)

        return image, mask