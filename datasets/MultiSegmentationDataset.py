#do the necessary imports
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision import transforms

class MultiSegmentationDataset(Dataset):

    def __init__(self, image_dir, mode = 0 | 1 | 2, mask_dir=None, masks_dict=None, transform=None, classes=None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mode = mode
        self.masks_dict = masks_dict
        self.images = os.listdir(image_dir)
        self.image_paths = [os.path.join(self.image_dir, image) for image in sorted(os.listdir(self.image_dir))]
        if mode == 0:
            self.mask_paths = [os.path.join(self.mask_dir, mask) for mask in sorted(os.listdir(self.mask_dir))]
            self.n_classes = classes
        elif mode == 1:
            #for each directory containing masks of a different class, create a list of paths to the masks
            self.mask_paths = {key: [os.path.join(self.mask_dir, mask) for mask in sorted(os.listdir(self.mask_dir))] for key in sorted(self.masks_dict.keys())}
            #for each key assert each file list has the same length as the length of the image list
            for key in self.mask_paths.keys():
                assert len(self.mask_paths[key]) == len(self.image_paths), "number of masks for a class must be equal to number of images, if some images do not have a mask of a specific class, consider using mode 2"
            
            self.n_classes = len(self.masks_dict.keys())
        elif mode == 2:
            #use this mode if you have a directory for every image, in that directory there are the masks for different classes
            self.mask_dirs = [os.path.join(self.mask_dir, mask_dir) for mask_dir in sorted(os.listdir(self.mask_dir))]
            assert classes is not None, "classes must be provided for mode 2"
            self.classes = sorted(classes)
            self.n_classes = len(self.classes)

        

    def __len__(self):
        return len(self.image_paths)

    #this function deal with masks that are essentially black and white images with different values between 0-255 indicating different classes
    def process_mask_for_mode_zero(self, mask_path):

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #assert mask only has 2 dimensions, if not then squeeze
        if len(mask.shape) > 2:
            mask = np.squeeze(mask)
        #convert to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        #convert to one hot encoding
        mask = torch.nn.functional.one_hot(mask.to(torch.int64), num_classes=self.n_classes).permute(2,0,1).float()
        
        return mask

    #this function creates a mask for separate masks belonging to separate directories and processes them into one mask
    def process_mask_for_mode_one(self, idx, H, W):

        background = np.ones((H, W))
        mask = np.zeros((self.n_classes+1, H, W))
        for i, key in enumerate(self.masks_dict.keys()):
            mask_path = self.mask_paths[key][idx]
            mask[i+1] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask[i+1] = np.clip(mask[i+1], 0, 1)
            background = background - mask[i+1]
        
        background = np.clip(background, 0, 1)
        mask[0] = background

        #convert to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask
    
    def process_mask_for_mode_two(self, idx, H, W):

        background = np.ones((H, W))
        mask = np.zeros((self.n_classes+1, H, W))
        mask_dir = self.mask_dirs[idx]
        dir_files = sorted(list(os.listdir(mask_dir)))
        for file in dir_files:
            for cls_ix, cls in enumerate(self.classes):
                if cls in file:
                    mask[cls_ix+1] = cv2.imread(os.path.join(mask_dir, file), cv2.IMREAD_GRAYSCALE)
                    mask[cls_ix+1] = np.clip(mask[cls_ix+1], 0, 1)
                    background = background - mask[cls_ix+1]
        
        background = np.clip(background, 0, 1)
        mask[0] = background

        #convert to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        return mask
        

    def __getitem__(self, idx):

        #process image
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
        
        #process mask
        if self.mode == 0:
            mask_path = self.mask_paths[idx]
            mask = self.process_mask_for_mode_zero(mask_path)
        elif self.mode == 1:
            H, W = image.shape[1:]
            mask = self.process_mask_for_mode_one(idx, H, W)
        elif self.mode == 2:
            H, W = image.shape[1:]
            mask = self.process_mask_for_mode_two(idx, H, W)
        
        return image, mask