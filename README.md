# PyTorch Segmentation Helper Classes

This repository provides two PyTorch-based helper classes that aim to simplify the task of training segmentation models on custom datasets. These classes are `BinarySegmentationDataset` and `MultiSegmentationDataset`.

## Table of Contents
- [Installation](#installation)
- [Requirements](#requirements)
- [Usage](#usage)
  - [BinarySegmentationDataset](#binarysegmentationdataset)
  - [MultiSegmentationDataset](#multisegmentationdataset)
- [Test](#test)

## Installation

To utilize these classes, you can clone this repository or simply download the `.py` files and include them in your existing project. Both classes are fully compatible with PyTorch's DataLoader class.

```
git clone https://github.com/masterbatcoderman10/Segmentation-Datasets.git
```

## Requirements

- Python 3.x
- PyTorch
- OpenCV

## Usage

### BinarySegmentationDataset

The `BinarySegmentationDataset` class is designed for datasets where each image has a corresponding binary mask.

#### Initialization

```python
from your_module import BinarySegmentationDataset

dataset = BinarySegmentationDataset(img_dir='path/to/images', mask_dir='path/to/masks', transform=your_transforms)
```

#### Arguments

- `img_dir` (str): Directory containing the images.
- `mask_dir` (str): Directory containing the masks.
- `transform` (callable, optional): Transform to be applied on the images.
- `mask_transform` (callable, optional): Transform to be applied on the masks.

#### Methods

The class provides standard PyTorch Dataset methods such as `__len__()` and `__getitem__()`.

### MultiSegmentationDataset

The `MultiSegmentationDataset` class is designed for more complex segmentation tasks where each image could have multiple masks corresponding to different classes.

#### Initialization

```python
from your_module import MultiSegmentationDataset

dataset = MultiSegmentationDataset(image_dir='path/to/images', mode=0, mask_dir='path/to/masks', classes=your_classes)
```

#### Arguments

- `image_dir` (str): Directory path containing the images.
- `mode` (int): Operational mode (0, 1, or 2).
  - `0`: Single mask file for each image, with different classes encoded by pixel values.
  - `1`: Multiple mask files for each image, each belonging to a different class, stored in separate class directories.
  - `2`: Single directory for each image containing multiple mask files, each belonging to a different class.
- `mask_dir` (str, optional): Directory path containing the masks.
- `masks_dict` (dict, optional): Dictionary mapping class names to mask files, required only if mode `1` is used.
- `transform` (callable, optional): Transform function to be applied to images.
classes (list | int, optional): 
  - List of class names, required for mode 2. 
  - Number of classes or list of class names if used with mode 0.

#### Methods

The class provides methods for handling masks in different operational modes: `process_mask_for_mode_zero()`, `process_mask_for_mode_one()`, and `process_mask_for_mode_two()`.

## Test

You can find detailed usage examples in the dataset_test.ipynb notebook. Within the `tests` folder.

Note: The datasets used to test the classes are not included in this repository. You can use your own datasets or download the ones used in the notebook from the following links:


