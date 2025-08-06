
from pathlib import Path
import numpy as np
import rasterio
import math
from collections import OrderedDict

import torch
from torch.utils.data import Dataset

from utils import make_tuple


root_dir = Path(__file__).parents[1]
data_dir = root_dir / 'data'

REF_t0 = '00'
PRE_t1 = '01'
REF_PREFIX_2 = '02'
MODIS_PREFIX = 'MODIS'
Landsat_PREFIX = 'Landsat'
Sentinel_PREFIX = 'Sentinel'
SCALE_FACTOR = 3


from pathlib import Path
from collections import OrderedDict

def get_pair_path_with_masks(im_dir):
    """
    Retrieves and pairs image files with their corresponding mask files.

    Args:
        im_dir (str): Path to the directory containing image and mask files.

    Returns:
        list of tuples: Each tuple contains the paths to an image and its corresponding mask, 
                        i.e., [(image_path, mask_path), ...].

    Raises:
        FileNotFoundError: If a corresponding mask file is not found for an image.
    """

    paths = []
    order = OrderedDict()
    order[0] = REF_t0 + '_' + MODIS_PREFIX
    order[1] = REF_t0 + '_' + Landsat_PREFIX
    order[2] = REF_t0 + '_' + Sentinel_PREFIX
    order[3] = PRE_t1 + '_' + MODIS_PREFIX
    order[4] = PRE_t1 + '_' + Landsat_PREFIX

    for prefix in order.values():
        for path in Path(im_dir).glob('*.tif'):
            if path.name.startswith(prefix):
                # Construct the corresponding mask path
                mask_name = path.stem.replace(prefix, f"{prefix}_mask") + '.npy'
                mask_path = Path(im_dir) / mask_name
                if mask_path.exists():
                    paths.append((path.expanduser().resolve(), mask_path.expanduser().resolve()))
                else:
                    raise FileNotFoundError(f"Mask not found for image {path}")
                # exit the inner loop
                break

    return paths

def load_image_and_mask_pair(im_dir):
    """
    Load all image and mask pairs from the specified directory.

    Args:
        im_dir (str): Path to the directory containing both image and mask files.

    Returns:
        tuple:
            - images (list of np.ndarray): List of loaded image arrays.
            - masks (list of np.ndarray): List of corresponding mask arrays.
    """

    # Get image and mask paths as pairs
    pairs = get_pair_path_with_masks(im_dir)  # Function from before
    images = []
    masks = []

    for image_path, mask_path in pairs:
        # Load the image
        with rasterio.open(str(image_path)) as ds:
            im = ds.read().astype(np.float32)  # C*H*W (numpy.ndarray)
            images.append(im)

        # Load the mask
        mask = np.load(mask_path).astype(np.float32)  # H*W (numpy.ndarray)
        masks.append(mask)

    return images, masks

def im2tensor(im):
    im = torch.from_numpy(im)
    return im


def im2tensor_mask(mask):
    out = torch.from_numpy(mask)
    return out

class PatchSet(Dataset):
    """
    Custom PyTorch dataset that divides each image and its corresponding mask into smaller patches.

    This is useful for training on high-resolution satellite imagery where loading entire images 
    into memory is inefficient. Patches are extracted with a sliding window strategy.

    """

    def __init__(self, image_dir, image_size, patch_size, patch_stride=None):
        super(PatchSet, self).__init__()
        patch_size = make_tuple(patch_size)
        if not patch_stride:
            patch_stride = patch_size
        else:
            patch_stride = make_tuple(patch_stride)

        self.root_dir = image_dir
        self.image_size = image_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.image_dirs = [p for p in self.root_dir.glob('*') if p.is_dir()]
        self.num_im_pairs = len(self.image_dirs)

        self.num_patches_x = math.ceil((image_size[0] - patch_size[0] + 1) / patch_stride[0])
        self.num_patches_y = math.ceil((image_size[1] - patch_size[1] + 1) / patch_stride[1])
        self.num_patches = self.num_im_pairs * self.num_patches_x * self.num_patches_y

        self.transform = im2tensor
        self.transform_mask = im2tensor_mask

    def map_index(self, index):
        id_n = index // (self.num_patches_x * self.num_patches_y)
        residual = index % (self.num_patches_x * self.num_patches_y)
        id_x = self.patch_stride[0] * (residual % self.num_patches_x)
        id_y = self.patch_stride[1] * (residual // self.num_patches_x)
        return id_n, id_x, id_y


    def __getitem__(self, index):
        id_n, id_x, id_y = self.map_index(index)
        
        images, masks = load_image_and_mask_pair(self.image_dirs[id_n])

        image_patches = [None] * len(images)
        mask_patches = [None] * len(masks)

        # Ensure the masks have a shape of (1, height, width)
        masks = [mask[np.newaxis, ...] if len(mask.shape) == 2 else mask for mask in masks]

        # Initialize patches for images and masks
        image_patches = [None] * len(images)
        mask_patches = [None] * len(masks)
        #scales = [1, 1, SCALE_FACTOR]

        scales = [SCALE_FACTOR, 1, SCALE_FACTOR]

        for i in range(len(image_patches)):
            scale = scales[i % 3]

            # Extract patches for images
            im = images[i][:,
                id_x * scale:(id_x + self.patch_size[0]) * scale,
                id_y * scale:(id_y + self.patch_size[1]) * scale]
            image_patches[i] = self.transform(im)

            # Extract patches for masks
            mask = masks[i][:,
                id_x * scale:(id_x + self.patch_size[0]) * scale,
                id_y * scale:(id_y + self.patch_size[1]) * scale]
            mask_patches[i] = self.transform_mask(mask)

        # Clean up memory
        del images[:]
        del masks[:]
        del images
        del masks

        return image_patches, mask_patches


    def __len__(self):
        return self.num_patches