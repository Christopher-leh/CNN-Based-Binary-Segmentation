import torch
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from torchvision import tv_tensors
from torch.utils.data import Dataset
import os
from PIL import Image
import random


class Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, augmentation=None):
        """
        Dataset class for loading images and their masks.
        takes in the directory paths for images and masks, and an optional augmentation function
        """

        self.images_dir = images_dir
        self.masks_dir = masks_dir

        # sorted list of image and mask file paths
        self.image_paths = sorted(
            [os.path.join(images_dir, f) for f in os.listdir(images_dir)]
        )
        self.mask_paths = sorted(
            [os.path.join(masks_dir, f) for f in os.listdir(masks_dir)]
        )

        # set augmentation function (can be None)
        self.augmentation = augmentation

        # Base Transformations, resize to 256x256 and convert to tensor
        self.img_base = T.Compose(
            [
                T.ToImage(),  # convert to PIL Image
                T.ToDtype(torch.float32, scale=True),  # scale to [0,1]
                T.Resize((256, 256), antialias=True),  # resize, antialiasing for images
            ]
        )

        # Mask Transformations with nearest neighbor interpolation to preserve binary values
        self.mask_base = T.Compose(
            [
                T.ToImage(),  # convert to PIL Image
                T.ToDtype(
                    torch.float32, scale=False
                ),  # keep original values (0 or 255)
                T.Resize(
                    (256, 256),
                    antialias=False,  # no antialiasing for masks
                    interpolation=T.InterpolationMode.NEAREST,  # no smoothing for masks
                ),
            ]
        )

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # load with PIL
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        # Transform
        image = self.img_base(image)
        mask = self.mask_base(mask)

        # Binarization to ensure mask is 0 or 1
        mask = (mask > 0.5).float()

        # Augmentation
        if self.augmentation is not None:
            mask_wrapped = tv_tensors.Mask(
                mask
            )  # in tv_tensors to preserve the binary nature of the mask during augmentation
            image, mask = self.augmentation(image, mask_wrapped)
            mask = mask.data  # convert back to tensor from tv_tensors.Mask
            mask = (
                mask > 0.5
            ).float()  # ensure mask is still binary after augmentation

        return image, mask

    def __len__(self):
        return len(self.image_paths)


class DualRotation:
    """
    Rotates both image and mask by the same random angle within the given range.
    own implementation to add fill values for image and mask separately.
    fill_img: fill value for the image (e.g., 0.5 for gray)
    fill_mask: fill value for the mask (e.g., 0 for black)
    """

    def __init__(self, degrees, fill_img=0.5, fill_mask=0):
        self.degrees = degrees
        self.fill_img = fill_img  # Gray for the image
        self.fill_mask = fill_mask  # Black for the mask

    def __call__(self, image, mask):
        # Random angle
        angle = random.uniform(-self.degrees, self.degrees)

        # Rotate image with given fill value
        image = F.rotate(image, angle, fill=self.fill_img)

        # Rotate mask with given fill value and nearest neighbor interpolation to preserve binary nature
        mask = F.rotate(
            mask, angle, fill=self.fill_mask, interpolation=T.InterpolationMode.NEAREST
        )

        return image, mask
