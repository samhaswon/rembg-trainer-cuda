"""
Helper code for loading and altering images in dataset
"""
import gc
import random

import imageio.v3 as iio
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import Dataset


class RandomCrop:
    """
    A class for performing random cropping of given image.
    Treats white pixels as empty space, and strives to return as few of them as possible.
    """

    THRESHOLDS = [
        0.5,
        0.8,
        0.9,
        0.95,
        0.98,
        0.99,
    ]  # allowed percentages of white pixels
    start_threshold_index = 0  # chosen percentage that we aim for

    def __init__(self, output_size, index=0):
        """
        Initialize the RandomCrop transformer.

        Parameters:
        - output_size (int or tuple): The desired size of the cropped image.
        - index (int): The starting threshold index.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.start_threshold_index = index

    @staticmethod
    def _calculate_white_percentage(img):
        """Calculate the percentage of white pixels in the image."""
        white_pixels = np.sum(img == 255)
        total_pixels = img.size
        return white_pixels / total_pixels

    def __call__(self, sample):
        """
        Apply the random crop to the input image + mask.

        Parameters:
        - sample (dict): Dictionary containing an image and its mask.

        Returns:
        - Dictionary containing the random crop of image and mask.
        """
        image, label = sample["image"], sample["label"]

        w, h = image.size
        grid_size = self.output_size[0]

        # splitting image into grid for faster search
        cells = [(i, j) for i in range(0, w, grid_size) for j in range(0, h, grid_size)]
        random.shuffle(cells)

        # looking for as non-empty cell as possible
        # lowering threshold of whiteness if none are found
        threshold_sequence = RandomCrop.THRESHOLDS[RandomCrop.start_threshold_index :]
        for threshold in threshold_sequence:
            for i, j in cells:
                if i + self.output_size[0] <= w and j + self.output_size[1] <= h:
                    cropped_image = tf.crop(image, i, j, *self.output_size)
                    cropped_label = tf.crop(label, i, j, *self.output_size)

                    if (
                        self._calculate_white_percentage(np.array(cropped_image))
                        <= threshold
                    ):
                        return {"image": cropped_image, "label": cropped_label}

        raise ValueError("Fully white image is given :(")


class HorizontalFlip:
    """
    A class to perform horizontal flipping of images.
    """

    def __call__(self, sample):
        """
        Flip the image and mask horizontally.

        Parameters:
        - sample (dict): Dictionary containing an image and its mask.

        Returns:
        - Dictionary containing the horizontally flipped image and mask.
        """
        image, label = sample["image"], sample["label"]

        # Apply horizontal flip
        image = tf.hflip(image)
        label = tf.hflip(label)

        return {"image": image, "label": label}


class VerticalFlip:
    """
    A class to perform vertical flipping of images.
    """

    def __call__(self, sample):
        """
        Flip the image and mask vertically.

        Parameters:
        - sample (dict): Dictionary containing an image and its mask.

        Returns:
        - Dictionary containing the vertically flipped image and mask.
        """
        image, label = sample["image"], sample["label"]

        # Apply vertical flip
        image = tf.vflip(image)
        label = tf.vflip(label)

        return {"image": image, "label": label}


class Rotation:
    """
    A class to rotate images by a given angle.
    """

    def __init__(self, degrees):
        """
        Initialize the Rotation transformer.

        Parameters:
        - degrees (float): The angle by which the image should be rotated.
        """
        self.degrees = degrees

    def __call__(self, sample):
        """
        Rotate the image and mask by a specified angle.

        Parameters:
        - sample (dict): Dictionary containing an image and its mask.

        Returns:
        - Dictionary containing the rotated image and mask.
        """
        image, label = sample["image"], sample["label"]

        # Apply rotation
        image = tf.rotate(image, self.degrees)
        label = tf.rotate(label, self.degrees)

        return {"image": image, "label": label}


class Resize:
    """
    A class to resize images to a specified size.
    """

    def __init__(self, size=1024):
        """
        Initialize the Resize transformer.

        Parameters:
        - size (int): The desired size of the image after resizing.
        """
        self.size = size

    def __call__(self, sample):
        """
        Resize the image and mask to the specified size.

        Parameters:
        - sample (dict): Dictionary containing an image and its mask.

        Returns:
        - Dictionary containing the resized image and mask.
        """
        image, label = sample["image"], sample["label"]

        # Resize both the image and label
        image = tf.resize(image, [self.size, self.size])
        label = tf.resize(label, [self.size, self.size])

        return {"image": image, "label": label}


class ToTensorLab:
    """
    A class to convert images from PIL format to PyTorch tensors.
    """

    def __call__(self, sample):
        """
        Convert the image and label from PIL format to PyTorch tensors.

        Parameters:
        - sample (dict): Dictionary containing an image and its mask.

        Returns:
        - Dictionary containing the image and mask as tensors.
        """
        from u2net_train import HALF_PRECISION

        image, label = sample["image"], sample["label"]

        # Convert to tensor
        image = tf.to_tensor(image)
        label = tf.to_tensor(label)

        if HALF_PRECISION:
            image, label = image.half(), label.half()

        return {"image": image, "label": label}


class SalObjDataset(Dataset):
    """
    Custom dataset class for salient object detection. This class helps in
    loading images, their corresponding masks, and applying the desired
    transformations before feeding them to the network.
    """

    def __init__(self, img_name_list, lbl_name_list, transform=None):
        """
        Initialize the SalObjDataset.

        Parameters:
        - img_name_list (list): List of paths to the images.
        - lbl_name_list (list): List of paths to the corresponding masks.
        - transform (callable, optional): Optional transform to be applied to both image & mask.
        """
        self.img_name_list = img_name_list
        self.lbl_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)

    def __getitem__(self, idx):
        """
        Fetch an image and its corresponding label, apply any transformations if needed,
        and return them as a dictionary.

        Parameters:
        - idx (int): Index of the desired sample.

        Returns:
        - Dictionary containing an image and its label.
        """
        # Read the images
        image_array = iio.imread(self.img_name_list[idx])
        label_array = iio.imread(self.lbl_name_list[idx])

        # Convert arrays to PIL images for compatibility with existing transforms
        image = Image.fromarray(image_array)
        label = Image.fromarray(label_array).convert("L")  # Convert RGB to grayscale
        # TODO add a check here if it even needs to be converted, to increase perf

        # Clean up memory
        del image_array, label_array
        gc.collect()

        sample = {"image": image, "label": label}

        # Apply the transformations
        if self.transform:
            sample = self.transform(sample)

        return sample
