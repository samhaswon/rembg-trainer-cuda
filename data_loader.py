import gc
import random

import imageio.v3 as iio
import numpy as np
import torchvision.transforms.functional as tf
from PIL import Image
from torch.utils.data import Dataset


class RandomCrop:
    THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
    start_threshold_index = 0  # variable to keep track of the current threshold index

    def __init__(self, output_size, index=0):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
        self.start_threshold_index = index

    @staticmethod
    def _calculate_white_percentage(img):
        white_pixels = np.sum(img == 255)
        total_pixels = img.size
        return white_pixels / total_pixels

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        w, h = image.size
        grid_size = self.output_size[
            0
        ]  # Assuming you want the grid to be of the same size as the crop

        cells = [(i, j) for i in range(0, w, grid_size) for j in range(0, h, grid_size)]
        random.shuffle(cells)

        threshold_sequence = (RandomCrop.THRESHOLDS[RandomCrop.start_threshold_index:])
        try:
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

        except ValueError as e:
            print(f"Error: {str(e)}")
            raise


class HorizontalFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply horizontal flip
        image = tf.hflip(image)
        label = tf.hflip(label)

        return {"image": image, "label": label}


class VerticalFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply vertical flip
        image = tf.vflip(image)
        label = tf.vflip(label)

        return {"image": image, "label": label}


class Rotation:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply rotation
        image = tf.rotate(image, self.degrees)
        label = tf.rotate(label, self.degrees)

        return {"image": image, "label": label}


class Resize:
    def __init__(self, size=1024):
        self.size = size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Resize both the image and label
        image = tf.resize(image, [self.size, self.size])
        label = tf.resize(label, [self.size, self.size])

        return {"image": image, "label": label}


class ToTensorLab:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Convert to tensor
        image = tf.to_tensor(image)
        label = tf.to_tensor(label)

        return {"image": image, "label": label}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.img_name_list = img_name_list
        self.lbl_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        image_array = iio.imread(self.img_name_list[idx])
        label_array = iio.imread(self.lbl_name_list[idx])

        # Convert arrays to PIL images for compatibility with existing transforms
        image = Image.fromarray(image_array)
        label = Image.fromarray(label_array)

        del image_array, label_array
        gc.collect()

        # If your images aren't already RGB or grayscale, you can convert them:
        # image = image.convert("RGB")
        # label = label.convert("L")

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        # save_dir = "transformed_images"
        # os.makedirs(save_dir, exist_ok=True)

        # Save the transformed image and label
        # save_image(sample["image"], os.path.join(save_dir, f"transformed_image_{idx}.png"))
        # save_image(sample["label"], os.path.join(save_dir, f"transformed_label_{idx}.png"))

        return sample
