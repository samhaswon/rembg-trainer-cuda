import os

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision.utils import save_image
from torch.utils.data import Dataset
from PIL import Image
import imageio


class RandomCrop:
    step_count = 0  # static variable to keep track of the count

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (
                output_size,
                output_size,
            )  # Convert to tuple (height, width)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        while True:  # keep cropping until a non-white crop is found
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=self.output_size
            )
            cropped_image = TF.crop(image, i, j, h, w)
            cropped_label = TF.crop(label, i, j, h, w)

            # Convert the cropped image to numpy array
            numpy_img = np.array(cropped_image)

            # Check if all pixel values are 255 (full white)
            if not np.all(numpy_img == 255):
                break  # exit the loop if the crop isn't full white

        # Save the cropped images
        """save_dir = "steps"
        os.makedirs(save_dir, exist_ok=True)
        RandomCrop.step_count += 1
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(cropped_image)
        save_image(
            tensor_img, os.path.join(save_dir, f"step{RandomCrop.step_count}.png")
        )"""

        return {"image": cropped_image, "label": cropped_label}


class ToTensorLab:
    def __init__(self, flag=0):
        self.flag = flag

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Convert to tensor
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        return {"image": image, "label": label}


class SalObjDataset(Dataset):
    def __init__(self, img_name_list, lbl_name_list, transform=None):
        self.img_name_list = img_name_list
        self.lbl_name_list = lbl_name_list
        self.transform = transform

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, idx):
        image_array = imageio.imread(self.img_name_list[idx])
        label_array = imageio.imread(self.lbl_name_list[idx])

        # Convert arrays to PIL images for compatibility with existing transforms
        image = Image.fromarray(image_array)
        label = Image.fromarray(label_array)

        # If your images aren't already RGB or grayscale, you can convert them:
        # image = image.convert("RGB")
        # label = label.convert("L")

        sample = {"image": image, "label": label}

        if self.transform:
            sample = self.transform(sample)

        return sample
