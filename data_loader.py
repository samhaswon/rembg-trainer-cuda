import gc
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
            image = TF.crop(image, i, j, h, w)
            label = TF.crop(label, i, j, h, w)

            # Convert the cropped image to numpy array
            numpy_img = np.array(image)

            white_pixels = np.sum(numpy_img == 255)
            total_pixels = numpy_img.size

            if white_pixels / total_pixels <= 0.8:
                break  # exit the loop if 80% or less of the pixels are white

        # Save the cropped images
        """save_dir = "steps"
        os.makedirs(save_dir, exist_ok=True)
        RandomCrop.step_count += 1
        to_tensor = transforms.ToTensor()
        tensor_img = to_tensor(cropped_image)
        save_image(
            tensor_img, os.path.join(save_dir, f"step{RandomCrop.step_count}.png")
        )"""
        return {"image": image, "label": label}


class HorizontalFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply horizontal flip
        image = TF.hflip(image)
        label = TF.hflip(label)

        return {"image": image, "label": label}


class VerticalFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply vertical flip
        image = TF.vflip(image)
        label = TF.vflip(label)

        return {"image": image, "label": label}


class Rotation90:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply 90-degree rotation
        image = TF.rotate(image, 90)
        label = TF.rotate(label, 90)

        return {"image": image, "label": label}


class Rotation270:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply 270-degree rotation
        image = TF.rotate(image, 270)
        label = TF.rotate(label, 270)

        return {"image": image, "label": label}


class Resize:
    def __init__(self, size=1024):
        self.size = size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Resize both the image and label
        image = TF.resize(image, (self.size, self.size))
        label = TF.resize(label, (self.size, self.size))

        return {"image": image, "label": label}


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
