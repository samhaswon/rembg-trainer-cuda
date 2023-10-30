import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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

        return {"image": cropped_image, "label": cropped_label}


class HorizontalFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply horizontal flip
        flipped_image = TF.hflip(image)
        flipped_label = TF.hflip(label)

        return {"image": flipped_image, "label": flipped_label}


class VerticalFlip:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply vertical flip
        flipped_image = TF.vflip(image)
        flipped_label = TF.vflip(label)

        return {"image": flipped_image, "label": flipped_label}


class Rotation90:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply 90-degree rotation
        rotated_image = TF.rotate(image, 90)
        rotated_label = TF.rotate(label, 90)

        return {"image": rotated_image, "label": rotated_label}


class Rotation270:
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Apply 270-degree rotation
        rotated_image = TF.rotate(image, 270)
        rotated_label = TF.rotate(label, 270)

        return {"image": rotated_image, "label": rotated_label}


class Resize:
    def __init__(self, size=1024):
        self.size = size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        # Resize both the image and label
        resized_image = TF.resize(image, (self.size, self.size))
        resized_label = TF.resize(label, (self.size, self.size))

        return {"image": resized_image, "label": resized_label}


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
