"""
This script trains a deep learning model on an image dataset using various augmentations like flips, rotations, and crops.
The model is intended to use with rembg for background removal.
"""
import json
import os
import argparse
import time
from typing import List

import bitsandbytes as bnb
from PIL import Image
import sys
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
import torchvision.transforms.v2.functional as tf

from data_loader import (
    RandomCrop,
    Resize,
    ToTensorLab,
    VerticalFlip,
    HorizontalFlip,
    Rotation,
)
from model import U2NET, U2NETP

SAVE_FRQ = 1
CHECK_FRQ = 1
MAIN_SIZE = 512
IN_CHANNELS = 4
OUT_CHANNELS = 1
TRAIN_UNETP = True

BATCH_SIZE = 8
NUM_EPOCHS = 100

PREDICTION_FOLDERS = ["u2net", "u2netp", "birefnet"]

#: float16 if true, float32 if false
USE_AMP = True

# Defining BCE Loss for Binary Cross Entropy
bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

train_configs = {
    "plain_resized": {
        "name": "Plain Images",
        "message": "Learning the dataset itself...\n",
        "transform": [Resize(MAIN_SIZE), ToTensorLab()],
        "batch_factor": 1,
    },
    "flipped_v": {
        "name": "Vertical Flips",
        "message": "Learning the vertical flips of dataset images...\n",
        "transform": [Resize(MAIN_SIZE), VerticalFlip(), ToTensorLab()],
        "batch_factor": 1,
    },
    "flipped_h": {
        "name": "Horizontal Flips",
        "message": "Learning the horizontal flips of dataset images...\n",
        "transform": [Resize(MAIN_SIZE), HorizontalFlip(), ToTensorLab()],
        "batch_factor": 1,
    },
    "rotated_l": {
        "name": "Left Rotations",
        "message": "Learning the left rotations of dataset images...\n",
        "transform": [Resize(MAIN_SIZE), Rotation(90), ToTensorLab()],
        "batch_factor": 1,
    },
    "rotated_r": {
        "name": "Right Rotations",
        "message": "Learning the right rotation of dataset images...\n",
        "transform": [Resize(MAIN_SIZE), Rotation(270), ToTensorLab()],
        "batch_factor": 1,
    },
    "crops": {
        "name": "256px Crops",
        "message": "Augmenting dataset with random crops...\n",
        "transform": [Resize(2304), RandomCrop(256, 0), ToTensorLab()],
        "batch_factor": 16,  # because they are smaller => we can fit more in memory
    },
    "crops_loyal": {
        "name": "Different crops",
        "message": "Augmenting dataset with different crops...\n",
        "transform": [Resize(2304), RandomCrop(256, 3), ToTensorLab()],
        "batch_factor": 16,  # same here
    },
}


class RefinerDataset(Dataset):
    """
    Custom dataset class for salient object detection. This class helps in
    loading images, their corresponding masks, and applying the desired
    transformations before feeding them to the network.
    """

    def __init__(self, img_name_list, lbl_name_list, mask_name_list, boxes, transform=None):
        """
        Initialize the dataset.

        Parameters:
        - img_name_list (list): List of paths to the images.
        - lbl_name_list (list): List of paths to the corresponding masks.
        - transform (callable, optional): Optional transform to be applied to both image & mask.
        """
        self.img_name_list = img_name_list
        self.lbl_name_list = lbl_name_list
        self.mask_name_list = mask_name_list
        self.boxes = boxes
        self.transform = transform

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.img_name_list)

    def __getitem__(self, idx) -> dict:
        """
        Fetch an image and its corresponding label, apply any transformations if needed,
        and return them as a dictionary.

        Parameters:
        - idx (int): Index of the desired sample.

        Returns:
        - Dictionary containing an image and its label.
        """

        # Load images from the appropriate files
        image = Image.open(self.img_name_list[idx]).convert('RGB')
        label = Image.open(self.lbl_name_list[idx]).convert('L')
        mask = Image.open(self.mask_name_list[idx]).convert('L')
        x1, y1, x2, y2 = self.boxes[idx]

        image.putalpha(mask)

        # Convert PIL images to tensors and close them.
        image2 = tf.to_image(image)
        image.close()
        image = image2
        image = tf.to_dtype(image, dtype=torch.float32, scale=True)
        label2 = tf.to_image(label)
        label.close()
        label = label2
        label = tf.to_dtype(label, dtype=torch.float32, scale=True)

        sample = {"image": image[:, y1:y2, x1:x2], "label": label[:, y1:y2, x1:x2]}

        # Apply the transformations
        if self.transform:
            sample = self.transform(sample)

        return sample


def dice_loss(predict, target, smooth=1.0):
    """
    Calculates the Dice Loss.


    Returns:
        float: Dice Loss value.
    """
    predict = predict.contiguous()
    target = target.contiguous()

    intersection = (predict * target).sum(dim=2).sum(dim=2)

    loss = 1 - (
        (2.0 * intersection + smooth)
        / (predict.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)
    )

    return loss.mean()


def get_args():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="A program that trains ONNX model for use with rembg"
    )

    parser.add_argument(
        "-i",
        "--tra_image_dir",
        type=str,
        default="images",
        help="Directory with images.",
    )
    parser.add_argument(
        "-m",
        "--tra_masks_dir",
        type=str,
        default="masks",
        help="Directory with masks.",
    )
    parser.add_argument(
        "-s",
        "--save_frq",
        type=int,
        default=5,
        help="Frequency of saving onnx model (every X epochs).",
    )
    parser.add_argument(
        "-c",
        "--check_frq",
        type=int,
        default=5,
        help="Frequency of saving checkpoints (every X epochs).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=3,
        help="Size of a single batch loaded into memory. 1 is lowest possible; it may run on 8gb GPUs but also may not. 3 works well on 32gb of shared memory.",
    )
    parser.add_argument(
        "-p",
        "--plain_resized",
        type=int,
        default=5,
        help="Number of training epochs for plain_resized.",
    )
    parser.add_argument(
        "-vf",
        "--vflipped",
        type=int,
        default=2,
        help="Number of training epochs for flipped_v.",
    )
    parser.add_argument(
        "-hf",
        "--hflipped",
        type=int,
        default=2,
        help="Number of training epochs for flipped_h.",
    )
    parser.add_argument(
        "-left",
        "--rotated_l",
        type=int,
        default=2,
        help="Number of training epochs for rotated_l.",
    )
    parser.add_argument(
        "-right",
        "--rotated_r",
        type=int,
        default=2,
        help="Number of training epochs for rotated_r.",
    )
    parser.add_argument(
        "-r",
        "--rand",
        type=int,
        default=20,
        help="Number of training epochs for 256px crops.",
    )
    parser.add_argument(
        "-l",
        "--loyal",
        type=int,
        default=7,
        help="Number of training epochs for different 256px crops.",
    )

    return parser.parse_args()


def get_device():
    """
    Determines the device to run the model on (GPU/CPU).

    Returns:
        torch.device: Device type ('cuda', 'mps', or 'cpu').
    """
    if torch.cuda.is_available():
        print("NVIDIA CUDA acceleration enabled")
        torch.multiprocessing.set_start_method("spawn")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders acceleration enabled")
        torch.multiprocessing.set_start_method("fork")
        return torch.device("mps")
    else:
        print("No GPU acceleration :/")
        return torch.device("cpu")


def save_model_as_onnx(model, device, ite_num, input_tensor_size=(1, IN_CHANNELS, MAIN_SIZE, MAIN_SIZE)):
    """
    Saves the model in ONNX format.

    Parameters:
        model (nn.Module): The trained model.
        device (torch.device): The device where the model is located.
        ite_num (int): Amount of epochs already done.
        input_tensor_size (tuple, optional): The size of the input tensor. Defaults to (1, 3, 320, 320).
    """
    x = torch.randn(*input_tensor_size, requires_grad=True)
    x = x.to(device)

    onnx_file_name = f"saved_models/{ite_num}.onnx"
    torch.onnx.export(
        model,
        x,
        onnx_file_name,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Model saved to:", onnx_file_name, "\n")
    del x


def save_checkpoint(state, filename="saved_models/checkpoint.pth.tar"):
    """
    Saves the model's state as a checkpoint.

    Parameters:
        state (dict): State of the model to save.
        filename (str, optional): Path to save the checkpoint. Defaults to "saved_models/checkpoint.pth.tar".
    """
    torch.save({"state": state}, filename)


def load_checkpoint(net, optimizer, scaler, filename="saved_models/checkpoint.pth.tar"):
    """
    Loads model state from a checkpoint.

    Parameters:
        net (nn.Module): Model architecture.
        optimizer (Optimizer): Optimizer used during training.
        filename (str, optional): Path to the checkpoint. Defaults to "saved_models/checkpoint.pth.tar".

    Returns:
        dict: Counts of training epochs for various augmentations.
    """
    training_counts = {
        "plain_resized": 0,
        "flipped_v": 0,
        "flipped_h": 0,
        "rotated_l": 0,
        "rotated_r": 0,
        "crops": 0,
        "crops_loyal": 0,
    }

    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint["state"]["state_dict"])
        optimizer.load_state_dict(checkpoint["state"]["optimizer"])
        scaler.load_state_dict(checkpoint["state"]["scaler"])

        # Update the dictionary with values from the checkpoint
        # Only updates keys that exist in both dictionaries
        # This is done for expandability in future
        for key in training_counts:
            if key in checkpoint["state"]["training_counts"]:
                training_counts[key] = checkpoint["state"]["training_counts"][key]

        print(f"Loading checkpoint '{filename}'...")
    else:
        print(f"No checkpoint file found at '{filename}'. Starting from scratch...")
    print("\n———")

    return training_counts


def multi_loss_fusion(d_list, labels_v):
    """
    Combines BCE and Dice losses. Gives more weight to dice loss.

    Parameters:
        d_list (list): List of predicted outputs.
        labels_v (Tensor): Ground truth/target outputs.

    Returns:
        float: Combined loss value.
    """
    bce_losses = [bce_loss(d, labels_v) for d in d_list]
    dice_losses = [dice_loss(d, labels_v) for d in d_list]
    w_bce, w_dice = 2 / 3, 1 / 3
    combined_losses = [
        w_bce * bce + w_dice * dice for bce, dice in zip(bce_losses, dice_losses)
    ]
    total_loss = sum(combined_losses)
    # return combined_losses[0], total_loss
    return total_loss


def get_dataloader(batch_size):
    """
    Creates a DataLoader for the dataset.

    Parameters:
        tra_img_name_list (list): List of image filenames.
        tra_lbl_name_list (list): List of mask filenames.
        transform (transforms.Compose): Transformations to apply.
        batch_size (int): Amount of tensors to load into memory at once.

    Returns:
        DataLoader: DataLoader object for the dataset.
    """
    with open("boxes.json", "r") as f:
        boxes: List[str, int, int, int, int] = json.load(f)

    tra_img_name_list = [f"./images/{filename}" for filename, *_ in boxes]
    tra_img_name_list *= len(PREDICTION_FOLDERS)
    tra_lbl_name_list = [f"./masks/{filename}" for filename, *_ in boxes]
    tra_lbl_name_list *= len(PREDICTION_FOLDERS)
    tra_mask_name_list = [
        f"./{pf}/{filename}"
        for pf in PREDICTION_FOLDERS
        for filename, *_ in boxes
    ]
    boxes = [(x1, y1, x2, y2) for _, x1, y1, x2, y2 in boxes]
    boxes *= len(PREDICTION_FOLDERS)

    # Dataset with given transform
    dataset = RefinerDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        mask_name_list=tra_mask_name_list,
        boxes=boxes,
    )

    cores = 8  # freeing up memory a bit

    # DataLoader for the dataset
    dataloader = DataLoader(
        dataset, batch_size=max(1, int(batch_size)), shuffle=True, num_workers=cores, prefetch_factor=4
    )

    return dataloader


def train_model(net, optimizer, scheduler, dataloader, device, scaler):
    """
    Trains the model for a single epoch.

    Parameters:
        net (nn.Module): Model architecture.
        optimizer (Optimizer): Optimizer used during training.
        scheduler (lr_scheduler): Learning rate scheduler.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to train on (e.g., GPU/CPU).
        scaler:
    """
    epoch_loss = 0.0

    print(" ")
    for i, data in enumerate(dataloader):
        sys.stdout.write("\033[F")
        print(f"        Iteration: {i + 1:4}/{len(dataloader)}, ", end="")
        inputs = data["image"].to(device)
        labels = data["label"].to(device)

        with torch.autocast(device_type=device.__str__(), dtype=torch.float16, enabled=USE_AMP):
            outputs = net(inputs)
            combined_loss = multi_loss_fusion(outputs, labels)

        scaler.scale(combined_loss).backward()
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=1.0
        )  # Clip gradients if their norm exceeds 1
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        epoch_loss += combined_loss.item()

        print(f"loss: {epoch_loss / (i + 1):.5f}")

    return epoch_loss


def train_epochs(
    net, optimizer, scheduler, dataloader, device, epochs, training_counts, key, train_count, train_target, scaler
):
    """
    Train the model for given amount of epochs. Updates training counts.

    Parameters:
        net (nn.Module): The model architecture to be trained.
        optimizer (Optimizer): The optimizer used during training.
        scheduler (lr_scheduler): Scheduler to adjust the learning rate during training.
        dataloader (DataLoader): DataLoader object supplying the training data.
        device (torch.device): The device on which the training will take place (e.g., GPU/CPU).
        epochs (range): Number of epochs for which the model will be trained.
        training_counts (dict): Dictionary tracking the number of epochs trained for different configurations.
        key (str): Key for the specific training configuration.

    Returns:
        nn.Module: Trained model.
    """
    for index, epoch in enumerate(epochs):
        start_time = time.time()

        # this is where the training occurs!
        print(f"    Epoch: {epoch + 1}/{epochs[-1] + 1}")
        epoch_loss = train_model(net, optimizer, scheduler, dataloader, device, scaler)
        print(f"    Loss per epoch: {epoch_loss}\n")

        if sum(training_counts.values()) == 3:
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            perf = minutes + (seconds / 60)
            print(f"    Expected performance is {perf:.1f} minutes per epoch.\n")
        # Increment the corresponding training count
        training_counts[key] += 1

        # Saves model every save_frq iterations or during the last one
        if sum(training_counts.values()) % SAVE_FRQ == 0 or index + 1 == len(epochs):
            # in ONNX format! ^_^ UwU
            save_model_as_onnx(net, device, sum(training_counts.values()))

        # Saves checkpoint every check_frq epochs or during the last one
        # if sum(training_counts.values()) % CHECK_FRQ == 0 or index + 1 == len(epochs):
        if train_count % CHECK_FRQ == 0 or train_count + 1 == train_target:
            save_checkpoint(
                {
                    "epoch_count": epoch + 1,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "training_counts": training_counts,
                    "scaler": scaler.state_dict(),
                }
            )
            print("Checkpoint made\n")

    return net


def main():
    """
    Main function for initiating training of the model on the dataset.
    """
    device = get_device()

    global SAVE_FRQ, CHECK_FRQ
    targets = {
        "plain_resized": NUM_EPOCHS,
        "flipped_h": 0,
        "flipped_v": 0,
        "rotated_l": 0,
        "rotated_r": 0,
        "crops": 0,
        "crops_loyal": 0,
    }

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    if TRAIN_UNETP:
        net = U2NETP(IN_CHANNELS, OUT_CHANNELS)
    else:
        net = U2NET(IN_CHANNELS, OUT_CHANNELS)
    net.to(device)
    net.train()

    optimizer = bnb.optim.AdamW8bit(
        net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    grad_scaler = torch.cuda.amp.GradScaler()

    training_counts = load_checkpoint(net, optimizer, grad_scaler)

    print("———\n")

    scheduler = CosineAnnealingLR(optimizer, T_max=sum(targets.values()), eta_min=1e-6)

    def create_and_train(transform, batch_size, epochs, train_type, train_count, train_target):
        """Creates a dataloader and trains the network using the given parameters."""
        dataloader = get_dataloader(
            BATCH_SIZE
        )
        train_epochs(
            net,
            optimizer,
            scheduler,
            dataloader,
            device,
            epochs,
            training_counts,
            train_type,
            train_count,
            train_target,
            grad_scaler
        )

    complete = {
        "plain_resized": False,
        "flipped_h": False,
        "flipped_v": False,
        "rotated_l": False,
        "rotated_r": False,
        "crops": False,
        "crops_loyal": False,
    }

    while not all(list(complete.values())):
        # Training loop
        for train_type, config in train_configs.items():
            if training_counts[train_type] < targets[train_type]:
                print(config["message"])
                # epochs = range(training_counts[train_type], targets[train_type])
                transform = transforms.Compose(config["transform"])

                create_and_train(
                    transform, BATCH_SIZE * config["batch_factor"], range(1), train_type,
                    training_counts[train_type], targets[train_type]
                )
            else:
                print(f"Completed {train_type}")
                complete[train_type] = True

    print("Nothing left to do!")


if __name__ == "__main__":
    main()
