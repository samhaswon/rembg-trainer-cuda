import os
import glob
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import U2NET
from data_loader import (
    RandomCrop,
    ToTensorLab,
    SalObjDataset,
)

SAVE_FRQ = 0
CHECK_FRQ = 0

bce_loss = nn.BCELoss(reduction="mean")


def get_args():
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
        "-l",
        "--tra_label_dir",
        type=str,
        default="masks",
        help="Directory with masks.",
    )
    parser.add_argument(
        "-e", "--epoch_num", type=int, default=200, help="Number of epochs."
    )
    parser.add_argument(
        "-s",
        "--save_frq",
        type=int,
        default=2,
        help="Frequency of saving onnx model (every X epochs).",
    )
    parser.add_argument(
        "-c",
        "--check_frq",
        type=int,
        default=1,
        help="Frequency of saving checkpoints (every X epochs).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        default=25,
        help="Size of a single batch. Warning: affects VRAM usage! Set to amount of VRAM in gigabytes - 20%%.",
    )

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        print("NVIDIA CUDA acceleration enabled")
        torch.multiprocessing.set_start_method("spawn")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders acceleration enabled")
        torch.multiprocessing.set_start_method("fork")
        return torch.device("mps")
    else:
        print("No GPU acceleration :/")
        return torch.device("cpu")


def save_model_as_onnx(model, device, ite_num, input_tensor_size=(1, 3, 320, 320)):
    x = torch.randn(*input_tensor_size, requires_grad=True)
    x = x.to(device)

    onnx_file_name = "saved_models/{}.onnx".format(ite_num)
    torch.onnx.export(
        model,
        x,
        onnx_file_name,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print("Model saved to:", onnx_file_name, "\n")


def save_checkpoint(state, filename="saved_models/checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(net, optimizer, filename="saved_models/checkpoint.pth.tar"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        epoch_count = checkpoint["epoch_count"]
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            f"Loading checkpoint '{filename}', trained for {epoch_count} epochs..."
        )
        return epoch_count
    else:
        print(f"No checkpoint found at '{filename}'. Starting from scratch...")
        return 0


def load_dataset(img_dir, lbl_dir, ext):
    img_list = glob.glob(os.path.join(img_dir, "*" + ext))
    lbl_list = [os.path.join(lbl_dir, os.path.basename(img)) for img in img_list]

    return img_list, lbl_list


def muti_bce_loss_fusion(d_list, labels_v):
    losses = [bce_loss(d, labels_v) for d in d_list]
    total_loss = sum(losses)
    return losses[0], total_loss


def get_dataloaders(tra_img_name_list, tra_lbl_name_list, batch):
    # Dataset with random crops
    salobj_dataset_crop = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([RandomCrop(320), ToTensorLab(flag=0)])
    )

    # Dataset with resized images
    salobj_dataset_resize = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([transforms.Resize((1024, 1024)), ToTensorLab(flag=0)])
    )

    # DataLoader for random crops
    salobj_dataloader_crop = DataLoader(
        salobj_dataset_crop,
        batch_size=batch,
        shuffle=True,
        num_workers=8
    )

    # DataLoader for resized images
    salobj_dataloader_resize = DataLoader(
        salobj_dataset_resize,
        batch_size=int(batch/3),
        shuffle=True,
        num_workers=8
    )

    return salobj_dataloader_crop, salobj_dataloader_resize


def train_model(net, optimizer, dataloader, device):
    epoch_loss = 0.0

    for i, data in enumerate(dataloader):
        inputs = data["image"].to(device).float()
        labels = data["label"].to(device).float()

        optimizer.zero_grad()

        outputs = net(inputs)

        first_output, combined_loss = muti_bce_loss_fusion(outputs, labels)
        combined_loss.backward()
        optimizer.step()

        epoch_loss += combined_loss.item()

        if i<10:
            print(
                f"  Iteration:  {i + 1}/{len(dataloader)}, "
                f"loss per iteration: {epoch_loss / (i + 1)}"
            )
        else:
            print(
                f"  Iteration: {i+1}/{len(dataloader)}, "
                f"loss per iteration: {epoch_loss / (i+1)}"
            )

    return epoch_loss


def train_epochs(net, optimizer, dataloader, device, epochs):
    print("---\n")

    for epoch in epochs:
        start_time = time.time()

        # this is where the training occurs!
        print(f"Epoch: {epoch + 1}/{len(epochs)}")
        epoch_loss = train_model(net, optimizer, dataloader, device)
        print(f"Loss per epoch: {epoch_loss}\n")

        if epoch == 0:
            print(
                f"Expected performance is {time.time() - start_time:.2f} seconds per epoch.\n"
            )

        # Saves model every save_frq iterations or during the last one
        if (epoch + 1) % SAVE_FRQ == 0 or epoch + 1 == len(epochs):
            save_model_as_onnx(
                net, device, epoch + 1
            )  # in ONNX format! ^_^ UwU

        # Saves checkpoint every check_frq epochs or during the last one
        if (epoch + 1) % CHECK_FRQ == 0 or epoch + 1 == len(epochs):
            save_checkpoint(
                {
                    "epoch_count": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )
            if epoch + 1 < len(epochs):
                print("Checkpoint saved. Loading next crops…\n")
            else:
                print("Final checkpoint made")

    return net


def main():
    args = get_args()
    device = get_device()

    global SAVE_FRQ, CHECK_FRQ
    SAVE_FRQ = args.save_frq
    CHECK_FRQ = args.check_frq
    tra_image_dir = args.tra_image_dir
    tra_label_dir = args.tra_label_dir
    epoch_num = args.epoch_num
    batch = args.batch

    tra_img_name_list, tra_lbl_name_list = load_dataset(
        tra_image_dir, tra_label_dir, ".png"
    )

    print(
        "Images: {},".format(len(tra_img_name_list)), "masks:", len(tra_lbl_name_list)
    )

    if len(tra_img_name_list) != len(tra_lbl_name_list):
        print("Different amounts of images and masks, can't proceed mate")
        return



    net = U2NET(3, 1)
    net.to(device)
    net.train()

    optimizer = optim.Adam(
        net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    salobj_dataloader_crop, salobj_dataloader_resize = get_dataloaders(tra_img_name_list, tra_lbl_name_list, batch)

    epochs = range(load_checkpoint(net, optimizer), epoch_num)

    # Training loop
    train_epochs(net, optimizer, salobj_dataloader_crop, device, epochs)


if __name__ == "__main__":
    main()
