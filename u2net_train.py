import os
import glob
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Resize
from model import U2NET
from data_loader import (
    RandomCrop,
    ToTensorLab,
    SalObjDataset,
)

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
        default=500,
        help="Frequency of saving onnx model (every X iterations).",
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
    print("Model saved to:", onnx_file_name)


def save_checkpoint(state, filename="saved_models/checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(net, optimizer, filename="saved_models/checkpoint.pth.tar"):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        iteration_count = checkpoint["iteration_count"]
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(
            f"Loading checkpoint '{filename}', trained for {iteration_count} iterations..."
        )
        return iteration_count
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


def create_dataloader(img_name_list, lbl_name_list, transform, batch_size):
    salobj_dataset = SalObjDataset(
        img_name_list=img_name_list, lbl_name_list=lbl_name_list, transform=transform
    )
    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,  # adjust accordingly
    )
    return salobj_dataloader


def train_for_one_epoch(net, optimizer, dataloader, device, save_frq):
    net.train()

    epoch_loss = 0.0

    for i, data in enumerate(dataloader):
        start_time = time.time()

        inputs = data["image"].to(device).float()
        labels = data["label"].to(device).float()

        optimizer.zero_grad()
        outputs = net(inputs)

        first_output, combined_loss = muti_bce_loss_fusion(outputs, labels)
        combined_loss.backward()
        optimizer.step()

        epoch_loss += combined_loss.item()
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(
            f"Loss for this batch: {combined_loss.item()}, Time taken for this batch: {elapsed_time:.2f}s"
        )

        # Save model every save_frq iterations
        if (i + 1) % save_frq == 0:
            iteration_count = (epoch + 1) * len(
                dataloader
            ) + i  # compute overall iteration count
            save_model_as_onnx(net, device, iteration_count)

    return epoch_loss / len(dataloader)


def train_model(net, optimizer, dataloader, device, epoch_num, save_frq, check_frq):
    iteration_count = load_checkpoint(net, optimizer)
    cumulative_loss = 0.0
    epoch_loss = 0.0
    print("---\n")

    for epoch in range(int(iteration_count / len(dataloader)), epoch_num):
        net.train()

        for i, data in enumerate(dataloader):
            iteration_count += 1
            start_time = time.time()

            inputs = data["image"].to(device).float()
            labels = data["label"].to(device).float()

            optimizer.zero_grad()
            outputs = net(inputs)

            first_output, combined_loss = muti_bce_loss_fusion(outputs, labels)
            combined_loss.backward()
            optimizer.step()

            epoch_loss += combined_loss.item()
            cumulative_loss += combined_loss.item()
            end_time = time.time()
            elapsed_time = end_time - start_time

            print(
                f"Epoch: {epoch + 1}/{epoch_num}, Iteration: {iteration_count}/{len(dataloader)*(epoch+1)}\n"
                f"Loss per epoch: {epoch_loss / iteration_count}, "
                f"loss per run: {cumulative_loss / iteration_count}\n"
            )

            if iteration_count == 3:
                print(
                    f"Expected performance is {elapsed_time:.2f} per {len(dataloader)} crops.\n"
                )

            # Saves model every save_frq iterations or during the last one
            if iteration_count % save_frq == 0 or (
                epoch + 1 == epoch_num and iteration_count == len(dataloader)
            ):
                save_model_as_onnx(
                    net, device, iteration_count
                )  # in ONNX format! ^_^ UwU

        # Saves checkpoint every check_frq iterations or during the last one
        if (epoch + 1) % check_frq == 0 or epoch + 1 == epoch_num:
            save_checkpoint(
                {
                    "iteration_count": iteration_count,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )
            if epoch + 1 < epoch_num:
                print("Checkpoint saved. Loading next crops…\n")
            else:
                print("Final checkpoint made")

        epoch_loss = 0.0

    return net


def main():
    args = get_args()
    device = get_device()

    # subprocess.run(["ulimit", "-n", "4096"], shell=True, check=True)

    tra_image_dir = args.tra_image_dir
    tra_label_dir = args.tra_label_dir
    epoch_num = args.epoch_num
    save_frq = args.save_frq
    check_frq = args.check_frq

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

    for epoch in range(epoch_num):
        if (
            epoch < 10
        ):  # For the first 10 epochs, train on full images resized to 1024x1024
            transform = transforms.Compose([Resize((1024, 1024)), ToTensorLab(flag=0)])
            salobj_dataloader = create_dataloader(
                tra_img_name_list, tra_lbl_name_list, transform, int(batch / 3)
            )
        else:  # After 10 epochs, train on 320x320 random crops
            transform = transforms.Compose([RandomCrop(320), ToTensorLab(flag=0)])
            salobj_dataloader = create_dataloader(
                tra_img_name_list, tra_lbl_name_list, transform, batch
            )

        avg_epoch_loss = train_for_one_epoch(
            net, optimizer, salobj_dataloader, device, save_frq
        )

        print(f"Average Loss for Epoch {epoch+1}: {avg_epoch_loss}")

        if (epoch + 1) % check_frq == 0:
            save_checkpoint(
                {
                    "iteration_count": (epoch + 1) * len(salobj_dataloader),
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
            )

    print("Training finished!")


if __name__ == "__main__":
    main()
