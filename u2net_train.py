import os
import glob
import torch
import argparse
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

bce_loss = nn.BCELoss(reduction="mean")


def get_args():
    parser = argparse.ArgumentParser(
        description="A program that trains ONNX model for use with rembg"
    )

    parser.add_argument(
        "--tra_image_dir",
        type=str,
        default="images",
        help="Directory with images.",
    )
    parser.add_argument(
        "--tra_label_dir",
        type=str,
        default="masks",
        help="Directory with masks.",
    )
    parser.add_argument("--epoch_num", type=int, default=200, help="Number of epochs.")
    parser.add_argument("--save_frq", type=int, default=300, help="Frequency to save.")
    parser.add_argument(
        "--batch",
        type=int,
        default=20,
        help="Single batch size. Warning: affects VRAM usage! Set to VRAM GBs minus one.",
    )

    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        print("CUDA acceleration enabled")
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders acceleration enabled")
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
            f"Loading checkpoint '{filename}'… (trained for {iteration_count} iterations)"
        )
        return iteration_count
    else:
        print(f"No checkpoint found at '{filename}'. Starting from scratch…")
        return 0


def load_dataset(img_dir, lbl_dir, ext):
    img_list = glob.glob(os.path.join(img_dir, "*" + ext))
    lbl_list = [os.path.join(lbl_dir, os.path.basename(img)) for img in img_list]

    return img_list, lbl_list


def muti_bce_loss_fusion(d_list, labels_v):
    losses = [bce_loss(d, labels_v) for d in d_list]
    total_loss = sum(losses)
    return losses[0], total_loss


def train_model(net, optimizer, dataloader, device, epoch_num, save_frq):
    print("---\n")
    iteration_count = load_checkpoint(net, optimizer)
    cumulative_loss = 0.0
    epoch_loss = 0.0

    for epoch in range(int(iteration_count / len(dataloader)), epoch_num):
        net.train()

        for i, data in enumerate(dataloader):
            iteration_count += 1

            inputs = data["image"].to(device).float()
            labels = data["label"].to(device).float()

            optimizer.zero_grad()
            outputs = net(inputs)

            first_output, combined_loss = muti_bce_loss_fusion(outputs, labels)
            combined_loss.backward()
            optimizer.step()

            epoch_loss += combined_loss.item()
            cumulative_loss += combined_loss.item()

            print(
                f"Epoch: {epoch + 1}/{epoch_num}, Iteration: {iteration_count}/{len(dataloader)*(epoch+1)}\n"
                f"Loss per epoch: {epoch_loss / iteration_count}, "
                f"loss per run: {cumulative_loss / iteration_count}\n"
            )

            # Saves model every save_frq iterations
            if iteration_count % save_frq == 0:
                save_model_as_onnx(
                    net, device, iteration_count
                )  # in ONNX format! ^_^ UwU
                print("Model saved")

            if iteration_count % len(dataloader) == 0:
                if epoch + 1 < epoch_num:
                    print("Checkpoint saved. Loading next crops…")
                else:
                    print("Final checkpoint")

        save_checkpoint(
            {
                "iteration_count": iteration_count,
                "state_dict": net.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
        )

        epoch_loss = 0.0

    save_model_as_onnx(net, device, iteration_count)
    print("Model saved for the last time")

    return net


def main():
    args = get_args()
    device = get_device()

    tra_image_dir = args.tra_image_dir
    tra_label_dir = args.tra_label_dir
    epoch_num = args.epoch_num
    save_frq = args.save_frq
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

    salobj_dataset = SalObjDataset(
        img_name_list=tra_img_name_list,
        lbl_name_list=tra_lbl_name_list,
        transform=transforms.Compose([RandomCrop(320), ToTensorLab(flag=0)]),
        # the model will be trained on many random 320*320 crops of your images
    )
    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=8,  # also reduce this if you don't have many cores available
    )

    net = U2NET(3, 1)
    net.to(device)

    optimizer = optim.Adam(
        net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    # Training loop
    train_model(net, optimizer, salobj_dataloader, device, epoch_num, save_frq)


if __name__ == "__main__":
    main()
