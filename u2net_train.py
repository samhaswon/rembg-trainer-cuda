import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from data_loader import (
    RandomCrop,
    ToTensorLab,
    SalObjDataset,
)
from model import U2NET

bce_loss = nn.BCELoss(reduction="mean")


def save_model_as_onnx(model, dev, ite_num, input_tensor_size=(1, 3, 320, 320)):
    x = torch.randn(*input_tensor_size, requires_grad=True)
    x = x.to(dev)

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


def muti_bce_loss_fusion(d_list, labels_v):
    losses = [bce_loss(d, labels_v) for d in d_list]
    total_loss = sum(losses)
    return losses[0], total_loss


def main():
    # Choosing backend
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("CUDA Acceleration enabled")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple M1 acceleration enabled")
    else:
        device = torch.device("cpu")
        print("No GPU acceleration :/")

    # Directories and model specifications
    tra_image_dir = "images"
    tra_label_dir = "masks"
    image_ext = ".png"
    label_ext = ".png"
    epoch_num = 50
    save_frq = 300
    batch = (
        20  # Affects VRAM usage! 20 uses ~20+ gb of VRAM. Reduce to suit your hardware.
    )

    tra_img_name_list = glob.glob(os.path.join(tra_image_dir, "*" + image_ext))

    tra_lbl_name_list = []
    for img_path in tra_img_name_list:
        img_name = img_path.split(os.sep)[-1]

        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]

        tra_lbl_name_list.append(os.path.join(tra_label_dir, imidx + label_ext))

    print("Images: ", len(tra_img_name_list))
    print("Masks: ", len(tra_lbl_name_list))

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
        num_workers=6,  # also reduce this if you don't have many cores available
    )

    net = U2NET(3, 1)
    net.to(device)

    optimizer = optim.Adam(
        net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
    )

    # Training loop
    print("Launching...")
    print("---")

    ite_num = 0
    running_loss = 0.0
    running_tar_loss = 0.0

    for epoch in range(epoch_num):
        net.train()

        for i, data in enumerate(salobj_dataloader):
            ite_num += 1

            inputs, labels = data["image"].type(torch.FloatTensor).to(device), data[
                "label"
            ].type(torch.FloatTensor).to(device)

            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss2, loss = muti_bce_loss_fusion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_tar_loss += loss2.item()

            print(
                f"[Epoch: {epoch + 1}/{epoch_num}, Iteration: {ite_num}] Train Loss: {running_loss/ite_num}, Target Loss: {running_tar_loss/ite_num}"
            )

            # Saves model every save_frq iterations
            if ite_num % save_frq == 0:
                save_model_as_onnx(net, device, ite_num)  # in ONNX format! ^_^ UwU
                print("Model saved")
                running_loss = 0.0
                running_tar_loss = 0.0

    save_model_as_onnx(net, device, ite_num)
    print("Model saved for the last time")


if __name__ == "__main__":
    main()
