from model import ImageTransformer
from u2net_train import train_configs, get_args, get_device, get_dataloader, save_model_as_onnx, load_dataset, \
    save_checkpoint
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms

SAVE_FRQ = 0
CHECK_FRQ = 0
MAIN_SIZE = 1024
IN_CHANNELS = 3
OUT_CHANNELS = 1


def train_model(net, optimizer, dataloader, device):
    """
    Trains the model for a single epoch.

    Parameters:
        net (nn.Module): Model architecture.
        optimizer (Optimizer): Optimizer used during training.
        scheduler (lr_scheduler): Learning rate scheduler.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to train on (e.g., GPU/CPU).
    """
    net.train()  # Set model to training mode
    running_loss = 0.0

    print(" ")
    for i, data in enumerate(dataloader):
        # Print iteration information
        sys.stdout.write("\033[F")
        print(f"        Iteration: {i + 1:4}/{len(dataloader)}, ", end="")

        # Get inputs and labels
        inputs = data["image"].to(device)
        labels = data["label"].to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(inputs)

        # Compute loss
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

        print(f"Loss: {loss.item():.4f}")
    return running_loss


def train_epochs(
        net, optimizer, dataloader, device, epochs, training_counts, key, train_count, train_target
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
        epoch_loss = train_model(net, optimizer, dataloader, device)
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
                }
            )
            print("Checkpoint made\n")

    return net


def load_checkpoint(net, optimizer, filename="saved_models/checkpoint.pth.tar"):
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


if __name__ == '__main__':
    args = get_args()
    SAVE_FRQ = args.save_frq
    CHECK_FRQ = args.check_frq
    tra_image_dir = args.tra_image_dir
    tra_label_dir = args.tra_masks_dir
    batch = args.batch

    targets = {
        "plain_resized": args.plain_resized,
        "flipped_h": args.hflipped,
        "flipped_v": args.vflipped,
        "rotated_l": args.rotated_l,
        "rotated_r": args.rotated_r,
        "crops": args.rand,
        "crops_loyal": args.loyal,
    }

    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

    tra_img_name_list, tra_lbl_name_list, mask_name_list = load_dataset(
        tra_image_dir, tra_label_dir, ".*"
    )

    print(f"Images: {format(len(tra_img_name_list))}, masks: {len(tra_lbl_name_list)}")
    if len(tra_img_name_list) != len(tra_lbl_name_list):
        print("Different amounts of images and masks, can't proceed.")
        exit(1)

    device = get_device()

    # Model
    net = ImageTransformer(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        img_size=(MAIN_SIZE, MAIN_SIZE),
        patch_size=32,
        embed_dim=768,
        depth=16,
        num_heads=12,
        ff_dim=3072).to(device)

    # Loss Function
    criterion = nn.MSELoss()  # Use MSELoss for regression tasks
    # criterion = nn.KLDivLoss(reduction='batchmean')  # Use Kullback-Leibler Divergence for the loss function.

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0001)

    net.to(device)
    net.train()

    training_counts = load_checkpoint(net, optimizer)

    for key, count in training_counts.items():
        if targets[key] < count:
            targets[key] = count
        print(
            f"Task: {train_configs[key]['name']:<17} Epochs done: {count}/{targets[key]}"
        )

    print("———\n")

    complete = {
        "plain_resized": False,
        "flipped_h": False,
        "flipped_v": False,
        "rotated_l": False,
        "rotated_r": False,
        "crops": False,
        "crops_loyal": False,
    }


    def create_and_train(transform, batch_size, epochs, train_type, train_count, train_target):
        """Creates a dataloader and trains the network using the given parameters."""
        dataloader = get_dataloader(
            tra_img_name_list, tra_lbl_name_list, transform, batch_size
        )
        train_epochs(
            net,
            optimizer,
            dataloader,
            device,
            epochs,
            training_counts,
            train_type,
            train_count,
            train_target
        )


    while not all(list(complete.values())):
        # Training loop
        for train_type, config in train_configs.items():
            if training_counts[train_type] < targets[train_type]:
                print(config["message"])
                # epochs = range(training_counts[train_type], targets[train_type])
                transform = transforms.Compose(config["transform"])

                create_and_train(
                    transform, batch * config["batch_factor"], range(1), train_type,
                    training_counts[train_type], targets[train_type]
                )

                # training_counts[train_type] = targets[train_type]
                # training_counts[train_type] += 1
            else:
                print(f"Completed {train_type}")
                complete[train_type] = True

    print("Nothing left to do!")
