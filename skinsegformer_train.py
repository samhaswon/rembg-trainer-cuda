#!/usr/bin/env python3
"""
Fine-tune BiRefNet_lite-2K with Hugging Face Trainer on a folder dataset.

Dataset layout
--------------
root/
  images/
    xxx.png|jpg|jpeg|webp
  masks/
    xxx.png|jpg|jpeg|webp  (same basename as image)

Usage
-----
python birefnet_train_torch.py \
  --data_root /path/to/root \
  --output_dir ./birefnet_skin_out \
  --image_size 1024 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --num_train_epochs 10

Requirements
------------
pip install "transformers>=4.44" datasets accelerate torch torchvision pillow
"""

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import bitsandbytes as bnb
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import SkinSegFormerConfig, SkinSegFormer


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_pairs(root: Path, images_name: str = "images", masks_name: str = "masks") -> List[Tuple[Path, Path]]:
    """
    List (image, mask) pairs by matching filenames in two folders.

    Parameters
    ----------
    root : Path
        Root folder containing 'images' and 'masks'.
    images_name : str, optional
        The name of the folder containing images
    masks_name : str, optional
        The name of the folder containing masks.

    Returns
    -------
    list[tuple[Path, Path]]
        Sorted list of pairs.
    """
    img_dir = root / images_name
    msk_dir = root / masks_name
    if not img_dir.is_dir() or not msk_dir.is_dir():
        raise FileNotFoundError("Expected subfolders 'images' and 'masks' under data_root.")

    images = sorted([p for p in img_dir.iterdir() if _is_image(p)])
    pairs = []
    for ip in images:
        mp = msk_dir / ip.name
        if mp.exists() and _is_image(mp):
            pairs.append((ip, mp))
    if not pairs:
        raise RuntimeError("No matching image/mask pairs found.")
    return pairs


def load_image(path: Path) -> torch.Tensor:
    """
    Load image as CHW float tensor in [0,1].

    Parameters
    ----------
    path : Path
        Image path.

    Returns
    -------
    torch.Tensor
        Tensor of shape (3, H, W).
    """
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    return t


def load_mask(path: Path) -> torch.Tensor:
    """
    Load mask as 1xHxW float tensor in [0,1], preserves soft edges.

    Parameters
    ----------
    path : Path
        Mask path.

    Returns
    -------
    torch.Tensor
        Tensor of shape (1, H, W).
    """
    m = Image.open(path).convert("L")
    arr = np.asarray(m, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    t = torch.from_numpy(arr)[None, ...]
    return t.clamp(0.0, 1.0)


def resize_and_pad(x: torch.Tensor, target: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Resize a CHW tensor to the longer side == target, keep aspect, then pad to multiple of 32.

    Parameters
    ----------
    x : torch.Tensor
        CHW tensor.
    target : int
        Target longer side.

    Returns
    -------
    tuple
        (resized+pad tensor, (orig_h, orig_w))
    """
    c, h, w = x.shape
    if h >= w:
        new_h = target
        new_w = max(1, int(round(w * target / h)))
    else:
        new_w = target
        new_h = max(1, int(round(h * target / w)))
    x = F.interpolate(x[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]

    pad_h = (32 - new_h % 32) % 32
    pad_w = (32 - new_w % 32) % 32
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, (h, w)


def resize_and_pad_square(x: torch.Tensor, target: int) -> torch.Tensor:
    """
    Resize CHW tensor so the longer side == target, preserve aspect,
    then pad to (target, target). target should be a multiple of 32.
    """
    c, h, w = x.shape
    if h >= w:
        new_h = target
        new_w = max(1, int(round(w * target / h)))
    else:
        new_w = target
        new_h = max(1, int(round(h * target / w)))

    x = F.interpolate(x[None], size=(new_h, new_w), mode="bilinear", align_corners=False)[0]

    pad_h = target - new_h
    pad_w = target - new_w
    # pad=(left, right, top, bottom)
    x = F.pad(x, (0, pad_w, 0, pad_h))
    return x


class FolderSegDataset(Dataset):
    """
    Simple folder dataset for image segmentation with soft masks.
    """

    def __init__(
        self,
        pairs: List[Tuple[Path, Path]],
        image_size: int,
        gaussian_noise_std: float = 0.0,
        gaussian_noise_prob: float = 0.0,
    ) -> None:
        self.pairs = pairs
        self.image_size = image_size
        self.gaussian_noise_std = max(0.0, float(gaussian_noise_std))
        self.gaussian_noise_prob = float(min(1.0, max(0.0, gaussian_noise_prob)))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ip, mp = self.pairs[idx]
        img = load_image(ip)
        msk = load_mask(mp)

        # ensure target is multiple of 32 once, at startup or here
        target = ( (self.image_size + 31) // 32 ) * 32

        img_sq = resize_and_pad_square(img, target)
        msk_sq = resize_and_pad_square(msk, target)

        if self.gaussian_noise_std > 0.0 and torch.rand(1).item() < self.gaussian_noise_prob:
            noise = torch.randn_like(img_sq) * self.gaussian_noise_std
            img_sq = (img_sq + noise).clamp(0.0, 1.0)
        # masks with nearest, same final size

        return {"pixel_values": img_sq, "labels": msk_sq, "id": ip.name}


@dataclass
class Collator:
    """
    Identity collator, already resized and padded.
    """

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features], dim=0)
        labels = torch.stack([f["labels"] for f in features], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}


class BCEPlusDice(nn.Module):
    """
    BCE + soft Dice for alpha-like masks.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = pred.clamp(0, 1)
        num = 2 * (pred * target).sum(dim=(1, 2, 3))
        den = pred.pow(2).sum(dim=(1, 2, 3)) + target.pow(2).sum(dim=(1, 2, 3)) + eps
        return 1.0 - (num / den)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Model may output logits already in 0..1 or raw; be safe
        if logits.shape[1] != 1:
            raise ValueError("Expecting single-channel output.")
        pred = torch.sigmoid(logits) if (logits.min() < 0 or logits.max() > 1) else logits
        bce = self.bce(pred, target)
        dice = self.dice_loss(pred, target).mean()
        return self.bce_weight * bce + self.dice_weight * dice


def split_pairs(pairs: List[Tuple[Path, Path]], eval_size: int, seed: int) -> Tuple[List, List]:
    """
    Deterministic split, 100 samples for eval by default.

    Returns
    -------
    (train_pairs, eval_pairs)
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    eval_idx = sorted(idx[: min(eval_size, len(pairs))].tolist())
    train_idx = sorted(idx[min(eval_size, len(pairs)) :].tolist())
    return [pairs[i] for i in train_idx], [pairs[i] for i in eval_idx]


def compute_seg_metrics(eval_pred, label_ids) -> dict:
    # eval_pred.predictions and eval_pred.label_ids are numpy arrays
    preds = torch.tensor(eval_pred)   # (N, 1, H, W)
    labels = torch.tensor(label_ids)    # (N, 1, H, W)

    preds = torch.sigmoid(preds)
    mae = (preds - labels).abs().mean().item()

    # IoU@0.5 for quick reference (binarized)
    bin_pred = (preds >= 0.5).float()
    bin_lab  = labels.round()
    inter = (bin_pred * bin_lab).sum(dim=(1, 2, 3))
    union = bin_pred.sum(dim=(1, 2, 3)) + bin_lab.sum(dim=(1, 2, 3)) - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()

    # Optional: soft Dice as a nicer segmentation score
    num = 2 * (preds * labels).sum(dim=(1, 2, 3))
    den = preds.pow(2).sum(dim=(1, 2, 3)) + labels.pow(2).sum(dim=(1, 2, 3)) + 1e-6
    dice = (num / den).mean().item()

    return {"mae": mae, "iou50": iou, "dice": dice}


def _extract_logits(out):
    """
    Dig through dict/list/tuple wrappers and return a torch.Tensor.
    Accepts patterns like:
      - tensor
      - {'logits': tensor} or {'pred': tensor} or {'alpha': tensor}
      - [tensor, ...] or (tensor, ...)
      - [ {'pred': tensor}, ... ]  (pick the last highest-res)
      - {'logits': [tensor_low, tensor_hi]}
    """
    def get_from(obj):
        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, dict):
            for k in ("logits", "pred", "alpha", "out", "y", "out_alpha"):
                if k in obj:
                    t = get_from(obj[k])
                    if t is not None:
                        return t
            # fallthrough: try any value
            for v in obj.values():
                t = get_from(v)
                if t is not None:
                    return t
        if isinstance(obj, (list, tuple)):
            # prefer the last item (often highest-resolution)
            for item in reversed(obj):
                t = get_from(item)
                if t is not None:
                    return t
        return None

    logits = get_from(out)
    if logits is None or not torch.is_tensor(logits):
        raise RuntimeError(
            f"Cannot extract tensor logits from BiRefNet output of type {type(out)}."
        )
    return logits


def save_checkpoint(state, filename="saved_models/checkpoint.pth.tar"):
    """
    Saves the model's state as a checkpoint.

    Parameters:
        state (dict): State of the model to save.
        filename (str, optional): Path to save the checkpoint. Defaults to "saved_models/checkpoint.pth.tar".
    """
    torch.save({"state": state}, filename)


def load_checkpoint(net, optimizer, filename="saved_models/checkpoint.pth.tar") -> int:
    """
    Loads model state from a checkpoint.

    Parameters:
        net (nn.Module): Model architecture.
        optimizer (Optimizer): Optimizer used during training.
        filename (str, optional): Path to the checkpoint. Defaults to "saved_models/checkpoint.pth.tar".

    Returns:
        dict: Counts of training epochs for various augmentations.
    """

    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        net.load_state_dict(checkpoint["state"]["state_dict"])
        if filename != "saved_models/checkpoint.pth.tar":
            optimizer.load_state_dict(checkpoint["state"]["optimizer"])

        # Update the dictionary with values from the checkpoint
        # Only updates keys that exist in both dictionaries
        # This is done for expandability in future
        print("Checkpoint loaded")
        return checkpoint["state"]["epoch"]
    else:
        print(f"No checkpoint file found at '{filename}'. Starting from scratch...")
    print("\n———")

    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval_subset", type=int, default=100)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--num_train_epochs", type=int, default=100)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--fp16", type=lambda x: x.lower() != "false", default=False)
    return ap.parse_args()


def main() -> None:
    torch.backends.cudnn.benchmark = False
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method("spawn")

    pairs = list_pairs(Path(args.data_root))
    # train_pairs, eval_pairs = split_pairs(pairs, eval_size=args.eval_subset, seed=args.seed)
    train_pairs = pairs
    eval_pairs = list_pairs(Path(args.data_root), images_name="eval_images", masks_name="eval_masks")

    train_ds = DataLoader(
        FolderSegDataset(
            train_pairs,
            image_size=args.image_size,
            gaussian_noise_std=0.02,
            gaussian_noise_prob=0.35,
        ),
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=8,
    )
    eval_ds = DataLoader(
        FolderSegDataset(
            eval_pairs,
            image_size=args.image_size,
            gaussian_noise_std=0.0,
            gaussian_noise_prob=0.0,
        ),
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=8,
    )

    cfg = SkinSegFormerConfig(
        num_classes=1,
        enc_dim=384,
        enc_depth=12,
        enc_heads=6,
        dec_dim=256,
        dec_depth=6,
        dec_heads=8,
    )
    model = SkinSegFormer(cfg)
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1E-5,
    )
    epochs_done = load_checkpoint(model, optimizer, filename="saved_models/checkpoint.pth.tar")
    epochs_left = args.num_train_epochs - epochs_done
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs_left, eta_min=1e-6)
    criterion = BCEPlusDice(bce_weight=0.33, dice_weight=0.67)
    use_fp16 = args.fp16

    device = "cuda:1"
    model.to(device)

    for i in range(epochs_left):
        model.train()
        train_progress_bar = tqdm(total=len(train_ds), desc=f"Training [{i + 1}/{epochs_left}]")
        for data in train_ds:
            # inputs = data["pixel_values"]
            inputs = data["pixel_values"].to(device, non_blocking=True)
            # labels = data["labels"].to("cuda:1", non_blocking=True)
            labels = data["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16):
                outputs = model(inputs)
            logits = _extract_logits(outputs)

            loss = criterion(logits, labels)
            loss.backward()
            train_progress_bar.set_postfix({"loss": loss.item()})
            optimizer.step()

            scheduler.step()
            train_progress_bar.update(1)
        train_progress_bar.close()
        model.eval()
        save_checkpoint(
            {
                "epoch": epochs_done + i + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            f"saved_models/checkpoint_{epochs_done + i + 1}.pth.tar",
        )
        eval_progress_bar = tqdm(total=len(eval_ds), desc=f"Evaluating [{i + 1}/{epochs_left}]")
        eval_results = {"mae": 0.0, "iou50": 0.0, "dice": 0.0}
        for data in eval_ds:
            inputs = data["pixel_values"].to("cuda:1", non_blocking=True)
            labels = data["labels"]
            with torch.inference_mode():
                outputs = model(inputs)
            results = compute_seg_metrics(outputs.cpu(), labels)
            eval_results["mae"] += results["mae"]
            eval_results["iou50"] += results["iou50"]
            eval_results["dice"] += results["dice"]
            eval_progress_bar.update(1)
        eval_progress_bar.close()
        eval_results["mae"] = eval_results["mae"] / (len(eval_ds) // args.per_device_eval_batch_size)
        eval_results["iou50"] = eval_results["iou50"] / (len(eval_ds) // args.per_device_eval_batch_size)
        eval_results["dice"] = eval_results["dice"] / (len(eval_ds) // args.per_device_eval_batch_size)
        print(f"Epoch {i + 1 + epochs_done} eval: {eval_results}")


if __name__ == "__main__":
    main()
