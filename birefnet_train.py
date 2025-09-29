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
python train_birefnet_lite2k.py \
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
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoModelForImageSegmentation,
    AutoModel,
    AutoConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


class BiRefNetHFAdapter(nn.Module):
    """
    Adapts ZhengPeng7/BiRefNet_lite-2K to HF Trainer conventions.

    - Accepts {'pixel_values': tensor} from the collator
    - Calls underlying model as base(x)
    - Returns {'logits': tensor} where tensor is (N, 1, H, W)
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.proj = None  # lazily built if base outputs != 1 channel

    def _extract_logits(self, out):
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

    def forward(self, pixel_values=None, **kwargs):
        if pixel_values is None and "x" not in kwargs:
            raise TypeError("Expected 'pixel_values' or 'x' tensor.")
        x = pixel_values if pixel_values is not None else kwargs.pop("x")
        out = self.base(x)
        logits = self._extract_logits(out)

        # normalize to shape (N, 1, H, W)
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)
        if logits.shape[1] != 1:
            if self.proj is None:
                self.proj = nn.Conv2d(logits.shape[1], 1, kernel_size=1, bias=True)
                nn.init.zeros_(self.proj.bias)
                nn.init.xavier_uniform_(self.proj.weight)
            logits = self.proj(logits)

        return {"logits": logits}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    List (image, mask) pairs by matching filenames in two folders.

    Parameters
    ----------
    root : Path
        Root folder containing 'images' and 'masks'.

    Returns
    -------
    list[tuple[Path, Path]]
        Sorted list of pairs.
    """
    img_dir = root / "images"
    msk_dir = root / "masks"
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

    def __init__(self, pairs: List[Tuple[Path, Path]], image_size: int) -> None:
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ip, mp = self.pairs[idx]
        img = load_image(ip)
        msk = load_mask(mp)

        # ensure target is multiple of 32 once, at startup or here
        target = ( (self.image_size + 31) // 32 ) * 32

        img_sq = resize_and_pad_square(img, target)
        # masks with nearest, same final size
        msk_sq = F.interpolate(msk[None], size=img_sq.shape[-2:], mode="nearest")[0].clamp(0, 1)

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
        self.bce = nn.BCELoss()

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


class SegTrainer(Trainer):
    """
    Trainer subclass with custom loss and metrics for soft masks.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.criterion = BCEPlusDice()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # Try common keys; BiRefNet HF head uses AutoModelForImageSegmentation mapping.
        logits = outputs.get("logits", None)
        if logits is None:
            # Fallback: many custom models return a list of multi-scale outputs
            # Pick the first or last map as the supervision signal.
            if isinstance(outputs, (list, tuple)):
                logits = outputs[-1]
            else:
                # Try attribute access
                logits = getattr(outputs, "pred", None)
        if logits is None:
            raise RuntimeError("Cannot find segmentation logits in model outputs.")
        loss = self.criterion(logits, labels)
        return (loss, {"logits": logits}) if return_outputs else loss


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


def compute_seg_metrics(eval_pred) -> Dict[str, float]:
    """
    Compute MAE and IoU@0.5 for quick feedback.

    Parameters
    ----------
    eval_pred
        Tuple(logits, labels).

    Returns
    -------
    dict
        Metrics dict.
    """
    logits, labels = eval_pred
    if isinstance(logits, (list, tuple)):
        logits = logits[-1]
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    preds = torch.sigmoid(logits)
    mae = (preds - labels).abs().mean().item()

    # IoU@0.5 on binarized maps, just for sanity checks
    bin_pred = (preds >= 0.5).float()
    inter = (bin_pred * labels.round()).sum(dim=(1, 2, 3))
    union = bin_pred.sum(dim=(1, 2, 3)) + labels.round().sum(dim=(1, 2, 3)) - inter
    iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()

    return {"mae": mae, "iou50": iou}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--image_size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--eval_subset", type=int, default=100)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--num_train_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--fp16", type=lambda x: x.lower() != "false", default=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    pairs = list_pairs(Path(args.data_root))
    train_pairs, eval_pairs = split_pairs(pairs, eval_size=args.eval_subset, seed=args.seed)

    train_ds = FolderSegDataset(train_pairs, image_size=args.image_size)
    eval_ds = FolderSegDataset(eval_pairs, image_size=args.image_size)

    # Load model, trust remote code since repo provides custom PreTrainedModel mapping.
    # Config auto-maps to BiRefNet and AutoModelForImageSegmentation.  :contentReference[oaicite:2]{index=2}
    base_model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet_lite-2K",
        trust_remote_code=True
    )
    model = BiRefNetHFAdapter(base_model)

    # after building `model` (the adapter)
    with torch.no_grad():
        dummy = torch.zeros(2, 3, 256, 256).to(next(model.parameters()).device)
        out = model(pixel_values=dummy)
        assert isinstance(out, dict) and torch.is_tensor(out["logits"]), type(out)
        assert out["logits"].shape[:2] == (2, 1), out["logits"].shape

    # TrainingArguments
    total_bs = args.per_device_train_batch_size
    grad_accum = max(1, math.floor(2 / total_bs))  # mild default normalize to ~2 effective batch
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=args.fp16,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        logging_steps=50,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to=[],
    )

    trainer = SegTrainer(
        model=model,
        args=training_args,
        data_collator=Collator(),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_seg_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
