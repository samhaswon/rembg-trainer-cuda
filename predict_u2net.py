import os
import time
from typing import Union, Tuple, List, Optional, Any

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage  # For typing
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from model import BiRefNet
from model.u2net import U2NET, U2NETP

DEVICE = 'cuda'
# MODEL = "u2net"
# MODEL = "u2netp"
MODEL = "birefnet"

IMAGE_SIZE: int = 1728


class BiRefNetHFAdapter(nn.Module):
    """
    Wraps the base model to present {'logits': (N, 1, H, W)}.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.base(pixel_values)
        return out[0]


def resize_and_pad_square(x: torch.Tensor, target: int) -> Tuple[torch.Tensor, int, int]:
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
    return x, pad_h, pad_w


def post_process(mask: np.ndarray) -> np.ndarray:
    """
    Morphs and blurs the mask to make it a bit better (generally speaking).
    :param mask: The mask to post-process.
    :return: The post-processed mask.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
    return mask


class TorchSessionU2NET:
    """
    Session for Torch inference with post-processing.
    """

    def __init__(self, half_precision: bool = False, use_small: bool = False):
        if use_small:
            self.net = U2NETP(3, 1)
            model_path = "./u2netp.pth"
            self.input_size = [512, 512]
        else:
            self.net = U2NET(3, 1)
            model_path = "./u2net.pth"
            self.input_size = [1024, 1024]
        if torch.cuda.is_available():
            self.net.load_state_dict(
                torch.load(model_path, weights_only=False)
            )
            self.net.cuda()
        else:
            self.net.load_state_dict(
                torch.load(
                    model_path,
                    map_location=torch.device(DEVICE),
                    weights_only=False
                )
            )
        self.net.eval()
        self.half_precision = half_precision

    def remove(
            self,
            img: PILImage,
            size: Union[Tuple[int, int], None] = None,
            mask_only: bool = False
    ) -> np.ndarray:
        """
        Runs inferencing with post-processing.
        :param img: The image to be processed.
        :param size: Unused, for compatibility with other onnx code.
        :param mask_only: If True, it returns only the mask.
        :return: Either the mask (L or A) or the original image with the
        alpha channel applied (RGBA).
        """
        image_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1)
        image_tensor = F.interpolate(
            torch.unsqueeze(image_tensor, 0), self.input_size, mode="bilinear").type(torch.float32)
        image_tensor = torch.divide(image_tensor, torch.max(image_tensor)).type(torch.float32)
        image_tensor = image_tensor.to(DEVICE)
        with (torch.no_grad(),
              torch.autocast(
                  device_type=DEVICE,
                  dtype=torch.float16,
                  enabled=self.half_precision)
              ):
            img_result = self.net(image_tensor)
        img_result = img_result[0][:, 0, :, :]
        result_array = img_result.cpu().data.numpy()

        # Norm the prediction
        re_max = np.max(result_array)
        re_min = np.min(result_array)
        result_array = (result_array - re_min) / (re_max - re_min)
        result_array = np.squeeze(result_array)

        alpha_channel = np.uint8(result_array * 255)
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)
        alpha_channel = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))


class TorchSessionBiRefNet:
    """
    Session for Torch inference with post-processing.
    """

    def __init__(self, net: torch.nn.Module, half_precision: bool = False, use_small: bool = False):
        self.net = net
        self.net.eval()
        self.half_precision = half_precision

    def remove(
            self,
            img: PILImage,
            size: Optional[Tuple[int, int]] = None,
            mask_only: bool = False
    ) -> np.ndarray[Any, Tuple[np.uint8]]:
        """
        Runs inferencing with post-processing.
        :param img: The image to be processed.
        :param size: Unused, for compatibility with other onnx code.
        :param mask_only: If True, it returns only the mask.
        :return: Either the mask (L or A) or the original image with the
        alpha channel applied (RGBA).
        """
        if size is None:
            size = IMAGE_SIZE, IMAGE_SIZE
        image_tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
        image_tensor, pad_h, pad_w = resize_and_pad_square(image_tensor, size[0])
        image_tensor = image_tensor.to(DEVICE)
        image_tensor = image_tensor.unsqueeze(0)
        with (torch.inference_mode(),
              torch.autocast(
                  device_type=DEVICE,
                  dtype=torch.float16,
                  enabled=self.half_precision)
              ):
            img_result = self.net(image_tensor)
        img_result = img_result[..., :IMAGE_SIZE - pad_h, :IMAGE_SIZE - pad_w]
        img_result = torch.sigmoid(img_result)
        img_result = torch.where(img_result > 0.5, img_result, torch.tensor(0.0))
        # img_result = (img_result > 0.5).to(torch.float32)
        result_array = img_result.cpu().data.numpy()

        # Norm the prediction
        result_array = np.squeeze(result_array)
        alpha_channel = np.uint8(result_array * 255)
        alpha_channel = cv2.resize(alpha_channel, img.size, interpolation=cv2.INTER_LANCZOS4)
        alpha_channel = cv2.morphologyEx(alpha_channel, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        alpha_channel = cv2.GaussianBlur(alpha_channel, (3, 3), 0)
        if mask_only:
            return alpha_channel
        return np.dstack((np.array(img), alpha_channel))


if __name__ == '__main__':
    image_list = [x.path for x in os.scandir("./images")]
    image_list.sort()
    image_list = image_list[:1134]

    if not os.path.exists(MODEL):
        os.makedirs(MODEL)

    if MODEL == "u2net":
        session = TorchSessionU2NET(use_small=False)
    elif MODEL == "u2netp":
        session = TorchSessionU2NET(use_small=True)
    elif MODEL == "birefnet":
        checkpoint = torch.load("./checkpoint_14.pth.tar", map_location='cpu', weights_only=False)
        net = BiRefNet()
        net.load_state_dict(checkpoint["state"]["state_dict"])

        model = BiRefNetHFAdapter(net).to(DEVICE)
        session = TorchSessionBiRefNet(net=model, half_precision=False)
    else:
        raise ValueError("Unknown model")

    for image_path in tqdm(
            image_list,
            desc="Inferencing",
            total=len(image_list)
    ):
        pil_image = Image.open(image_path)
        if pil_image.mode == "RGBA":
            pil_image = pil_image.convert(mode="RGB")
        prediction = np.array(session.remove(pil_image, mask_only=True))
        pil_image.close()
        cv2.imwrite(f"./{MODEL}/" + image_path.split("/")[-1], prediction)
