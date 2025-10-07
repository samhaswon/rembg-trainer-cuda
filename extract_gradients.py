import math
import os

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    image_list = [x for x in os.listdir("D:/mq_data/images")]
    mask_list = [x for x in os.listdir("D:/mq_data/masks")]
    output_path = "D:/mq_data/gradients/"

    assert len(image_list) == len(mask_list)
    # image_list = image_list[500:510]
    # mask_list = mask_list[500:510]

    print(f"Found {len(image_list)} images and {len(mask_list)} masks")

    for image_path, mask_path in tqdm(
            zip(image_list, mask_list), desc="Mask processing", total=len(image_list)
    ):
        image = cv2.imread(f"D:/mq_data/images/{image_path}")
        pil_mask = Image.open(f"D:/mq_data/masks/{mask_path}")
        if pil_mask.mode != "LA":
            pil_mask = pil_mask.convert(mode="LA")
        mask = np.array(pil_mask).astype(np.int16)
        pil_mask.close()

        if image.shape[2] == 4:
            image = image[..., :3]
        k = int(math.log10(max(image.shape))) * 3
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(image, ksize=(k, k), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
        value_channel = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)[..., 2].astype(np.uint16)
        gradient = np.where(mask[..., 0] < value_channel, mask[..., 0], 255)
        gradient = (255 - gradient) * (np.where(mask[..., 1] > 128, mask[..., 1], 0) / 255.0)
        gradient = np.clip(gradient, 0, 255).astype(np.uint8)
        cv2.imwrite(output_path + image_path, gradient)

        # Test partial TF
        # hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # hsv_img[..., 2] = np.clip(hsv_img[..., 2].astype(np.int16) - gradient, 0, 255).astype(np.uint8)
        # cv2.imwrite(output_path + "t" + img_path, cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR))
