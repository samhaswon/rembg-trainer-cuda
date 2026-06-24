import os

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    image_list = [x for x in os.listdir("/home/samuel/da/mq_data/images")]
    mask_list = [x for x in os.listdir("/home/samuel/da/mq_data/masks")]
    output_path = "/home/samuel/da/mq_data/alphas/"

    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    assert len(image_list) == len(mask_list)
    # image_list = image_list[500:510]
    # mask_list = mask_list[500:510]

    print(f"Found {len(image_list)} images and {len(mask_list)} masks")

    for mask_path in tqdm(
            mask_list, desc="Alpha processing", total=len(image_list)
    ):
        pil_mask = Image.open(f"/home/samuel/da/mq_data/masks/{mask_path}")
        if pil_mask.mode != "LA":
            pil_mask = pil_mask.convert(mode="LA")
        mask = np.array(pil_mask)
        pil_mask.close()

        alpha = mask[..., 1]
        cv2.imwrite(output_path + mask_path, alpha)
