import argparse
import multiprocessing
import os
from PIL import Image
from typing import List


def resize_and_convert(image_list: List[str], size: int, color: str = "RGB"):
    for image_path in image_list:
        image = Image.open(image_path)
        image = image.resize((size, size))
        image = image.convert(color)
        image.save(image_path)


if __name__ == '__main__':
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
    formats = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr", "LAB", "HSV", "I", "F"}

    parser = argparse.ArgumentParser(
        description="Convert a dataset to reduce excess CPU utilization"
    )
    parser.add_argument(
        "-i",
        "--images",
        type=str,
        default="images",
        help="Directory with training images."
    )
    parser.add_argument(
        "-m",
        "--masks",
        type=str,
        default="masks",
        help="Directory with masks/result images."
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1024,
        help="The (square) size of the resulting images."
    )
    parser.add_argument(
        "--i_color",
        type=str,
        default="RGB",
        help="PIL name of the conversion to do for input images."
    )
    parser.add_argument(
        "--m_color",
        type=str,
        default="L",
        help="PIL name of the conversion to do for mask/result images."
    )
    args = parser.parse_args()

    if args.i_color not in formats:
        print(f"Input color {args.i_color} is not valid")
        exit(1)
    if args.m_color not in formats:
        print(f"Mask/result color {args.m_color} is not valid")
        exit(1)

    img_list = [args.images + os.path.sep + x for x in os.listdir(args.images)]
    lbl_list = [args.masks + os.path.sep + x for x in os.listdir(args.masks)]

    if len(img_list) != len(lbl_list):
        print(f"The number of images ({len(img_list)}) is not equal to the number of masks/results ({len(lbl_list)}). "
              f"Exiting...")
        exit(1)

    num_processes = multiprocessing.cpu_count() - 1
    pool = multiprocessing.Pool(processes=num_processes)

    split_images = [img_list[i:i+num_processes] for i in range(0, len(img_list), num_processes)]
    split_labels = [lbl_list[i:i+num_processes] for i in range(0, len(lbl_list), num_processes)]

    for sub_list in split_images:
        pool.apply_async(resize_and_convert, args=(sub_list, args.size, args.i_color,))
    for sub_list in split_labels:
        pool.apply_async(resize_and_convert, args=(sub_list, args.size, args.m_color,))

    # Start, do the work, and wait for results
    pool.close()
    pool.join()
    print("Done")
