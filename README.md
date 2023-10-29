# rembg trainer

This code allows you to easily train U2-Net model in [ONNX](https://github.com/onnx/onnx) format to use with [rembg](https://github.com/danielgatis/rembg]) tool

This work is based off [U2Net](https://github.com/xuebinqin/U-2-Net) repo, which is under Apache loicence. The derivative work is licenced under MIT; do whatever you want with it.

Default parameters are optimized for systems with 32gb of memory, like Apple M1 Pro. Model is saved every 300 iterations. Adjust as necessary.

## How to use

- Clone this repo
- Install requirements.txt
- Put your images into “images” folder
- Put your masks into “masks” folder; or see [below](#mask-generation)
- Adjust parameters of u2net_train.py and then execute it
  - TODO: ability to continue off where you've finished last time
- Go grab yourself a [nice latte](https://www.youtube.com/shorts/h75W1uhL-iQ) and wait........... and wait.....
- After you've done waiting, run rembg like this:

```bash
rembg p -w input output -m u2net_custom -x '{"model_path": "/saved_models/u2net/2700.onnx"}'
# input — folder with images to have their backgrounds removed
# output — folder for resulting images
# adjust path(s) as necessary!
```

## Mask extraction

If you already have some images with removed background, then you can create masks off them using supplied alpha.py script. Create directory “clean”, put your pngs there, and launch the script. Be vary: the script is very CPU-heavy. It also requires ImageMagick tool to be installed and present in PATH.

Basically, then you will have the following folder structure:

- images — source images, will be needed for training
- clean — images with removed background, to extract masks
- masks — required for training, to teach model where the background was
