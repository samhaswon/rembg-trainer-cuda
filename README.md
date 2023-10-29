# rembg trainer

This code allows you to easily train U2-Net model in [ONNX](https://github.com/onnx/onnx) format to use with [rembg](https://github.com/danielgatis/rembg]) tool

This work is based off [U2Net](https://github.com/xuebinqin/U-2-Net) repo, which is under Apache loicence. The derivative work is licenced under MIT; do as you please with it.

Default parameters are fine-tuned for systems with 32gb of memory, like the Apple M1 Pro. Model is saved every 300 iterations. Give it a tweak if you need to.

## Fancy a go?

- Clone this repo
- Install `requirements.txt`
- Put your images into `images` folder
- Put your masks into `masks` folder; or see [below](#mask-generation)
- Tinker with the settings of `u2net_train.py` and then give it a whirl.
  - TODO: ability to continue off where you've finished last time
- Go grab yourself a [nice latte](https://www.youtube.com/shorts/h75W1uhL-iQ) and wait........... and wait.....
- Once you've had your fill of waiting, here's how you use resulting model with rembg:

```bash
rembg p -w input output -m u2net_custom -x '{"model_path": "/saved_models/u2net/2700.onnx"}'
# input — folder with images to have their backgrounds removed
# output — folder for resulting images processed with custom model
# adjust path(s) as necessary!
```

## Mask extraction

If you already have a bunch of images with removed background, then you can create masks off them using the provided `alpha.py` script. Create a directory called `clean`, put your pngs there, and launch the script.

But fair warning mate: the script is very CPU-heavy. Oh, and you'll need the ImageMagick tool installed and present in your PATH.

So, at the end of the day, you will have the following folder structure:

- `images` — source images, will be needed for training
- `clean` — images with removed background, to extract masks
- `masks` — required for training, to teach model where the background was
