# rembg trainer

This code allows you to easily train U2-Net model in [ONNX](https://github.com/onnx/onnx) format to use with [rembg](https://github.com/danielgatis/rembg]) tool.

This work is based off [U2Net](https://github.com/xuebinqin/U-2-Net) repo, which is under Apache licence. The derivative work is loicensed under MIT; do as you please with it.

A couple of notes on performance:

- Default parameters are fine-tuned for maximum performance on systems with 32gb of processing memory, like the Apple M1 Pro. Adjust accordingly.
- Computations are performed in float32, because float16 support on Metal is a bit undercooked at the moment.
- If this is your first time using CUDA on Windows, you'd have to install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- For CUDA, you can easily rewrite this code with half precision calculations for increased performance. Apex library can help you with that; I don't have such plans at the moment.
- For acceleration on AMD GPUs, please refer to installation guide of [AMD ROCm platform](https://rocm.docs.amd.com/en/latest/how_to/pytorch_install/pytorch_install.html). No code changes will be required.

If the training is interrupted for any reason, don't worry — the program saves its state regularly, allowing you to resume from where you left off. Frequency of saving can be adjusted.

## Fancy a go?

- Download the latest release
- Install `requirements.txt`
- Put your images into `images` folder
- Put their masks into `masks` folder; or see [below](#mask-extraction)
- Launch `python3 u2net_train.py --help` for more details on supported command line flags
- Launch script with your desired configuration
- Go grab yourself a [nice latte](https://www.youtube.com/shorts/h75W1uhL-iQ) and wait........... and wait.....
- Once you've had your fill of waiting, here's how you use resulting model with rembg:

```bash
rembg p -w input output -m u2net_custom -x '{"model_path": "/saved_models/u2net/27.onnx"}'
# input — folder with images to have their backgrounds removed
# output — folder for resulting images processed with custom model
# adjust path(s) as necessary!
```

## Mask extraction

If you already have a bunch of images with removed background, then you can create masks off them using the provided `alpha.py` script. Create a directory called `clean`, put your pngs there, and launch the script.

But fair warning mate: the script is very CPU-heavy. Oh, and you'll need the `ImageMagick` tool installed and present in your PATH.

So, at the end of the day, you will end up with the following folder structure:

- `images` — source images, will be needed for training
- `masks` — required for training, to teach model where the background was
- `clean` — images with removed background, to extract masks (they're not used for actual training)

## Leave your mark 👉👈🥺

Buy me ~~a coffee~~ an alcohol-free cider [here](http://buymeacoffee.com/jonathunky)
