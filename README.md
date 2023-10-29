# rembg trainer

This code allows you to easily train U2Net model in ONNX format to use with rembg tool

This work is based off U2Net repo, which is licenced under Apache. The derivative work is licenced under MIT; do whatever you want with it.

Default parameters are optimized for systems with 32gb of memory, like Apple M1 Pro. Model is saved every 300 iterations. Adjust as necessary.

How to use:
- Clone this repo
- Install requirements.txt
  - TODO: here's how
  - TODO: cleanup reqs
- Put your images into folder “images”
- Put your masks into folder “masks”; or see below
- Adjust parameters of u2net_train.py and then execute it
- Go grab a nice latte and wait...........

If you already have some images with removed background, then you can create masks off them using supplied alpha.py script. Create directory “clean”, put your pngs there, and launch the script. Be vary: the script is very CPU-heavy.

Basically, then you will have the following folder structure:
- images — source images, will be needed for training
- clean — images with removed background, to extract masks
- masks — required for training, to teach model where the background was
