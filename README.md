# rembg trainer

This code allows you to easily train U2Net model in ONNX format to use with rembg tool

This work is based off U2Net repo, which is licenced under Apache. The derivative work is licenced under GPL v something.

Default parameters are optimized for systems with 32gb of memory, like Apple M1 Pro. Model is saved every 300 iterations. Adjust as necessary.

How to use:
Clone this repo
Install requirements.txt
  TODO: here's how
  TODO: cleanup reqs
Put your images into folder images
Run alpha.sh to generate masks (or supply your own masks; or create some similar stuff if you're on Windows)
  TODO: I'll rewrite this in Python later for convenience
Adjust parameters of u2net_train.py and then execute it
Go grab a nice latte and wait...........
