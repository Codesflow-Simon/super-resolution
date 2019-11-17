# super-resolution
A Tensorflow 2.0 based implementation of

Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR), winner of the NTIRE 2017 super-resolution challenge.
Wide Activation for Efficient and Accurate Image Super-Resolution (WDSR), winner of the NTIRE 2018 super-resolution challenge (realistic tracks).
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN).
A DIV2K data provider automatically downloads DIV2K training and validation images of given scale (2, 3, 4 or 8) and downgrade operator ("bicubic", "unknown", "mild" or "difficult").

# Environment setup
`conda env create -f environment.yml`

`conda activate sisr`

# Testing
The models can be tested using the xxxx_test.py scripts.
There are some test images in `/demos`. The complete dataset can be downloaded at https://data.vision.ee.ethz.ch/cvl/DIV2K/.
The path to the image should be passed into the `load_image` function in the test file.
