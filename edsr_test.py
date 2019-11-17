import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from model import resolve_single
from model.edsr import edsr

from utils import load_image, plot_sample

model = edsr(scale=4, num_res_blocks=16)
model.load_weights('weights/edsr-16-x4/weights.h5')

lr = load_image('demo/DIV2k_valid_LR_difficult/0801x4d.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)