import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from model import resolve_single
from model.wdsr import wdsr_b
from utils import load_image, plot_sample

model = wdsr_b(scale=4, num_res_blocks=32)
model.load_weights('weights/wdsr-b-32-x4/weights.h5')

lr = load_image('demo/DIV2k_valid_LR_difficult/0801x4d.png')
sr = resolve_single(model, lr)

plot_sample(lr, sr)
