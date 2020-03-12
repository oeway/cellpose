import numpy as np
import time, os, sys
import mxnet as mx

from skimage.io import imread
import glob
import sys
from cellpose import models, utils

# check if GPU working, and if so use it
use_gpu = utils.use_gpu()
if use_gpu:
    device = mx.gpu()
else:
    device = mx.cpu()

# model_type='cyto' or model_type='nuclei'
model = models.Cellpose(device, model_type='cyto', net_avg=True)

# list of files
files = ['./images/221_G2_1_blue_red_green.jpg']

imgs = [imread(f) for f in files]
nimg = len(imgs)

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
channels = [[1, 3]]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# if rescale is set to None, the size of the cells is estimated on a per image basis
# if you want to set the size yourself, set it to 30. / average_cell_diameter
masks, flows, styles, diams = model.eval(imgs, rescale=0.2, channels=channels, net_avg=False)



import matplotlib.pyplot as plt
from cellpose import plot, transforms

for idx in range(nimg):
    img = transforms.reshape(imgs[idx], channels[idx])
    img = plot.rgb_image(img)
    maski = masks[idx]
    flowi = flows[idx][0]

    fig = plt.figure(figsize=(12,3))
    # can save images (set save_dir=None if not)
    plot.show_segmentation(fig, img, maski, flowi)
    plt.tight_layout()
    plt.savefig('output.png')