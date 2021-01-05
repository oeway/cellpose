import os
import torch
from cellpose import resnet_torch, models, io, transforms
from utils import segment_image
import imageio

import tensorflow as tf

os.makedirs('./data', exist_ok=True)

file_path = './data/test/000_img.png'
image = io.imread(file_path)
channels = [2, 1]
diameter = 30
diam_mean = 30
rescale = diam_mean / diameter

keras_model = tf.keras.models.load_model('./data/keras_model.h5')
def keras_predict(x):
    x = x.transpose([0, 2, 3, 1])
    y, _ = keras_model.predict(x)
    return y.transpose([0, 3, 1, 2])

maski = segment_image(keras_predict, image, channels, rescale=rescale)

imageio.imsave('./data/test_output_keras.png', maski.astype('uint16'))

