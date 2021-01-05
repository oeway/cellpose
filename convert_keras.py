import os
import torch
import logging
from cellpose import io, resnet_torch
from utils import segment_image

import numpy as np
from torch.autograd import Variable

# Requirements:
# torch==1.7.0
# torchvision==0.8.0
# git+https://github.com/oeway/cellpose@0b1f88b5d297eeb56bbda53ab65efcee1c12a558#egg=cellpose
# git+https://github.com/nerox8664/onnx2keras@b5d77c0920eeefef86ea309d7776452a85ec454b#egg=onnx2keras
# pytorch2keras==0.2.4
# tensorflowjs==2.8.2

from pytorch2keras import pytorch_to_keras

os.makedirs('./data', exist_ok=True)

folder = './data/train/models/'
cpmodel_path = folder+'cellpose_residual_on_style_off_concatenation_off_train_2021_01_04_23_12_23.917462'

net = resnet_torch.CPnet([2, 32, 64, 128, 256], 
                        3, 
                        3,
                        residual_on=True, 
                        style_on=False,
                        concatenation=False,
                        mkldnn=False)
net.load_state_dict(torch.load(cpmodel_path))

def torch_predict(x):
    X = torch.from_numpy(x).float().to('cpu')
    net.eval()
    y, _ = net(X)
    return y.detach().cpu().numpy()


import imageio
file_path = './data/test/000_img.png'
image = io.imread(file_path)
channels = [2, 1]
rescale = 1.0

maski = segment_image(torch_predict, image, channels, rescale=rescale)
imageio.imsave('./data/test_output_torch.png', maski.astype('uint16'))

print('Number of parameters: ' + str(sum(p.numel() for p in net.parameters())))

input_np = np.random.uniform(0, 1, (1, 2, 224, 224))
input_var = Variable(torch.FloatTensor(input_np))

net.to('cpu')

output_pytorch, _ = net(input_var)

k_model = pytorch_to_keras(net, input_var, [(2, None, None)], verbose=True, change_ordering=True)

k_model.summary()
k_model.save('./data/keras_model.h5')

input_np_keras = input_np.transpose([0, 2, 3, 1])

# verify output, only works with GPU
output_keras, _ = k_model.predict(input_np_keras)
output_pytorch = output_pytorch.detach().numpy().transpose([0, 2, 3, 1])

assert np.allclose(output_keras, output_pytorch, atol=1e-5)

model_json = k_model.to_json()
with open("./data/keras_model.json", "w") as json_file:
    json_file.write(model_json)

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(k_model, './data/tfjs/')