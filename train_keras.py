import os
import torch
from cellpose import utils, dynamics, resnet_torch, models, io, transforms
from utils import load_train_test_data, keras_loss_fn
import imageio
import numpy as np
import tensorflow as tf
import random

os.makedirs('./data', exist_ok=True)

file_path = './data/test/000_img.png'
image = io.imread(file_path)
channels = [2, 1]
diameter = 30
diam_mean = 30
rescale = diam_mean / diameter

model = tf.keras.models.load_model('./data/keras_model.h5')
def keras_predict(x):
    x = x.transpose([0, 2, 3, 1])
    y, _ = model.predict(x)
    return y.transpose([0, 3, 1, 2])

channels = [1, 2]

model.compile("Adam", loss=[keras_loss_fn, None])
train_images, train_labels, train_names, test_images, test_labels, test_names = load_train_test_data('./data/hpa_dataset_v2/train', './data/hpa_dataset_v2/test', ['er.png', 'nuclei.png'], 'cell_masks.png', 1.0)

train_images, train_labels, test_images, test_labels, run_test = transforms.reshape_train_test(train_images, train_labels,
                                                                                                   test_images, test_labels,
                                                                                                   channels, normalize=True)                     
train_flows = dynamics.labels_to_flows(train_labels, files=train_names)
test_flows = dynamics.labels_to_flows(test_labels, files=test_names)

# switch to channel last format

# test_images = [img.transpose([1, 2, 0]) for img in test_images]
# test_flows = [img[1:].transpose([1, 2, 0]) for img in test_flows]
# test_images, test_flows = np.stack(test_images, axis=0), np.stack(test_flows, axis=0)

# train_images = [img.transpose([1, 2, 0]) for img in train_images]
# train_flows = [img[1:].transpose([1, 2, 0]) for img in train_flows]
# train_images, train_flows = np.stack(train_images, axis=0), np.stack(train_flows, axis=0)

batch_size = 3
rescale = False
diam_mean = 30
def generator(indexes):
    nimg = len(indexes)
    # compute average cell diameter
    if rescale:
        diam_train = np.array([utils.diameters(train_labels[k][0])[0] for k in range(len(train_labels))])
        diam_train[diam_train<5] = 5.
        scale_range = 0.5
    else:
        scale_range = 1.0 
    while True:
        for ibatch in range(0, nimg, batch_size):
            inds = indexes[ibatch: ibatch+batch_size]
            rsc = diam_train[inds] / diam_mean if rescale else np.ones(batch_size, np.float32)
            imgi, lbl, scale = transforms.random_rotate_and_resize(
                                    [train_images[i] for i in inds], Y=[train_labels[i][1:] for i in inds],
                                    rescale=rsc, scale_range=scale_range, unet=False)
            yield imgi.transpose([0, 2, 3, 1]), lbl.transpose([0, 2, 3, 1]) 

indexes = list(range(len(train_images)))
random.seed(134)
random.shuffle(indexes)
snapshot_weights = './data/hpa_dataset_v2/keras_model.h5'

# resume training
if os.path.exists(snapshot_weights):
    print('Resuming from previous snapshot: ' + snapshot_weights)
    model.load_weights(snapshot_weights)

callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=snapshot_weights, monitor="loss", save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir='./data/hpa_dataset_v2/logs')]

model.fit(generator(indexes), batch_size=batch_size, epochs=100, steps_per_epoch=1000, callbacks=callbacks)
model.save('./data/hpa_dataset_v2/keras_model_final.h5')