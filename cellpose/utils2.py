import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from skimage import measure, transform
from skimage.io import imsave
from cellpose import io, models, utils, dynamics, transforms
from imgseg import annotationUtils

def run_tiled(predict, imgi, augment=False, bsize=224, tile_overlap=0.1, batch_size=1, nclasses=3):
    IMG, ysub, xsub, Ly, Lx = transforms.make_tiles(imgi, bsize=bsize, 
                                                    augment=augment, tile_overlap=tile_overlap)
    ny, nx, nchan, ly, lx = IMG.shape
    IMG = np.reshape(IMG, (ny*nx, nchan, ly, lx))
    niter = int(np.ceil(IMG.shape[0] / batch_size))
    y = np.zeros((IMG.shape[0], nclasses, ly, lx))
    for k in range(niter):
        irange = np.arange(batch_size*k, min(IMG.shape[0], batch_size*k+batch_size))
        y0 = predict(IMG[irange])
        y[irange] = y0.reshape(len(irange), y0.shape[-3], y0.shape[-2], y0.shape[-1])
    if augment:
        y = np.reshape(y, (ny, nx, nclasses, bsize, bsize))
        y = transforms.unaugment_tiles(y, False)
        y = np.reshape(y, (-1, nclasses, bsize, bsize))
    
    yf = transforms.average_tiles(y, ysub, xsub, Ly, Lx)
    yf = yf[:,:imgi.shape[1],:imgi.shape[2]]
    return yf
        
def segment_image(predict, image, channels, rescale=1.0, augment=False, tile=True, tile_overlap=0.1, bsize=224): 
    images, nolist = models.convert_images(image, channels, False, True, False)
    image = images[0]
    input_shape = image.shape
    if rescale != 1.0:
        # scale to match diameter
        image = transforms.resize_image(image, rsz=rescale)
    # make image nchan x Ly x Lx for net
    imgs = np.transpose(image, (2,0,1))
    detranspose = (1,2,0)

    # pad image for net so Ly and Lx are divisible by 4
    imgs, ysub, xsub = transforms.pad_image_ND(imgs)
    # slices from padding
    slc = [slice(0, imgs.shape[n]+1) for n in range(imgs.ndim)]
    slc[-2] = slice(ysub[0], ysub[-1]+1)
    slc[-1] = slice(xsub[0], xsub[-1]+1)
    slc = tuple(slc)

    # run network
    y = run_tiled(predict, imgs, augment=False, bsize=bsize, tile_overlap=tile_overlap)

    # slice out padding
    y = y[slc]

    # transpose so channels axis is last again
    y = np.transpose(y, detranspose)
    
    maski = convert_mask(y, input_shape)

    if rescale != 1:
        maski = transforms.resize_image(maski, rsz=1.0/rescale)
    return maski


def convert_mask(y, input_shape, resample = True, cellprob_threshold = 0,niter=200, interp=True, use_gpu = False, flow_threshold = 0.4):
    if resample:
        y = transforms.resize_image(y, input_shape[-3], input_shape[-2])
    cellprob = y[:,:,-1]
    dP = y[:,:,:2].transpose((2,0,1))

    p = dynamics.follow_flows(-1 * dP * (cellprob > cellprob_threshold) / 5., 
                                niter=niter, interp=interp, use_gpu=use_gpu)

    maski = dynamics.get_masks(p, iscell=(cellprob>cellprob_threshold),
                                flows=dP, threshold=flow_threshold)
    maski = utils.fill_holes_and_remove_small_masks(maski)
    
    maski = transforms.resize_image(maski, input_shape[-3], input_shape[-2], 
                                    interpolation=cv2.INTER_NEAREST)
    return maski

def read_image(path, rescale=1.0, Lx=None, Ly=None):
    image = io.imread(path)
    
    # the first dimension is channel
    if len(image.shape) == 2:
        image = image[None, :,:]
    if rescale != 1.0 or (Lx is not None and Ly is not None):
        if rescale != 1.0:
            Lx = int(rescale*image.shape[1])
            Ly = int(rescale*image.shape[2])
        if image.shape[1] != Lx or image.shape[2] != Ly:
            dtype = image.dtype
            image = cv2.resize(image.transpose(1, 2, 0), (Lx, Ly)).transpose(2, 0, 1)
            # if dtype == np.uint8:
            #     image = (image * 255).astype('uint8')
            # elif dtype == np.uint16:
            #     image = (image * 65535).astype('uint16')
    return image

def get_image_folders(train_dir, image_channels, rescale=1.0):
    assert image_channels is not None
    subfolders = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    samples = []
    for folder in subfolders:
        for ch in image_channels:
            fch = glob.glob(folder + '/*%s*'%ch)
            if len(fch) != 1:
                continue
        samples.append(folder)
    return samples

def read_multi_channel_image(folder, image_channels, rescale=1.0):
    chs = []
    for ch in image_channels:
        fch = glob.glob(folder + '/*%s*'%ch)
        if len(fch) != 1:
            continue
        else:
            image = read_image(fch[0], rescale)
            chs.append(image)
    if len(chs) == 0:
        return None
    return np.concatenate(chs, axis=0)

def get_samples(train_dir, image_channels, mask_filter, rescale=1.0):
    assert image_channels is not None
    subfolders = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    image_names = []
    images = []
    label_names = []
    labels = []
    for folder in subfolders:
        chs = []
        for ch in image_channels:
            fch = glob.glob(folder + '/*%s*'%ch)
            if len(fch) != 1:
                continue
            else:
                image = read_image(fch[0], rescale)
                chs.append(image)
        maskfs = glob.glob(folder + '/*%s*'%mask_filter)
        if len(maskfs) != 1:
            annotation_file = os.path.join(folder, 'annotation.json')
            if os.path.exists(annotation_file):
                try:
                    # try to generate from annotation.json
                    geojson_to_label(annotation_file, save_as='_masks.png')
                    maskfs = glob.glob(folder + '/*_masks.png')
                    assert len(maskfs) >= 1
                except Exception as e:
                    print('===ERROR===>', folder, e)
                    raise e
            else:
                continue

        # Assuming we have grayscale images
        img = np.concatenate(chs, axis=0)
        images.append(img)
        image_names.append(folder + '/image')

        label_names.append(maskfs[0])
        image = read_image(maskfs[0], rescale)
        labels.append(image)
        

    flow_names = [n+ '_flows.tif' for n in image_names]

    if not all([os.path.exists(flow) for flow in flow_names]):
        flow_names = None
    else:
        for n in range(len(flow_names)):
            flows = read_image(flow_names[n], rescale)
            # detect size mismatch
            if flows.shape[1] != images[n].shape[1] or flows.shape[2] != images[n].shape[2]:
                continue
            if flows.shape[0]<4:
                labels[n] = np.concatenate((labels[n][np.newaxis,:,:], flows), axis=0) 
            else:
                labels[n] = flows

    return images, labels, image_names

def load_train_test_data(train_dir, test_dir, image_channels, label_filter, rescale=1.0, unet=False):

    # training data
    images, labels, image_names = get_samples(train_dir, image_channels, label_filter, rescale=rescale)
            
    # testing data
    test_images, test_labels, image_names_test = None, None, None
    if test_dir is not None:
        test_images, test_labels, image_names_test = get_samples(test_dir, image_channels, label_filter, rescale=rescale)
    return images, labels, image_names, test_images, test_labels, image_names_test

def geojson_to_label(file_open, save_as='_masks.png'):
    annotationsImporter = annotationUtils.GeojsonImporter()
    annot_dict_all, roi_size_all, image_size = annotationsImporter.load(file_open)

    assert image_size is not None

    annot_types = set(
        annot_dict_all[k]["properties"]["label"] for k in annot_dict_all.keys()
    )
    sample_folder, file = os.path.split(file_open)

    for annot_type in annot_types:
        file_name_save = os.path.join(sample_folder, annot_type + save_as)

        # Filter the annotations by label
        annot_dict = {
            k: annot_dict_all[k]
            for k in annot_dict_all.keys()
            if annot_dict_all[k]["properties"]["label"] == annot_type
        }

        # Create masks
        binaryMasks = annotationUtils.BinaryMaskGenerator(
            image_size=image_size, erose_size=5, obj_size_rem=500, save_indiv=True
        )
        mask_dict = binaryMasks.generate(annot_dict)

        mask = mask_dict['fill']
        labels = measure.label(mask)
        imsave(file_name_save, labels)


xent = tf.compat.v2.losses.BinaryCrossentropy(from_logits=True, reduction=tf.compat.v2.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

def keras_loss_fn(lbl, y):
    """ loss function between true labels lbl and prediction y """
    veci = 5. * lbl[:, :, :, 1:]
    lbl  = lbl[:,:, :, 0]>.5
    loss = tf.keras.backend.mean(tf.keras.backend.square(y[:,:, :, :2] - veci )) / 2.0
    # loss2 = xent(y[:, :, :, -1] , lbl)
    # loss = loss + loss2
    return loss
