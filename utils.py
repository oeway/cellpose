import numpy as np
import cv2
from cellpose import models, utils, dynamics, transforms

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