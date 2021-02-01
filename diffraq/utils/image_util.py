"""
image_util.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-28-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Image utility functions to be used by DIFFRAQ.

"""

import numpy as np

##############################################
###		Cropping / Padding / Masking		###
##############################################

def crop_image(img, cen, wid):
    if cen is None:
        cen = np.array(img.shape)[-2:].astype(int)//2

    sub_img = img[..., max(0, int(cen[1] - wid)) : min(img.shape[-2], int(cen[1] + wid)), \
                       max(0, int(cen[0] - wid)) : min(img.shape[-1], int(cen[0] + wid))]

    return sub_img

def pad_array(inarr, num_pad):
    N0 = len(inarr)
    NN = int(N0*num_pad)
    outarr = np.zeros((NN,NN)).astype(inarr.dtype)
    outarr[NN//2-N0//2:NN//2+N0//2+(N0%2),NN//2-N0//2:NN//2+N0//2+(N0%2)] = inarr
    return outarr

def round_aperture(img):
    #Build radius values
    rhoi = get_image_radii(img.shape[-2:])
    rtest = rhoi >= (min(img.shape[-2:]) - 1.)/2.

    #Get zero value depending if complex
    zero_val = 0.
    if np.iscomplexobj(img):
        zero_val += 0j

    #Set electric field outside of aperture to zero (make aperture circular through rtest)
    img[...,rtest] = zero_val

    #Get number of unmasked points
    NN_full = np.count_nonzero(~rtest)

    #cleanup
    del rhoi, rtest

    return img, NN_full

def get_image_radii(img_shp, cen=None):
    yind, xind = np.indices(img_shp)
    if cen is None:
        #TODO: is -1 correct?
        # cen = [(img_shp[-2] - 1)/2, (img_shp[-1] - 1)/2]
        cen = [(img_shp[-2] - 0)/2, (img_shp[-1] - 0)/2]
    return np.hypot(xind - cen[0], yind - cen[1])

##############################################
##############################################
