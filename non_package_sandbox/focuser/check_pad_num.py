import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from diffraq.utils import image_util
from scipy import fft
from scipy.ndimage import affine_transform
import time

def get_image():

    focal_length = 0.499
    image_distance = focal_length
    wave = [400e-9, 641e-9][1]
    num_pts = 512
    tel_diameter = 5e-3
    image_res = 13e-6/focal_length
    min_padding = 4
    dx0 = tel_diameter / num_pts
    num_img = 512

    targ_NN = num_pts * wave / (tel_diameter * image_res)

    #Make sure we have minimum padding
    tar = int(max(np.ceil(targ_NN), num_pts * min_padding))
    #Get next fasest (even)
    NN, dn = 1, 0
    while (NN % 2) != 0:
        NN = fft.next_fast_len(tar+dn)
        dn += 1
    print('done', NN)
    breakpoint()

#################################################

    pupil = np.ones((num_pts, num_pts)) + 0j

    NN0 = pupil.shape[-1]
    pupil, NN_full = image_util.round_aperture(pupil)

    #Create input plane indices
    et = (np.arange(NN0)/NN0 - 0.5) * NN0
    xx = (np.arange(NN)/NN - 0.5) * NN

    #Get output plane sampling
    dx = wave*image_distance/(dx0*NN)

    #Multiply by propagation kernels (lens and Fresnel)
    pupil *= propagation_kernel(et, dx0, wave, -focal_length)
    pupil *= propagation_kernel(et, dx0, wave, image_distance)

    #Pad pupil
    # pupil = image_util.pad_array(pupil, NN)

    dd = (NN - pupil.shape[-1])//2
    xtra = NN - (pupil.shape[1] + dd*2)
    pupil1 = np.pad(pupil.copy(), (dd, dd))
    pupil2 = np.pad(pupil.copy(), (dd, dd+xtra))

    xx1 = (np.arange(NN-xtra)/(NN-xtra) - 0.5) * (NN-xtra)
    xx2 = (np.arange(NN)/NN - 0.5) * NN


    img1, dx1 = finish_up(pupil1, num_img, targ_NN, NN, dx, xx1, wave, image_distance, NN_full)
    img2, dx2 = finish_up(pupil2, num_img, targ_NN, NN, dx, xx2, wave, image_distance, NN_full)

    # nrun = 10
    #
    # tik = time.perf_counter()
    # for i in range(nrun):
    #     FF = do_FFT(pupil)
    # tok = time.perf_counter()
    # print(f'time: {tok-tik:.2f}')

    # fig, axes = plt.subplots(2, figsize=(7,7), sharex=True,sharey=True)
    # axes[0].imshow(img1)
    # axes[1].imshow(img2)
    plt.figure()
    plt.semilogy(img1[len(img1)//2])
    plt.semilogy(img2[len(img2)//2], '--')

    breakpoint()

def finish_up(pupil, num_img, targ_NN, NN, dx, xx, wave, image_distance, NN_full):

    FF = do_FFT(pupil)

    #Trim back to normal size   #TODO: what is max extent from nyquist?
    # FF = image_util.crop_image(FF, None, self.sim.num_pts//2)

    #Multiply by Fresnel diffraction phase prefactor

    FF *= propagation_kernel(xx, dx, wave, image_distance)

    #Multiply by constant phase term
    FF *= np.exp(1j * 2.*np.pi/wave * image_distance)

    #(not used) Normalize by FFT + normalizations to match Fresnel diffraction
    # FF *= self.dx0**2./(wave*self.image_distance) / NN_full

    #Normalize such that peak is 1. Needs to scale for wavelength relative to other images
    FF /= NN_full

    img = FF.copy()

    #Turn into intensity
    img = np.real(img.conj()*img)

    #Finalize image
    img, dx = finalize_image(img, num_img, targ_NN, NN, dx)

    return img, dx

def propagation_kernel(et, dx0, wave, distance):
    return np.exp(1j * 2.*np.pi/wave * dx0**2. * (et[:,None]**2 + et**2) / (2.*distance))

def do_FFT(MM):
    return fft.ifftshift(fft.fft2(fft.fftshift(MM), workers=-1))

def finalize_image(img, num_img, targ_NN, true_NN, dx):
    #Resample onto theoretical resolution through affine transform
    scaling = true_NN / targ_NN

    #Make sure scaling leads to even number (will lead to small difference in sampling)
    NN = len(img)
    N2 = NN / scaling
    scaling = NN / (N2 - (N2%2))

    #Scale sampling
    dx *= scaling

    #Affine matrix
    affmat = np.array([[scaling, 0, 0], [0, scaling, 0]])
    out_shape = (np.ones(2) * NN / scaling).astype(int)

    #Do affine transform
    img = affine_transform(img, affmat, output_shape=out_shape, order=5)

    #Crop to match image size
    img = image_util.crop_image(img, None, num_img//2)

    return img, dx

get_image()

breakpoint()
