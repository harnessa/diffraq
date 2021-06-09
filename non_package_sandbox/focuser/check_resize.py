import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from diffraq.utils import image_util
from scipy import fft
from scipy.ndimage import affine_transform, shift
from PIL import Image
import cv2
import time
import h5py

def run_sim():
    wave = [400e-9, 641e-9, 725e-9][2]
    num_pts = 512
    tel_diameter = 5e-3
    min_padding = 4
    dx0 = tel_diameter / num_pts
    num_img = 512

    down_sample = [False, True][1]

    msamp = 2.1

    if down_sample:
        NN = num_img * min_padding * int(np.ceil(msamp))    #number to run (make sure target is also min padded for truth calc)
        targ_NN = NN / msamp                                #Where we want to end at
    else:
        NN = num_img  * min_padding
        targ_NN = NN  * msamp

    #new image res
    image_res = num_pts * wave / (tel_diameter * targ_NN)

    focal_length = 13e-6/image_res
    image_distance = focal_length

    print(NN, targ_NN)

    #################################################

    #Get pupil
    # pupil = np.ones((num_pts, num_pts)) + 0j
    mask = 'm12p9'
    with h5py.File(f'./pupil__{mask}__mask_1a.h5', 'r') as f:
        pupil = f['field'][3]

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

    #Save for later
    tru_pupil = pupil.copy()

    #Pad pupil
    pupil = image_util.pad_array(pupil, NN)
    xx = (np.arange(NN)/NN - 0.5) * NN

    #Propagate
    img = do_prop(pupil, num_img, dx, xx, wave, image_distance, NN_full)

    #Finalize AFFINE
    tik1 = time.perf_counter()
    aimg, adx = finalize_aff(img.copy(), num_img, targ_NN, NN, dx)
    tok1 = time.perf_counter()
    print(f'Aff: {tok1-tik1:.3f}')

    #Finalize PIL
    tik2 = time.perf_counter()
    cimg, cdx = finalize_cv2(img.copy(), num_img, targ_NN, NN, dx)
    tok2 = time.perf_counter()
    print(f'CV2: {tok2-tik2:.3f}')

    #Get integral targ_NN
    Ntt = int(np.round(targ_NN))
    Ntt += Ntt % 2

    #Get truth (not really truth with fractional targ_NN b/c we must make integer)
    tru_pupil = image_util.pad_array(tru_pupil, Ntt)
    dxtru = wave*image_distance/(dx0*Ntt)
    xxtru = (np.arange(Ntt)/Ntt - 0.5) * Ntt
    irdx = image_res * image_distance   #True image dx

    #Propagate tru image
    tru = do_prop(tru_pupil, num_img, dxtru, xxtru, wave, image_distance, NN_full)
    tru = image_util.crop_image(tru, None, num_img//2)

    #Difference
    adif = np.abs(tru - aimg).sum()
    cdif = np.abs(tru - cimg).sum()

    print(f'Adif: {adif:.3e}, Cdif: {cdif:.3e}')
    print(f'Adx: {adx-irdx:.3e}, Cdx: {cdx-irdx:.3e}, True dx: {dxtru-irdx:.3e}')

    #Plot
    pxx = xx[NN//2-len(aimg)//2:NN//2+len(aimg)//2]
    pxt = xxtru[Ntt//2-len(tru)//2:Ntt//2+len(tru)//2]
    plt.figure()
    plt.semilogy(pxx, aimg[len(aimg)//2], '-',  label='AFF')
    plt.semilogy(pxx, cimg[len(cimg)//2], '--', label='PIL')
    plt.semilogy(pxt, tru[len(tru)//2], ':', label='tru')
    plt.axvline(1.22*wave/tel_diameter/image_res, color='k', linestyle=':')
    plt.axvline(0, color='c', linestyle=':')
    plt.legend()
    plt.xlim([-20,20])
    # plt.ylim(bottom=1e-5)
    breakpoint()

def do_prop(pupil, num_img, dx, xx, wave, image_distance, NN_full):

    FF = do_FFT(pupil)

    #Trim back to normal size   #TODO: what is max extent from nyquist?
    # FF = image_util.crop_image(FF, None, self.sim.num_pts//2)

    #Multiply by Fresnel diffraction phase prefactor

    FF *= propagation_kernel(xx, dx, wave, image_distance)

    #Multiply by constant phase term
    FF *= np.exp(1j * 2.*np.pi/wave * image_distance)

    #Normalize such that peak is 1. Needs to scale for wavelength relative to other images
    FF /= NN_full

    img = FF.copy()

    #Turn into intensity
    img = np.real(img.conj()*img)

    return img

def propagation_kernel(et, dx0, wave, distance):
    return np.exp(1j * 2.*np.pi/wave * dx0**2. * (et[:,None]**2 + et**2) / (2.*distance))

def do_FFT(MM):
    return fft.ifftshift(fft.fft2(fft.fftshift(MM), workers=-1))

def finalize_aff(img, num_img, targ_NN, true_NN, dx):
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
    out_shape = (int(NN/scaling),) * 2

    #Do affine transform
    img = affine_transform(img, affmat, output_shape=out_shape, order=3)

    #Crop to match image size
    img = image_util.crop_image(img, None, num_img//2)

    return img, dx

def finalize_cv2(img, num_img, targ_NN, true_NN, dx):
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
    out_shape = (int(NN/scaling),) * 2

    #FIXME: not great interpolation when scaling not integer. but it is 20x faster

    #Do affine transform
    img = cv2.warpAffine(img, affmat, out_shape, \
        flags=cv2.WARP_INVERSE_MAP + cv2.INTER_LANCZOS4)

    #Crop to match image size
    img = image_util.crop_image(img, None, num_img//2)

    return img, dx


def finalize_pil(img, num_img, targ_NN, true_NN, dx):
    #Resample onto theoretical resolution
    scaling = true_NN / targ_NN

    #Get new image size
    NN = len(img)
    N2 = NN / scaling

    #Turn into integer number
    N2 = int(N2)
    scaling = NN / N2

    #Scale sampling
    dx *= scaling

    rs = [Image.NEAREST, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS][-1]

    #Resize image
    img = np.array(Image.fromarray(img).resize((N2, N2), reducing_gap=5, \
        resample=rs))

    #Crop to match image size
    img = image_util.crop_image(img, None, num_img//2)

    return img, dx

run_sim()
