import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from diffraq.utils import image_util
from scipy import fft
from scipy.ndimage import affine_transform, shift
from PIL import Image
import cv2
import time

def get_image():

    wave = [400e-9, 641e-9][1]
    num_pts = 512
    tel_diameter = 5e-3
    min_padding = 4
    dx0 = tel_diameter / num_pts
    num_img = 512

    upsamp = [False, True][1]

    dsamp = 2.3

    if upsamp:
        NN = num_img * min_padding * int(dsamp)
        targ_NN = NN / dsamp
    else:
        NN = num_img * min_padding
        targ_NN = NN * dsamp

    #new image res
    image_res = num_pts * wave / (tel_diameter * targ_NN)
    focal_length = 13e-6/image_res
    image_distance = focal_length

    # breakpoint()
    print(NN, targ_NN)

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
    pimg, pdx = finalize_cv2(img.copy(), num_img, targ_NN, NN, dx)
    tok2 = time.perf_counter()
    print(f'PIL: {tok2-tik2:.3f}')

    #Get truth
    tru_pupil = image_util.pad_array(tru_pupil, int(targ_NN))
    dxtru = wave*image_distance/(dx0*int(targ_NN))
    xxtru = (np.arange(int(targ_NN))/int(targ_NN) - 0.5) * int(targ_NN)
    irdx = image_res * image_distance

    tru = do_prop(tru_pupil, num_img, dxtru, xxtru, wave, image_distance, NN_full)
    tru = image_util.crop_image(tru, None, num_img//2)

    #Difference
    adif = np.abs(tru - aimg).sum()
    pdif = np.abs(tru - pimg).sum()

    print(f'Adif: {adif:.3e}, Pdif: {pdif:.3e}')
    # print(f'Adx: {adx:.3e}, Pdx: {pdx:.3e}, True dx: {dxtru:.3e}, Image Res: {irdx:.3e}')

    #Plot
    pxx = xx[NN//2-len(aimg)//2:NN//2+len(aimg)//2]
    pxt = xxtru[int(targ_NN)//2-len(tru)//2:int(targ_NN)//2+len(tru)//2]
    plt.figure()
    plt.semilogy(pxx, aimg[len(aimg)//2], '-',  label='AFF')
    plt.semilogy(pxx, pimg[len(pimg)//2], '--', label='PIL')
    plt.semilogy(pxt, tru[len(tru)//2], ':', label='tru')
    plt.axvline(1.22*wave/tel_diameter/image_res, color='k', linestyle=':')
    plt.legend()
    plt.xlim([-20,20])
    plt.ylim(bottom=1e-5)
    breakpoint()

def do_prop(pupil, num_img, dx, xx, wave, image_distance, NN_full):

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
    out_shape = (np.ones(2) * NN / scaling).astype(int)

    #Do affine transform
    img = affine_transform(img, affmat, output_shape=out_shape, order=5)

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

    #FIXME: not great interpolation when scaling not integer. but it is 20x faster

    #Do affine transform
    img = cv2.warpAffine(img, affmat, (int(N2),int(N2)), \
        flags=cv2.WARP_INVERSE_MAP + cv2.WARP_FILL_OUTLIERS + cv2.INTER_LANCZOS4)

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

    rs = [Image.NEAREST, Image.BILINEAR, Image.HAMMING, Image.BICUBIC, Image.LANCZOS][-2]

    #Resize image
    img = np.array(Image.fromarray(img).resize((N2, N2), reducing_gap=5, \
        resample=rs))

    #Crop to match image size
    img = image_util.crop_image(img, None, num_img//2)

    return img, dx


get_image()
