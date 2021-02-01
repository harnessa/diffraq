"""
focuser.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to propagate the diffracted field to the focal plane of the
    target imaging system.
"""

import numpy as np
from diffraq.utils import image_util
from scipy.ndimage import affine_transform
import scipy.fft as fft

class Focuser(object):

    def __init__(self, sim):
        self.sim = sim
        self.set_derived_parameters()

############################################
#####  Setup #####
############################################

    def set_derived_parameters(self):
        ### Image distances ###
        object_distance = {'source':self.sim.zz + self.sim.z0, \
            'occulter':self.sim.zz}[self.sim.focus_point]
        self.image_distance = 1./(1./self.sim.focal_length - 1./object_distance)
        self.image_res = self.sim.pixel_size / self.sim.focal_length

        #Input Spacing
        self.dx0 = self.sim.tel_diameter / self.sim.num_pts

        #Set padding
        self.get_padding()

    def get_padding(self):
        #Target padding required to properly sample (image distance drops out of top and bottom)
        self.targ_pad = self.sim.waves / (self.sim.tel_diameter * self.image_res)

        #Round padding to get to 2**n
        self.true_pad = (2**np.ceil( np.log10(self.sim.num_pts*self.targ_pad) / \
            np.log10(2)) / self.sim.num_pts).astype(int)

        #Make sure not too large
        if np.any(self.true_pad * self.sim.num_pts) > 2**12:
            self.sim.logger.error(f'Large Image size: {self.true_pad * self.sim.num_pts}', is_warning=True)

        #Get unique padding groups
        self.unq_true_pad = np.unique(self.true_pad)
        self.true_pad_group = [np.where(tp == self.unq_true_pad)[0][0] for tp in self.true_pad]

############################################
############################################

############################################
####	Main Function ####
############################################

    def calculate_image(self, in_pupil):
        #Get image size
        num_img = min(self.true_pad.max()*self.sim.num_pts, self.sim.image_size)

        #Create image container
        image = np.empty((len(self.sim.waves), num_img, num_img))

        #Create copy
        pupil = in_pupil.copy()
        del in_pupil

        #Round aperture and get number of points
        NN0 = pupil.shape[-1]
        pupil, NN_full = image_util.round_aperture(pupil)

        #Create input plane indices
        et = np.tile(np.arange(NN0) - (NN0 - 1.)/2., (NN0,1))

        #Loop through pad groups and calculate images that are the same size
        for pad in self.unq_true_pad:

            #This [padded] image size
            NN = NN0 * pad

            #Fine wavelengths in this pad group
            cur_inds = np.where(self.true_pad == pad)[0]

            #Loop through wavelengths in this pad group and calculate image
            for iw in cur_inds:

                #Propagate to image plane
                img = self.propagate_lens_diffraction(pupil[iw], \
                    self.sim.waves[iw], et, pad, NN, NN_full)

                #Turn into intensity
                img = np.real(img.conj()*img)

                #Finalize image
                img = self.finalize_image(img, num_img, self.targ_pad[iw], pad)

                #Store
                image[iw] = img

        #Cleanup
        del et, pupil, img

        return image

############################################
############################################

############################################
####	Image Propagation ####
############################################

    def propagate_lens_diffraction(self, pupil, wave, et, pad, NN, NN_full):

        #Get output plane sampling
        dx = wave*self.image_distance/(self.dx0*NN)

        #Store propagation distance
        zz = self.image_distance + self.sim.defocus

        #Multiply by propagation kernels (lens and Fresnel)
        pupil *= self.propagation_kernel(et, self.dx0, wave, -self.sim.focal_length)
        pupil *= self.propagation_kernel(et, self.dx0, wave, zz)

        #Pad pupil
        pupil = image_util.pad_array(pupil, pad)

        #Run FFT
        FF = self.do_FFT(pupil)

        #Trim back to normal size   #TODO: what is max extent from nyquist?
        FF = image_util.crop_image(FF, None, NN//pad//2)

        #Multiply by Fresnel diffraction phase prefactor
        FF *= self.propagation_kernel(et, dx, wave, zz)

        #Multiply by constant phase term
        FF *= np.exp(1j * 2.*np.pi/wave * zz)

        #Normalize by FFT + normalizations to match Fresnel diffraction (not used)
        # FF *= self.dx0**2./(wave*zz) / NN_full

        #Normalize such that peak is 1. Needs to scale for wavelength
        FF /= NN_full

        return FF

############################################
############################################

############################################
####	Misc Functions ####
############################################

    def propagation_kernel(self, et, dx0, wave, distance):
        return np.exp(1j * 2.*np.pi/wave * dx0**2. * (et.T**2 + et**2) / (2.*distance))

    def do_FFT(self, MM):
        return fft.ifftshift(fft.fft2(fft.fftshift(MM), workers=-1))

    def finalize_image(self, img, num_img, targ_pad, true_pad):
        #Resample onto theoretical resolution through affine transform
        scaling = true_pad / targ_pad

        #Pad image to help with resampling
        img = image_util.pad_array(img, 2)

        #Make sure scaling leads to even number (will lead to small difference in sampling)
        NN = len(img)
        N2 = NN / scaling
        scaling = NN / (N2 - (N2%2))

        #Affine matrix
        affmat = np.array([[scaling, 0, 0], [0, scaling, 0]])
        out_shape = (np.ones(2) * NN / scaling).astype(int)

        #Do affine transform
        img = affine_transform(img, affmat, output_shape=out_shape, order=5)

        #Crop to match image size
        img = image_util.crop_image(img, None, num_img//2)

        return img

############################################
############################################
