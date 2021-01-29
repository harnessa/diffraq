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

        ### Padding ###
        self.dx0 = self.sim.tel_diameter / self.sim.num_pts
        self.target_pad, self.true_pad = self.get_padding()

    def get_padding(self):
        #Target padding required to properly sample (image distance drops out of top and bottom)
        targ_pad = self.sim.waves / (self.sim.tel_diameter * self.image_res)

        #Round padding to get to 2**n
        true_pad = (2**np.ceil( np.log10(self.sim.num_pts*targ_pad) / \
            np.log10(2)) / self.sim.num_pts).astype(int)

        #Make sure not too large
        if np.any(true_pad * self.sim.num_pts) > 2**12:
            self.sim.logger.error(f'Large Image size: {true_pad * self.sim.num_pts}', is_warning=True)

        return targ_pad, true_pad

############################################
############################################

############################################
####	Main Function ####
############################################

    def calculate_image(self, pupil):
        #Get image size
        num_img = min(self.true_pad.max()*self.sim.num_pts, self.sim.image_size)

        #Create image container
        image = np.empty((len(self.sim.waves), num_img, num_img))

        #Loop through wavelengths and calculate image
        for iw in range(len(self.sim.waves)):

            #Get current image
            img, dx = self.get_image(pupil[iw], self.sim.waves[iw], self.true_pad[iw])

            #Finalize image
            img = self.finalize_image(img, num_img, self.target_pad[iw], self.true_pad[iw])

            #Store
            image[iw] = img

        return image

############################################
############################################

############################################
####	Image Propagation ####
############################################

    def get_image(self, pupil, wave, true_pad):
        #Round aperture
        pupil = image_util.round_aperture(pupil)

        #Pad array
        pupil = image_util.pad_array(pupil, true_pad)

        #Propagate to focal plane
        image, dx = self.propagate_lens_diffraction(pupil, wave)

        #Turn into intensity
        image = np.real(image.conj()*image)

        return image, dx

    def propagate_lens_diffraction(self, pupil, wave, dx0=None):
        #Input plane points
        NN = len(pupil)

        #Create input plane indices
        et = np.tile(np.arange(NN) - (NN - 1.)/2., (NN,1))

        #Get output plane sampling
        dx = wave*self.image_distance/(self.dx0*NN)

        #Store propagation distance
        zz = self.image_distance

        #Multiply by propagation kernels (lens and the Fresnel)
        pupil *= self.propagation_kernel(et.T, et, self.dx0, wave, -self.sim.focal_length)
        pupil *= self.propagation_kernel(et.T, et, self.dx0, wave, zz)

        #Run FFT
        FF = self.do_FFT(pupil)

        #Trim far outer reaches
        max_img = self.sim.num_pts * 2       #TODO: what is max extent from nyquist?
        FF = image_util.crop_image(FF, None, max_img//2)
        et = image_util.crop_image(et, None, max_img//2)

        #Multiply by Fresnel diffraction phase prefactor
        FF *= np.exp(1j * 2.*np.pi/wave * dx**2. * (et.T**2 + et**2) / (2.*zz))

        #Multiply by constant phase term
        FF *= np.exp(1j * 2.*np.pi/wave * zz)

        #Normalize by FFT + normalizations to match Fresnel diffraction (not used)
        # FF *= self.dx0**2./(wave*zz) / np.count_nonzero(np.abs(pupil) != 0)

        #Normalize such that peak is 1. Needs to scale for wavelength
        FF /= np.count_nonzero(np.abs(pupil) != 0)

        #Cleanup
        del et

        return FF, dx

############################################
############################################

############################################
####	Misc Functions ####
############################################

    def propagation_kernel(self, xi, et, dx0, wave, distance):
        return np.exp(1j * 2.*np.pi/wave * dx0**2. * (xi**2 + et**2) / (2.*distance))

    def do_FFT(self, MM):
        return np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(MM)))

    def finalize_image(self, img, num_img, targ_pad, true_pad):
        #Resample onto theoretical resolution through affine transform
        scaling = true_pad / targ_pad

        #Make sure scaling leads to even number (will lead to small difference in sampling)
        NN = len(img)
        N2 = NN / scaling
        scaling = NN / (N2 - (N2%2))

        #Affine matrix + offset
        affmat = np.array([[scaling, 0, 0], [0, scaling, 0]])
        out_shape = (np.array(img.shape) / scaling).astype(int)

        #Do affine transform
        img = affine_transform(img, affmat, output_shape=out_shape, order=5)

        #Crop to match image size
        img = image_util.crop_image(img, None, num_img//2)

        return img

############################################
############################################
