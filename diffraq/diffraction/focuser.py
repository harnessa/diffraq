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
        #Get image distance depending on focus point
        object_distance = {'source':self.sim.zz + self.sim.z0, \
            'occulter':self.sim.zz}[self.sim.focus_point]
        self.image_distance = 1./(1./self.sim.focal_length - 1./object_distance)

        #Add defocus to image distance
        self.image_distance += self.sim.defocus

        #Resolution
        self.image_res = self.sim.pixel_size / self.sim.focal_length

        #Input Spacing
        self.dx0 = self.sim.tel_diameter / self.sim.num_pts

        #Set padding
        self.get_padding()

    def get_padding(self):
        #Target number of FFT array (with padding) required to properly sample
        self.targ_NN = self.sim.num_pts * self.sim.waves / \
            (self.sim.tel_diameter * self.image_res)

        #Find next fastest size for FFT (not necessarily 2**n)
        self.true_NN = np.array([]).astype(int)
        for tn in self.targ_NN:
            #Make sure we have minimum padding
            tar = int(max(np.ceil(tn), self.sim.num_pts * self.sim.min_padding))
            #Get next fasest (even)
            tru, dn = 1, 0
            while (tru % 2) != 0:
                tru = fft.next_fast_len(tar+dn)
                dn += 1
            self.true_NN = np.concatenate((self.true_NN, [tru]))

        #Make sure not too large
        if np.any(self.true_NN > 2**14):
            bad = self.true_NN[self.true_NN > 2**14]
            self.sim.logger.error(f'Large Image size: {bad}', is_warning=True)

        #Make sure we are always oversampling
        if np.any(self.true_NN < self.targ_NN):
            bad = self.true_NN[self.true_NN < self.targ_NN]
            self.sim.logger.error(f'Undersampling: {bad}')

        #Get unique padding groups
        self.unq_true_NN = np.unique(self.true_NN)
        self.true_NN_group = [np.where(tn == self.unq_true_NN)[0][0] for tn in self.true_NN]

############################################
############################################

############################################
####	Main Function ####
############################################

    def calculate_image(self, in_pupil):
        #Get image size
        num_img = min(self.true_NN.max(), self.sim.image_size)

        #Create image container
        image = np.empty((len(self.sim.waves), num_img, num_img))
        sampling = np.empty(len(self.sim.waves))

        #Create copy
        pupil = in_pupil.copy()
        del in_pupil

        #Round aperture and get number of points
        NN0 = pupil.shape[-1]
        pupil, NN_full = image_util.round_aperture(pupil)

        #Create input plane indices
        et = (np.arange(NN0)/NN0 - 0.5) * NN0

        #Loop through pad groups and calculate images that are the same size
        for NN in self.unq_true_NN:

            #Fine wavelengths in this pad group
            cur_inds = np.where(self.true_NN == NN)[0]

            #Create focal plane indices
            xx = (np.arange(NN)/NN - 0.5) * NN

            #Loop through wavelengths in this pad group and calculate image
            for iw in cur_inds:

                #Propagate to image plane
                img, dx = self.propagate_lens_diffraction(pupil[iw], \
                    self.sim.waves[iw], et, xx, NN, NN_full)

                #Turn into intensity
                img = np.real(img.conj()*img)

                #Finalize image
                img, dx = self.finalize_image(img, num_img, self.targ_NN[iw], NN, dx)

                #Store
                image[iw] = img
                sampling[iw] = dx

        #Use mean sampling to calculate grid pts (this grid matches fft.fftshift(fft.fftfreq(NN, d=1/NN)))
        grid_pts = image_util.get_grid_points(image.shape[-1], dx=sampling.mean())

        #Convert to angular resolution
        grid_pts /= self.image_distance

        #Cleanup
        del et, pupil, img, xx

        return image, grid_pts

############################################
############################################

############################################
####	Image Propagation ####
############################################

    def propagate_lens_diffraction(self, pupil, wave, et, xx, NN, NN_full):

        #Get output plane sampling
        dx = wave*self.image_distance/(self.dx0*NN)

        #Multiply by propagation kernels (lens and Fresnel)
        pupil *= self.propagation_kernel(et, self.dx0, wave, -self.sim.focal_length)
        pupil *= self.propagation_kernel(et, self.dx0, wave, self.image_distance)

        #Pad pupil
        pupil = image_util.pad_array(pupil, NN)

        #Run FFT
        FF = self.do_FFT(pupil)

        #Multiply by Fresnel diffraction phase prefactor
        FF *= self.propagation_kernel(xx, dx, wave, self.image_distance)

        #Multiply by constant phase term
        FF *= np.exp(1j * 2.*np.pi/wave * self.image_distance)

        #(not used) Normalize by FFT + normalizations to match Fresnel diffraction
        # FF *= self.dx0**2./(wave*self.image_distance) / NN_full

        #Normalize such that peak is 1. Needs to scale for wavelength relative to other images
        FF /= NN_full

        return FF, dx

############################################
############################################

############################################
####	Misc Functions ####
############################################

    def propagation_kernel(self, et, dx0, wave, distance):
        return np.exp(1j * 2.*np.pi/wave * dx0**2. * (et[:,None]**2 + et**2) / (2.*distance))

    def do_FFT(self, MM):
        return fft.ifftshift(fft.fft2(fft.fftshift(MM), workers=-1))

    def finalize_image(self, img, num_img, targ_NN, true_NN, dx):
        #Resample onto theoretical resolution through affine transform
        scaling = true_NN / targ_NN

        #TODO: leads to bad results when scaling >> 1

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

############################################
############################################
