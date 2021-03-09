"""
bdw_focuser.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021

Description: Class to propagate the diffracted field to the focal plane of the
    target imaging system.
"""

import numpy as np
from scipy.ndimage import affine_transform
import scipy.fft as fft

class BDW_Focuser(object):

    def __init__(self, sim):
        self.sim = sim
        self.set_derived_parameters()

############################################
#####  Setup #####
############################################

    def set_derived_parameters(self):

        #Set wavelength as array since BDW is not currently multi-wave #TODO: remove
        self.waves = np.atleast_1d(self.sim.wave)

        #Get image distance depending on focus point
        object_distance = {'source':self.sim.z1 + self.sim.z0, \
            'occulter':self.sim.z1}[self.sim.focus_point]
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
        self.targ_NN = self.sim.num_pts * self.waves / \
            (self.sim.tel_diameter * self.image_res)

        #Find next fastest size for FFT (not necessarily 2**n)
        self.true_NN = np.array([fft.next_fast_len(int(np.ceil(tn))) for tn in self.targ_NN])

        #Make sure not too large
        if np.any(self.true_NN > 2**12):
            bad = self.true_NN[self.true_NN > 2**12]
            print(f'Large Image size: {bad}')

        #Make sure we are always oversampling
        if np.any(self.true_NN < self.targ_NN):
            bad = self.true_NN[self.true_NN < self.targ_NN]
            print(f'Undersampling: {bad}')

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
        image = np.empty((len(self.waves), num_img, num_img))
        sampling = np.empty(len(self.waves))

        #Create copy
        pupil = in_pupil.copy()
        del in_pupil

        #Round aperture and get number of points
        NN0 = pupil.shape[-1]
        pupil, NN_full = self.round_aperture(pupil)

        #Create input plane indices
        et = np.tile(np.arange(NN0)/NN0 - 0.5, (NN0,1)) * NN0

        #Loop through pad groups and calculate images that are the same size
        for NN in self.unq_true_NN:

            #Fine wavelengths in this pad group
            cur_inds = np.where(self.true_NN == NN)[0]

            #Loop through wavelengths in this pad group and calculate image
            for iw in cur_inds:

                #Propagate to image plane
                img, dx = self.propagate_lens_diffraction(pupil[iw], \
                    self.waves[iw], et, NN, NN_full)

                #Turn into intensity
                img = np.real(img.conj()*img)

                #Finalize image
                img, dx = self.finalize_image(img, num_img, self.targ_NN[iw], NN, dx)

                #Store
                image[iw] = img
                sampling[iw] = dx

        #Use mean sampling to calculate grid pts (this grid matches fft.fftshift(fft.fftfreq(NN, d=1/NN)))
        grid_pts = self.get_grid_points(image.shape[-1], dx=sampling.mean())

        #Convert to angular resolution
        grid_pts /= self.image_distance

        #Cleanup
        del et, pupil, img

        return image, grid_pts

############################################
############################################

############################################
####	Image Propagation ####
############################################

    def propagate_lens_diffraction(self, pupil, wave, et, NN, NN_full):

        #Get output plane sampling
        dx = wave*self.image_distance/(self.dx0*NN)

        #Multiply by propagation kernels (lens and Fresnel)
        pupil *= self.propagation_kernel(et, self.dx0, wave, -self.sim.focal_length)
        pupil *= self.propagation_kernel(et, self.dx0, wave, self.image_distance)

        #Pad pupil
        pupil = self.pad_array(pupil, NN)

        #Run FFT
        FF = self.do_FFT(pupil)

        #Trim back to normal size   #TODO: what is max extent from nyquist?
        FF = self.crop_image(FF, None, self.sim.num_pts//2)

        #Multiply by Fresnel diffraction phase prefactor
        FF *= self.propagation_kernel(et, dx, wave, self.image_distance)

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
####	Propagation Functions ####
############################################

    def propagation_kernel(self, et, dx0, wave, distance):
        return np.exp(1j * 2.*np.pi/wave * dx0**2. * (et.T**2 + et**2) / (2.*distance))

    def do_FFT(self, MM):
        return fft.ifftshift(fft.fft2(fft.fftshift(MM), workers=-1))

    def finalize_image(self, img, num_img, targ_NN, true_NN, dx):
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
        img = self.crop_image(img, None, num_img//2)

        return img, dx

############################################
############################################

############################################
####	Image Util Functions ####
############################################

    def crop_image(self, img, cen, wid):
        if cen is None:
            cen = np.array(img.shape)[-2:].astype(int)//2

        sub_img = img[..., max(0, int(cen[1] - wid)) : min(img.shape[-2], int(cen[1] + wid)), \
                           max(0, int(cen[0] - wid)) : min(img.shape[-1], int(cen[0] + wid))]

        return sub_img

    def pad_array(self, inarr, NN):
        return np.pad(inarr, (NN - inarr.shape[-1])//2)

    def round_aperture(self, img):
        #Build radius values
        rhoi = self.get_image_radii(img.shape[-2:])
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

    def get_grid_points(self, ngrid, width=None, dx=None):
        #Handle case for width supplied
        if width is not None:
            grid_pts = width*(np.arange(ngrid)/ngrid - 0.5)
            dx = width/ngrid

        #Handle case for spacing supplied
        elif dx is not None:
            grid_pts = dx*(np.arange(ngrid) - 0.5*ngrid)

        #Handle the odd case
        if ngrid % 2 == 1:
            #Shift for odd points
            grid_pts += dx/2

        return grid_pts

    def get_image_radii(self, img_shp, cen=None):
        yind, xind = np.indices(img_shp)
        if cen is None:
            cen = [img_shp[-2]/2, img_shp[-1]/2]
        return np.hypot(xind - cen[0], yind - cen[1])

############################################
############################################
