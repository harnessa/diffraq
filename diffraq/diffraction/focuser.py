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
from diffraq.quadrature import polar_quad
import finufft

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
        self.image_res = self.sim.pixel_size / self.image_distance

        #Input Spacing
        self.dx0 = self.sim.tel_diameter / self.sim.num_pts

        #Gaussian quad number
        self.rad_nodes, self.the_nodes = self.sim.angspec_radial_nodes, self.sim.angspec_theta_nodes

############################################
############################################

############################################
####	Main Function ####
############################################

    def calculate_image(self, in_pupil, grid_pts):

        #Run single lens or lens system?
        if self.sim.lens_system is None:
            return self.calc_single_lens(in_pupil, grid_pts)
        else:
            return self.calc_lens_system(in_pupil, grid_pts)

############################################
############################################

############################################
####	Single Lens ####
############################################

    def calc_single_lens(self, in_pupil, grid_pts):

        #Get image size
        num_img = self.sim.image_size

        #Create image container
        image = np.empty((len(self.sim.waves), num_img, num_img))

        #Create copy
        pupil = in_pupil.copy()
        del in_pupil

        #Round aperture and get number of points
        pupil, NN_full = image_util.round_aperture(pupil, grid_pts, self.sim.tel_diameter)

        #Get paraxial lens OPD
        rr = image_util.get_grid_radii(grid_pts)
        opd = -rr**2/(2*self.sim.focal_length)
        del rr

        #Setup angular spectrum
        fx, fy, wq = self.setup_angspec()

        #Target coordinates
        ox_1D = (np.arange(num_img) - num_img/2) * self.sim.pixel_size
        ox = np.tile(ox_1D, (num_img, 1))
        oy = ox.T.flatten()
        ox = ox.flatten()

        #Loop through wavelengths and calculate image
        for iw in range(len(self.sim.waves)):

            #Start initial field with current lens phase added
            u0 = pupil[iw] * np.exp(1j * 2*np.pi/self.sim.waves[iw] * opd)

            #Propagate
            img = self.propagate(u0, self.image_distance, \
                self.sim.waves[iw], fx, fy, wq, ox, oy)

            #Turn into intensity
            img = np.real(img.conj()*img)

            #Reshape into 2D and store
            image[iw] = img.reshape((num_img, num_img))

        #Convert output points to angle
        image_pts = ox_1D / self.image_distance

        #Cleanup
        del pupil, img, fx, fy, wq, ox, oy, u0

        return image, image_pts

############################################
############################################

############################################
####	Lens System ####
############################################

    def calc_lens_system(self, in_pupil, grid_pts):
        #TODO: add

        import matplotlib.pyplot as plt;plt.ion()
        breakpoint()
        
############################################
############################################

############################################
#####  Angular Spectrum Propagation #####
############################################

    def propagate(self, u0, zz, wave, fx, fy, wq, ox, oy):

        #Get transfer function
        fz2 = 1. - (wave*fx)**2 - (wave*fy)**2
        evind = fz2 < 0
        Hn = np.exp(1j* 2*np.pi/wave * zz * np.sqrt(np.abs(fz2)))
        Hn[evind] = 0

        #scale factor
        scl = 2*np.pi

        #Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
        angspec = finufft.nufft2d2(fx*scl*self.dx0, fy*scl*self.dx0, u0, \
            isign=-1, eps=self.sim.fft_tol)

        #Get solution with inverse NUFFT (nonuniform -> nonuniform)
        uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, ox, oy, \
            isign=1, eps=self.sim.fft_tol)

        #Normalize
        uu *= self.dx0**2

        return uu

############################################
############################################

############################################
#####  Angular Spectrum Setup #####
############################################

    def setup_angspec(self):

        #Use minimum wavelength to set bandwidth
        wave = self.sim.waves.min()

        #Critical distance
        zcrit = 2*self.sim.num_pts*self.dx0**2/wave

        #Calculate bandwidth
        if self.image_distance < zcrit:
            bf = 1/self.dx0
        elif zz >= 3*zcrit:
            bf = np.sqrt(2*self.sim.num_pts/(wave*self.image_distance))
        else:
            bf = 2*self.sim.num_pts*self.dx0/(wave*self.image_distance)

        #Get gaussian quad
        fx, fy, wq = polar_quad(lambda t: np.ones_like(t)*bf/2, self.rad_nodes, self.the_nodes)

        return fx, fy, wq

############################################
############################################
