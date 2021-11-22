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
import diffraq
import finufft

class Focuser(object):

    def __init__(self, sim):
        self.sim = sim
        self.num_pts = self.sim.num_pts
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
        self.dx0 = self.sim.tel_diameter / self.num_pts

        #Gaussian quad number
        self.rad_nodes, self.the_nodes = self.sim.angspec_radial_nodes, self.sim.angspec_theta_nodes

        #Build lens system
        self.lenses = diffraq.diffraction.Lens_System(self.sim.lens_system, self)

############################################
############################################

############################################
####	Main Function ####
############################################

    def calculate_image(self, in_pupil, grid_pts):

        #Create copy
        pupil = in_pupil.copy()
        del in_pupil

        #Get gaussian quad
        fx, fy, wq = polar_quad(lambda t: np.ones_like(t), self.rad_nodes, self.the_nodes)

        #Input coordinates
        x1 = np.arange(self.num_pts) - self.num_pts/2
        self.r1 = np.hypot(x1, x1[:,None])
        self.a1 = np.arctan2(x1[:,None], x1)

        #Target coordinates
        x2 = np.tile(x1, (self.num_pts, 1))
        self.y2 = x2.T.flatten()
        self.x2 = x2.flatten()
        del x1, x2

        #Loop through elements and propagate
        for ie in range(self.lenses.n_elements):

            #Propagate element
            pupil, dx2 = self.propagate_element(pupil, ie, fx, fy, wq)

        #Get image size #TODO: limit imposed by ang spec numerics
        num_img = self.sim.image_size

        #Trim to image
        image = image_util.crop_image(pupil, None, num_img//2)

        #Turn into intensity
        image = np.real(image.conj()*image)

        #Create output points
        image_pts = (np.arange(num_img) - num_img/2)*dx2/self.image_distance

        #Cleanup
        del fx, fy, wq, self.r1, self.x2, self.y2, pupil

        return image, image_pts

############################################
############################################

############################################
#####  Angular Spectrum Propagation #####
############################################

    def propagate_element(self, u0_waves, ie, fx, fy, wq):

        #Round aperture (always)
        u0_waves = image_util.round_aperture(u0_waves)

        #Create image container
        uu_waves = np.zeros((len(self.sim.waves), self.num_pts, self.num_pts)) + 0j

        #Get current element
        elem = getattr(self.lenses, f'element_{ie}')

        #Get propagation distance
        zz = elem.distance

        #Add defocus to last element
        if elem.is_last:
            zz += self.sim.defocus

        #Get spacings
        dx1 = elem.dx
        #if last plane, use image pixel sampling
        if elem.is_last:
            dx2 = self.sim.pixel_size
        else:
            dx2 = getattr(self.lenses, f'element_{ie+1}').dx

        #Get OPD
        opd = elem.opd_func(self.r1*dx1, self.a1)

        #Loop over wavelength
        for iw in range(len(self.sim.waves)):

            #Current wavelength
            wave = self.sim.waves[iw]

            #Get half-bandwidth
            hbf = self.get_bandwidth(wave, dx1, zz)

            #Apply phase function
            u0 = u0_waves[iw] * np.exp(1j*2*np.pi/wave * opd)

            #Get transfer function
            fz2 = 1. - (wave*hbf*fx)**2 - (wave*hbf*fy)**2
            evind = fz2 < 0
            Hn = np.exp(1j* 2*np.pi/wave * zz * np.sqrt(np.abs(fz2)))
            Hn[evind] = 0
            del fz2, evind

            #scale factor
            scl = 2*np.pi * hbf

            #Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
            angspec = finufft.nufft2d2(fx*scl*dx1, fy*scl*dx1, u0, \
                isign=-1, eps=self.sim.fft_tol)

            #Get solution with inverse NUFFT (nonuniform -> nonuniform)
            uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq*hbf**2, \
                self.x2*dx2, self.y2*dx2, isign=1, eps=self.sim.fft_tol)

            #Normalize
            uu *= dx1**2

            #Store
            uu_waves[iw] = uu.reshape(uu_waves.shape[-2:])

        #Cleanup
        del u0, Hn, angspec, uu

        return uu_waves, dx2

############################################
############################################

############################################
#####  Angular Spectrum Setup #####
############################################

    def get_bandwidth(self, wave, dx, zz):

        #Critical distance
        zcrit = 2*self.num_pts*dx**2/wave

        #Calculate bandwidth
        if zz < zcrit:
            bf = 1/dx
        elif zz >= 3*zcrit:
            bf = np.sqrt(2*self.num_pts/(wave*zz))
        else:
            bf = 2*self.num_pts*dx/(wave*zz)

        #Divide by two because radius of ang spec quadrature
        bf /= 2

        return bf

############################################
############################################
