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

    def calculate_image(self, pupil, grid_pts):

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
        uu_waves = np.zeros_like(u0_waves)

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

        #Get component strengths via 'ray-tracing'
        # lab2ray, ray2lab = self.get_component_strengths(elem, self.r1*dx1, self.a1)
        lab2ray = self.get_component_strengths(elem, self.r1*dx1, self.a1)

        #Loop over wavelength
        for iw in range(len(self.sim.waves)):

            #Current wavelength
            wave = self.sim.waves[iw]

            #Get half-bandwidth
            hbf = self.get_bandwidth(wave, dx1, zz)

            #Apply phase function
            u0_iw = u0_waves[iw] * np.exp(1j*2*np.pi/wave * opd)

            # if ie == 1:
            #     import matplotlib.pyplot as plt;plt.ion()
            #     fig, axes = plt.subplots(3, 2, figsize=(8,11))
            #     for i in range(3):
            #         axes[i,0].imshow(abs(u0_iw[i]))
            #         # axes[i,1].imshow(abs(np.sum(lab2ray[i]*u0_iw, 0)))
            #         dd = np.sum(lab2ray[i]*u0_iw, 0)
            #         dd = np.sum(ray2lab[i]*dd, 0)
            #         axes[i,1].imshow(abs(dd))
            #     breakpoint()

            #Get transfer function
            fz2 = 1. - (wave*hbf*fx)**2 - (wave*hbf*fy)**2
            evind = fz2 < 0
            Hn = np.exp(1j* 2*np.pi/wave * zz * np.sqrt(np.abs(fz2)))
            Hn[evind] = 0
            del fz2, evind

            #scale factor
            scl = 2*np.pi * hbf

            #Loop over polarization
            for ip in range(self.sim.npol):

                #Get input field
                u0 = np.sum(lab2ray[ip]*u0_iw, 0)

                #Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
                angspec = finufft.nufft2d2(fx*scl*dx1, fy*scl*dx1, u0, \
                    isign=-1, eps=self.sim.fft_tol)

                #Get solution with inverse NUFFT (nonuniform -> nonuniform)
                uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq*hbf**2, \
                    self.x2*dx2, self.y2*dx2, isign=1, eps=self.sim.fft_tol)

                #Normalize
                uu *= dx1**2

                #Reshape
                uu = uu.reshape(uu_waves.shape[-2:])

                #Rotate back to lab
                # uu = np.sum(ray2lab[ip]*uu, 0)

                #Store
                uu_waves[iw,ip] = uu

        #Cleanup
        # del u0, Hn, angspec, uu, opd, rays, u0_iw

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

############################################
#####  Ray Tracing #####
############################################

    def get_component_strengths(self, elem, rads, angs):

        #Return ones immediately if scalar
        if self.sim.npol == 1:
            return np.tile(np.eye(3)[:,:,None,None], (1,1) + rads.shape)

        #Strength depends on element
        if elem.kind == 'aperture':
            #If aperture, return ones
            return np.tile(np.eye(self.sim.npol)[:,:,None,None], (1,1) + rads.shape)

        elif elem.kind == 'polarizer':
            #If polarizer, rotate into polarizer angle
            return self.polarizer_matrix(angs, np.radians(elem.polarizer_angle))

        else:
            #Otherwise, assume lens (spherical)
            return self.lens_matrix(rads, angs, elem.focal_length)

    ############################################

    def lens_matrix(self, rads, angs, ff):
        theta = np.arctan(rads/ff)
        cost = np.cos(theta)
        sint = np.sin(theta)
        cosp = np.cos(angs)
        sinp = np.sin(angs)

        #Rotation matrix from Prajapati_2021
        rot_mat = np.sqrt(cost) * np.array([
            [cost + sinp**2*(1-cost), cosp*sinp*(cost-1), -sint*cosp],
            [cosp*sinp*(cost-1), 1 - sinp**2*(1-cost), -sint*sinp],
            [sint*cosp, sint*sinp, cost]
        ])

        #Cleanup
        del theta, cost, sint, cosp, sinp

        return rot_mat

    # def lens_matrix(self, rads, angs, ff):
    #     theta = np.arctan(rads/ff)
    #     cost = np.cos(theta)
    #     sint = np.sin(theta)
    #     cosp = np.cos(angs)
    #     sinp = np.sin(angs)
    #
    #     #Rotation matrix from Prajapati_2021
    #     rot_mat = lambda ct, st, cp, sp: \
    #         np.sqrt(ct) * np.array([
    #             [ct + sp**2*(1-ct), cp*sp*(ct-1), -st*cp],
    #             [cp*sp*(ct-1), 1 - sp**2*(1-ct), -st*sp],
    #             [st*cp, st*sp, ct]
    #         ])
    #
    #     #Forward and backward rotations
    #     lab2ray = rot_mat(cost,  sint, cosp, sinp)
    #     ray2lab = rot_mat(cost, -sint, cosp, sinp)
    #
    #     #Cleanup
    #     del theta, cost, sint, cosp, sinp
    #
    #     return lab2ray, ray2lab

    ############################################

    def polarizer_matrix(self, angs, pol_ang):
        #Keep both parallel and orthogonal to polarization directions
        cp, sp = np.cos(pol_ang), np.sin(pol_ang)
        co, so = np.cos(pol_ang + np.pi/2), np.sin(pol_ang + np.pi/2)
        #X is parallel, Y is orthogonal
        rot_mat = np.array([[cp**2, cp*sp, 0], [co*so, so**2, 0], [0,0,1]])
        #Add extra dimensions
        return rot_mat[...,None,None]

############################################
############################################
