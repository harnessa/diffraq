"""
thick_screen.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class representing the thick screen edge of an occulter/aperture
    to calculate the contribution from non-scalar diffraction.

"""

import numpy as np
from scipy.special import fresnel
import h5py

class ThickScreen(object):

    def __init__(self, **kwargs):
        #Set parameters
        for k,v in kwargs.items():
            setattr(self, k, v)

############################################
#####  Main Script #####
############################################

    def get_vector_field(self, edge_dists, normals, waves, Ex_comp, Ey_comp):

        #Get edge functions
        vfunc = self.load_vector_data(waves)

        #Build angle components
        cosa = np.cos(normals)
        sina = np.sin(normals)

        #Build fields for each wavelength
        vec_UU = np.empty((len(waves), 2, len(normals))) + 0j
        for iw in range(len(waves)):

            #Get s/p fields
            s_fld, p_fld = vfunc[iw](edge_dists)

            #Build horizontal, vertical, and crossed components
            mH = s_fld*cosa**2 + p_fld*sina**2
            mV = s_fld*sina**2 + p_fld*cosa**2
            mX = sina*cosa * (p_fld - s_fld)

            #Build incident field maps
            vec_UU[iw, 0] = Ex_comp*mH + Ey_comp*mX         #Horizontal
            vec_UU[iw, 1] = Ex_comp*mX + Ey_comp*mV         #Vertical

        #Cleanup
        del cosa, sina, s_fld, p_fld, mH, mV, mX

        return vec_UU

    def load_vector_data(self, waves):
        if self.is_sommerfeld:
            #Build Sommerfeld solution
            vfunc = self.load_sommerfeld(waves)

        elif self.maxwell_func is not None:
            #Build given functions for different wavelengths
            vfunc = []
            for wav in waves:
                vfunc.append(lambda d: self.maxwell_func(d, wav))

        else:
            #Load data from file
            vfunc = self.load_vector_file(waves)

        return vfunc

############################################
############################################

############################################
#####  Sommerfeld Solution #####
############################################

    def load_sommerfeld(self, waves):

        def F_func(s):
            S,C = fresnel(np.sqrt(2./np.pi)*s)
            return np.sqrt(np.pi/2.)*( 0.5*(1. + 1j) - (C + 1j*S))

        def sommerfeld(xx, wave):
            #Incident field
            U0 = np.heaviside(xx, 1)

            #Polar coordinates
            rho = abs(xx)
            phi = (1 - (U0 - 1))*np.pi

            #Wavelength
            kk = 2.*np.pi/wave

            #Build arguments of calculation
            uu = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi - np.pi/2.))
            vv = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi + np.pi/2.))
            pre = np.exp(-1j*np.pi/4.) / np.sqrt(np.pi) * np.exp(1j*kk*rho)

            #Intermediate calculations (minus incident field)
            Umid = pre*np.exp(-1j*uu**2.)*F_func(uu) - U0
            Gv = pre*np.exp(-1j*vv**2.)*F_func(vv)

            #Get field solution (minus incident field) for s,p polarization
            Us = Umid - Gv
            Up = Umid + Gv

            #Cleanup
            del U0, rho, phi, uu, vv, pre, Umid, Gv

            return Us, Up

        #Build Sommerfeld solution for different wavelengths
        vfunc = []
        for wav in waves:
            vfunc.append(lambda d: sommerfeld(d, wav))

        return vfunc

############################################
############################################

############################################
#####  Load from file #####
############################################

    def load_vector_file(self):
        #Load data from file and build interpolation function

        breakpoint()

############################################
############################################
