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
        vfunc_s, vfunc_p = self.load_vector_data(waves)

        #Build angle components
        cosa = np.cos(normals)
        sina = np.sin(normals)

        #TODO: do i still need the formulation from lotus if im calculating directly and not using rotations?
        #Build fields for each wavelength
        vec_UU = np.empty((len(waves), 2, len(normals))) + 0j
        for iw in range(len(waves)):

            #Get s/p fields
            s_fld = vfunc_s[iw](edge_dists)
            p_fld = vfunc_p[iw](edge_dists)

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
            vfunc_s, vfunc_p = self.load_sommerfeld(waves)

        elif self.maxwell_func is not None:
            #Build given functions for different wavelengths
            vfunc_s, vfunc_p = [], []
            for wav in waves:
                vfunc_s.append(lambda d: self.maxwell_func[0](d, wav))
                vfunc_p.append(lambda d: self.maxwell_func[1](d, wav))

        else:
            #Load data from file
            vfunc_s, vfunc_p = self.load_vector_file(waves)

        return vfunc_s, vfunc_p

############################################
############################################

############################################
#####  Sommerfeld Solution #####
############################################

    def load_sommerfeld(self, waves):

        def F_func(s):
            S,C = fresnel(np.sqrt(2./np.pi)*s)
            return np.sqrt(np.pi/2.)*( 0.5*(1. + 1j) - (C + 1j*S))

        G_func = lambda s: np.exp(-1j*s**2.)*F_func(s)

        def Uz(rho, phi, kk, EH_sign):
            uu = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi - np.pi/2.))
            vv = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi + np.pi/2.))
            pre = np.exp(-1j*np.pi/4.) / np.sqrt(np.pi) * np.exp(1j*kk*rho)
            return pre * (G_func(uu) - EH_sign*G_func(vv))

        def sommerfeld(xx, wave, EH_sign):
            #Incident field
            U0 = np.heaviside(xx, 1)

            #Polar coordinates
            rho = abs(xx)
            phi = (1 + EH_sign*(U0 - 1))*np.pi

            #Wavelength
            kk = 2.*np.pi/wave

            #Get field solution (minus incident field)
            uu = Uz(rho, phi, kk, EH_sign) - U0

            return uu

        #Build Sommerfeld solution for different wavelengths
        vfunc_s, vfunc_p = [], []
        for wav in waves:
            vfunc_s.append(lambda d: sommerfeld(d, wav,  1))
            vfunc_p.append(lambda d: sommerfeld(d, wav, -1))

        return vfunc_s, vfunc_p

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
