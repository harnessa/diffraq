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

        #Build angle components
        cosa = np.cos(normals)
        sina = np.sin(normals)

        #Build fields for each wavelength
        vec_UU = np.empty((len(waves), 2, len(normals))) + 0j
        for iw in range(len(waves)):

            #Get edge field
            s_fld, p_fld = self.get_edge_field(edge_dists, waves[iw])

            #Build horizontal, vertical, and crossed components (different from Op.Exp. paper b/c here, normal defined from horizontal and CCW)
            mH = s_fld*sina**2 + p_fld*cosa**2
            mV = s_fld*cosa**2 + p_fld*sina**2
            mX = sina*cosa * (s_fld - p_fld)

            #Build incident field maps
            vec_UU[iw, 0] = Ex_comp*mH + Ey_comp*mX         #Horizontal
            vec_UU[iw, 1] = Ex_comp*mX + Ey_comp*mV         #Vertical

        #Cleanup
        del cosa, sina, s_fld, p_fld, mH, mV, mX

        return vec_UU

    def get_edge_field(self, dd, wave):
        if self.is_sommerfeld:
            #Solve Sommerfeld solution
            return self.sommerfeld_solution(dd, wave)

        elif self.maxwell_func is not None:
            #Apply user-input function
            return self.maxwell_func(dd, wave)

        else:
            #Interpolate data from file
            return self.interpolate_edge_file(dd, wave)

############################################
############################################

############################################
#####  Sommerfeld Solution #####
############################################

    def F_func(self, s):
        S,C = fresnel(np.sqrt(2./np.pi)*s)
        return np.sqrt(np.pi/2.)*( 0.5*(1. + 1j) - (C + 1j*S))

    def sommerfeld_solution(self, dd, wave):

        #Incident field
        U0 = np.heaviside(dd, 1)

        #Polar coordinates
        rho = abs(dd)
        phi = (1 - (U0 - 1))*np.pi

        #Wavelength
        kk = 2.*np.pi/wave

        #Build arguments of calculation
        uu = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi - np.pi/2.))
        vv = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi + np.pi/2.))
        pre = np.exp(-1j*np.pi/4.) / np.sqrt(np.pi) * np.exp(1j*kk*rho)

        #Intermediate calculations (minus incident field)
        Umid = pre*np.exp(-1j*uu**2.)*self.F_func(uu) - U0
        Gv = pre*np.exp(-1j*vv**2.)*self.F_func(vv)

        #Get field solution (minus incident field) for s,p polarization
        Us = Umid + Gv
        Up = Umid - Gv

        #Cleanup
        del U0, rho, phi, uu, vv, pre, Umid, Gv

        return Us, Up

############################################
############################################

############################################
#####  Load from file #####
############################################

    def interpolate_edge_file(self, dd, wave):
        #Load data from file and build interpolation function for current wavelength
        with h5py.File(f'{self.maxwell_file}.h5', 'r') as f:
            xx = f['xx'][()]
            sfld = np.interp(dd, xx, f[f'{wave*1e9:.0f}_s'][()])
            pfld = np.interp(dd, xx, f[f'{wave*1e9:.0f}_p'][()])

        return sfld, pfld

############################################
############################################
