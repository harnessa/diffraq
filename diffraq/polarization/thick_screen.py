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

    def get_edge_field(self, dd, gw, wave):
        if self.is_sommerfeld:
            #Solve Sommerfeld solution
            return self.sommerfeld_solution(dd, wave)

        elif self.maxwell_func is not None:
            #Apply user-input function
            return self.maxwell_func(dd, wave)

        else:
            #Interpolate data from file
            return self.interpolate_file(dd, gw, wave)

############################################
############################################

############################################
#####  Sommerfeld Solution #####
############################################

    def F_func(self, s):
        S,C = fresnel(np.sqrt(2./np.pi)*s)
        return np.sqrt(np.pi/2.)*( 0.5*(1. + 1j) - (C + 1j*S))

    def sommerfeld_solution(self, dd, gw, wave):

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

    def interpolate_file(self, dd, gw, wave):
        #Split into gaps if applicable
        if len(gw) == 0:
            return self.interpolate_file_edge(dd, wave)
        else:
            return self.interpolate_file_gap(dd, gw, wave)

    def interpolate_file_edge(self, dd, wave):
        #Load data from file and build interpolation function for current wavelength
        with h5py.File(f'{self.maxwell_file}.h5', 'r') as f:
            xx = f['xx'][()]
            sf = f[f'{wave*1e9:.0f}_s'][()]
            pf = f[f'{wave*1e9:.0f}_p'][()]
            sfld = np.interp(dd, xx, sf, left=0j, right=0j)
            pfld = np.interp(dd, xx, pf, left=0j, right=0j)

        #Cleanup
        del xx, sf, pf

        return sfld, pfld

    def interpolate_file_gap(self, dd, gw, wave):

        #Get size of other axis
        ny = dd.size//gw.size

        #Load data from file and build interpolation function for current wavelength
        with h5py.File(f'{self.maxwell_file}.h5', 'r') as f:

            #Interpolate edge data for all (gaps will be overwritten -- just easier this way)
            xx = f['xx'][()]
            sf = f[f'{wave*1e9:.0f}_s'][()]
            pf = f[f'{wave*1e9:.0f}_p'][()]
            sfld = np.interp(dd, xx, sf, left=0j, right=0j)
            pfld = np.interp(dd, xx, pf, left=0j, right=0j)

            #Get widths
            widths = f['ww'][()]

            #Loop through widths and interpolate over gaps
            for i in range(len(widths)-1):
                #Get gap data
                xx = f[f'xx_gap_{i}'][()]
                sf = f[f'{wave*1e9:.0f}_gap_{i}_s'][()]
                pf = f[f'{wave*1e9:.0f}_gap_{i}_p'][()]

                #Get values in this gap region
                gind = np.where((gw >= widths[i]) & (gw < widths[i+1]))[0]

                #Skip if empty
                if len(gind) == 0:
                    continue

                #Expand gind into all points
                bigind = (gind*ny + np.arange(ny)[:,None]).ravel()

                #Interpolate this region
                sfld[bigind] = np.interp(dd[bigind], xx, sf, left=0j, right=0j)
                pfld[bigind] = np.interp(dd[bigind], xx, pf, left=0j, right=0j)

        #Cleanup
        del xx, sf, pf, gind, bigind, widths

        return sfld, pfld

############################################
############################################
