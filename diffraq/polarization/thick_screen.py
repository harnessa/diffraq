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

    def get_edge_field(self, dd, gw, wave, tq):
        if self.is_sommerfeld:
            #Solve Sommerfeld solution
            return self.sommerfeld_solution(dd, wave, tq)

        elif self.maxwell_func is not None:
            #Apply user-input function
            return self.maxwell_func(dd, wave)

        else:
            #Interpolate data from file
            return self.interpolate_file(dd, gw, wave, tq)

############################################
############################################

############################################
#####  Sommerfeld Solution #####
############################################

    def G_func(self, s):
        S,C = fresnel(np.sqrt(2./np.pi)*s)
        ans = (1. + 1j)/2 - (C + 1j*S)
        ans *= np.exp(-1j*s**2.)
        del S, C
        return ans

    def sommerfeld_solution(self, dd, wave, tq):

        #FIXME: add tilted source
        if tq is not None:
            breakpoint()

        #Wavelength
        kk = 2.*np.pi/wave

        #normal incidence
        phi0 = np.pi/2

        zz = 0./30 * 1e-6
        rho = np.sqrt(zz**2 + dd**2.)

        #Polar angle from Incident field (heaviside)
        if np.isclose(zz, 0):
            phi = (2 - np.heaviside(dd, 1))*np.pi
        else:
            phi = 2.*np.pi + np.arctan2(-zz, -dd)

        #Build arguments of calculation (assumes z=0)
        uu = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi - phi0))
        vv = -np.sqrt(2.*kk*rho)*np.cos(0.5*(phi + phi0))
        pre = (np.exp(-1j*np.pi/4.) / np.sqrt(np.pi) * np.sqrt(np.pi/2)) * \
            np.exp(1j*kk*rho)

        pre /= np.exp(-1j*kk*rho*np.cos(phi - phi0))

        #Intermediate calculations (minus incident field)
        Umid = pre*self.G_func(uu) - np.heaviside(dd, 1)
        Gv = pre*self.G_func(vv)

        #Cleanup
        del phi, uu, vv, pre, rho

        #Get field solution (minus incident field) for s,p polarization
        Us = Umid - Gv
        Up = Umid + Gv

        #Cleanup
        del Umid, Gv

        return Us, Up

############################################
############################################

############################################
#####  Load from file #####
############################################

    def interpolate_file(self, dd, gw, wave, tq):
        #Use tilted source if applicable
        if tq is not None:
            return self.interpolate_file_tilted(dd, wave, tq)

        #Split into gaps if applicable
        if len(gw) == 0:
            return self.interpolate_file_edge(dd, wave)
        else:
            return self.interpolate_file_gap(dd, gw, wave)

    ############################################

    def interpolate_file_edge(self, dd, wave):
        #Load data from file and build interpolation function for current wavelength
        with h5py.File(f'{self.maxwell_file}.h5', 'r') as f:
            xx = f[f'{wave*1e9:.0f}_x'][()]
            sf = f[f'{wave*1e9:.0f}_s'][()]
            pf = f[f'{wave*1e9:.0f}_p'][()]
            sfld = np.interp(dd, xx, sf, left=0j, right=0j)
            pfld = np.interp(dd, xx, pf, left=0j, right=0j)

        #Cleanup
        del xx, sf, pf

        return sfld, pfld

    ############################################

    def interpolate_file_tilted(self, dd, wave, tq):
        #Load data from file and build interpolation function for current wavelength
        with h5py.File(f'{self.maxwell_file}.h5', 'r') as f:
            xx = f[f'{wave*1e9:.0f}_x'][()]
            sf = f[f'{wave*1e9:.0f}_s'][()]
            pf = f[f'{wave*1e9:.0f}_p'][()]
            inn_angles = np.radians(f['inn_angles'][()])
            out_angles = np.radians(f['out_angles'][()])

        #Initialize fields
        sfld = np.zeros(len(dd)) + 0j
        pfld = np.zeros(len(dd)) + 0j

        #Tilt angles
        omg = tq[:,0]
        psi = abs(tq[:,1])

        #Check we span the full range
        if (omg.min() < inn_angles.min()) or (omg.max() > inn_angles.max()):
            print('\nWe need to simulate larger angles!\n')

        #Loop through angles and interpolate where those angles match
        for i in range(1, len(inn_angles)-1):
            for j in range(len(out_angles)-1):

                #Current angles
                inn_min = (inn_angles[i-1] + inn_angles[i])/2
                inn_max = (inn_angles[i+1] + inn_angles[i])/2
                out_min = out_angles[j]
                out_max = out_angles[j+1]

                #Get values in this angle region
                aind = np.where((omg >= inn_min) & (omg <= inn_max) &
                    (psi >= out_min) & (psi <= out_max))[0]

                #Skip if empty
                if len(aind) == 0:
                    continue

                #Interpolate this region
                sfld[aind] = np.interp(dd[aind], xx, sf[i,j], left=0j, right=0j)
                pfld[aind] = np.interp(dd[aind], xx, pf[i,j], left=0j, right=0j)

        #Multiply all fields by cos(tilt)
        sfld *= np.cos(np.hypot(omg, psi))
        pfld *= np.cos(np.hypot(omg, psi))

        #Cleanup
        del omg, psi, xx, sf, pf, aind

        return sfld, pfld

    ############################################

    def interpolate_file_gap(self, dd, gw, wave):

        #Load data from file and build interpolation function for current wavelength
        with h5py.File(f'{self.maxwell_file}.h5', 'r') as f:

            #Interpolate edge data for all (gaps will be overwritten -- just easier this way)
            xx = f[f'{wave*1e9:.0f}_x'][()]
            sf = f[f'{wave*1e9:.0f}_s'][()]
            pf = f[f'{wave*1e9:.0f}_p'][()]
            sfld = np.interp(dd, xx, sf, left=0j, right=0j)
            pfld = np.interp(dd, xx, pf, left=0j, right=0j)

            #Get widths
            widths = f['ww'][()]

            #Loop through widths and interpolate over gaps
            for i in range(len(widths)-1):
                #Get gap data
                xx = f[f'{wave*1e9:.0f}_gap_{i}_x'][()]
                sf = f[f'{wave*1e9:.0f}_gap_{i}_s'][()]
                pf = f[f'{wave*1e9:.0f}_gap_{i}_p'][()]

                #Get values in this gap region
                gind = np.where((gw >= widths[i]) & (gw < widths[i+1]))[0]

                #Skip if empty
                if len(gind) == 0:
                    continue

                #Interpolate this region
                sfld[gind] = np.interp(dd[gind], xx, sf, left=0j, right=0j)
                pfld[gind] = np.interp(dd[gind], xx, pf, left=0j, right=0j)

        #Cleanup
        del xx, sf, pf, gind, widths

        return sfld, pfld

############################################
############################################
