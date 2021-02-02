"""
solution_util.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-02-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Functions to compute the analytic solution for a circular occulter.
"""

import numpy as np
from diffraq.utils import misc_util
from scipy.special import jn

#FIXED parameters
vu_brk = 0.99
n_lom = 150

############################################
##### Circle Analytical Functions #####
############################################

def calculate_circle_solution(ss, wave, zz, z0, circle_rad, is_opaque):
    """Calculate analytic solution to circular disk over observation points ss."""

    #Derived
    kk = 2.*np.pi/wave
    zeff = zz * z0 / (zz + z0)

    #Lommel variables
    uu = kk*circle_rad**2./zeff
    vv = kk*ss*circle_rad/zz

    #Get value of where to break geometric shadow
    vu_val = np.abs(vv/uu)

    #Calculate inner region (shadow for disk, illuminated for aperture)
    vv_inn = vv[vu_val <= vu_brk]
    ss_inn = ss[vu_val <= vu_brk]
    EE_inn = get_field(uu, vv_inn, ss_inn, kk, zz, z0, is_opaque=is_opaque, is_shadow=is_opaque)

    #Calculate outer region (illuminated for disk, shadow for aperture)
    vv_out = vv[vu_val >= 2-vu_brk]
    ss_out = ss[vu_val >= 2-vu_brk]
    EE_out = get_field(uu, vv_out, ss_out, kk, zz, z0, is_opaque=is_opaque, is_shadow=not is_opaque)

    #Calculate at edge of geometric shadow with higher term lommel functions
    vv_mix = vv[(vu_val > vu_brk) & (vu_val < 2-vu_brk)]
    ss_mix = ss[(vu_val > vu_brk) & (vu_val < 2-vu_brk)]
    EE_mix = get_field(uu, vv_mix, ss_mix, kk, zz, z0, is_opaque=is_opaque, is_shadow=True)

    #Combine regions
    EE = np.concatenate((EE_inn,EE_mix,EE_out))
    ss = np.concatenate((ss_inn,ss_mix,ss_out))

    #Sort by xvalue
    EE = EE[np.argsort(ss)]
    ss = ss[np.argsort(ss)]

    return EE

def get_field(uu, vv, ss, kk, zz, z0, is_opaque=True, is_shadow=True):

    #Return empty if given empty
    if len(ss) == 0:
        return np.array([])

    #Shadow or illumination? Disk or Aperture?
    if (is_shadow and is_opaque) or (not is_shadow and not is_opaque):
        AA, BB = lommels_V(uu, vv, nt=n_lom)
    else:
        BB, AA = lommels_U(uu, vv, nt=n_lom)

    #Flip sign for aperture
    if not is_opaque:
        AA *= -1.

    #Calculate field due to mask QPF phase term
    EE = np.exp(1j*uu/2.)*(AA + 1j*BB*[1.,-1.][is_shadow])

    #Add illuminated beam
    if not is_shadow:
        EE += np.exp(-1j*vv**2./(2.*uu))

    #Add final plane QPF phase terms (diffraq solution ignores plane wave phase)
    EE *= np.exp(1j*kk*ss**2./(2.*zz))
    
    #Scale for diverging beam
    EE *= z0 / (zz + z0)

    return EE

def lommels_V(u,v,nt=10):
    VV_0 = 0.
    VV_1 = 0.
    for m in range(nt):
        VV_0 += (-1.)**m*(v/u)**(0+2.*m)*jn(0+2*m,v)
        VV_1 += (-1.)**m*(v/u)**(1+2.*m)*jn(1+2*m,v)
    return VV_0, VV_1

def lommels_U(u,v,nt=10):
    UU_1 = 0.
    UU_2 = 0.
    for m in range(nt):
        UU_1 += (-1.)**m*(u/v)**(1+2.*m)*jn(1+2*m,v)
        UU_2 += (-1.)**m*(u/v)**(2+2.*m)*jn(2+2*m,v)
    return UU_1, UU_2

############################################
############################################

############################################
####	Direct Integration  ####
############################################

def direct_integration(fresnum, u_shp, xq, yq, wq, grid_2D):
    lambdaz = 1./fresnum
    utru = np.empty(u_shp) + 0j
    for j in range(u_shp[0]):
        for k in range(u_shp[1]):
            utru[j,k] = 1/(1j*lambdaz) * np.sum(np.exp((1j*np.pi/lambdaz)* \
                ((xq - grid_2D[j,k])**2 + (yq - grid_2D[k,j])**2))*wq)
    return utru

############################################
############################################
