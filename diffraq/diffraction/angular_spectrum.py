"""
angular_spectrum.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-10-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: diffraction calculation of input quadrature to target grid using
    the angular spectrum method.

"""

import numpy as np
import finufft

def diffract_angspec(xq, yq, wq, u0, Dmax, wave, zz, grid_pts, tol, over_sample=4, is_babinet=False):
    """
    uu = diffract_angspec(aspec, wave, zz, ang_pts, grid_pts, tol):

    calculates diffraction propagation from input quadrature to target grid
        using the angular spectrum method and nufft

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        u0 = incident field
        wave = wavelength [meters]
        zz = propagation distance
        Dmax = max width of occulter
        grid_pts = target grid
        tol = tolerance to which to calculate FFT (via finufft)
        is_babinet = calculation implies Babinet's Principle and need to subtract 1?
        over_sample = over sampling of Nyquist

    Outputs:
        uu = complex field over target grid
    """

    #Get sampling requirements
    ngrid = len(grid_pts)
    dgrid = grid_pts[1] - grid_pts[0]
    dangs = 1/(2*Dmax*over_sample)
    D = (Dmax/2 + grid_pts.max())       #Maximum extent

    #Get angular spectrum points
    amax = max(D/np.hypot(zz,D)/wave, Dmax/(wave*zz)) * over_sample
    nangs = int(np.ceil(2*amax/dangs/2)) * 2
    angs_pts = (np.arange(nangs) - nangs/2)*dangs * wave
    dangs = angs_pts[1] - angs_pts[0]

    ###################################
    ### Angular Spectrum ###
    ###################################

    #Scale factor to become FT
    sc = 2.*np.pi/wave
    #Scaled grid spacing
    dka = sc * dangs

    #Max NU coord
    maxNU = dka * max(np.abs(xq).max(), np.abs(yq).max())

    #only if needed
    if maxNU > 3*np.pi:
        print('too coarse angle')
        #Wrap in case grid too coarse
        xq = np.mod(xq, 2*np.pi/dka)
        yq = np.mod(yq, 2*np.pi/dka)

    #Compute strengths
    cq = u0 * wq

    #Do FINUFFT
    aspec = finufft.nufft2d1(dka*xq, dka*yq, cq, (nangs, nangs), isign=1, eps=tol)

    ###################################
    ### Diffraction ###
    ###################################

    #Scaled grid spacing
    dkg = sc * dgrid

    #Get angular coordinates (assume symmetric grid input)
    abq = angs_pts.copy()

    #Max NU coord
    maxNU = dkg * np.abs(abq).max()

    #only if needed
    if maxNU > 3*np.pi:
        print('too coarse diff')
        #Wrap in case grid too coarse
        abq = np.mod(abq, 2*np.pi/dkg)

    #Get propagation constant
    kz2 = 1 - abq**2 - abq[:,None]**2
    kz = 2*np.pi/wave * np.sqrt(np.abs(kz2)) + 0j
    kz[kz2 < 0] *= 1j

    #Propagation kernel * aspec as strengths
    cq = aspec * np.exp(1j*kz*zz)
    cq = cq.flatten()

    #Cleanup
    del kz, kz2

    #Tile ang points
    abq = np.tile(abq, (nangs, 1))

    #Do FINUFFT (inverse)
    uu = finufft.nufft2d1(dkg*abq.flatten(), dkg*abq.T.flatten(), cq, \
        (ngrid, ngrid), isign=-1, eps=tol)

    #Subtract from Babinet field
    if is_babinet:
        uu = u0 - uu

    #Transpose to match visual representation
    uu = uu.T

    #Normalize
    uu *= dangs**2/wave**2

    #Cleanup
    del cq, abq, angs_pts

    return uu


def diffract_from_angspec(aspec, wave, zz, ang_pts, grid_pts, tol):
    """
    uu = diffract_from_angspec(aspec, wave, zz, ang_pts, grid_pts, tol):

    calculates diffraction propagation from input quadrature to target grid
        using the angular spectrum method and nufft

    Inputs:
        aspec = complex angular spectrum over target angular spectrum grid
        wave = wavelength [meters]
        zz = propagation distance
        ang_pts = target angular spectrum grid
        grid_pts = target grid
        tol = tolerance to which to calculate FFT (via finufft)

    Outputs:
        uu = complex field over target grid
    """

    #Scale factor to become FT
    sc = 2.*np.pi/wave
    #Scaled grid spacing
    dk = sc * (grid_pts[1] - grid_pts[0])
    ngrid = len(grid_pts)

    #Get angular coordinates (assume symmetric grid input)
    abq = ang_pts.copy()

    #Max NU coord
    maxNU = dk * np.abs(abq).max()

    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        abq = np.mod(abq, 2*np.pi/dk)

    #Get propagation constant
    kz2 = 1 - abq**2 - abq[:,None]**2
    kz = 2*np.pi/wave * np.sqrt(np.abs(kz2)) + 0j
    kz[kz2 < 0] *= 1j

    #Propagation kernel * aspec as strengths
    cq = aspec * np.exp(1j*kz*zz)
    cq = cq.flatten()

    #Cleanup
    del kz, kz2

    #Tile ang points
    abq = np.tile(abq, (len(ang_pts), 1))

    #Do FINUFFT (inverse)
    uu = finufft.nufft2d1(dk*abq.flatten(), dk*abq.T.flatten(), cq, \
        (ngrid, ngrid), isign=1, eps=tol)

    #Normalize
    uu *= (ang_pts[1] - ang_pts[0])**2/wave**2

    #Cleanup
    del cq, abq

    return uu

def calculate_angspec(xq, yq, wq, u0, wave, ang_pts, tol):
    """
    aspec = calculate_angspec(xq, yq, wq, wave, ang_pts, tol)

    calculates angular spectrum from input quadrature to target angular spectrum
        grid using finufft.

    Inputs:
        xq, yq = x,y coordinates of area quadrature nodes [meters]
        wq = area quadrature weights
        u0 = incident field
        wave = wavelength [meters]
        ang_pts = target angular spectrum grid
        tol = tolerance to which to calculate FFT (via finufft)

    Outputs:
        aspec = complex angular spectrum over target angular spectrum grid
    """

    #Scale factor to become FT
    sc = 2.*np.pi/wave
    #Scaled grid spacing
    dk = sc * (ang_pts[1] - ang_pts[0])
    ngrid = len(ang_pts)

    #Max NU coord
    maxNU = dk * max(np.abs(xq).max(), np.abs(yq).max())

    #only if needed
    if maxNU > 3*np.pi:
        #Wrap in case grid too coarse
        xq = np.mod(xq, 2*np.pi/dk)
        yq = np.mod(yq, 2*np.pi/dk)

    #Compute strengths
    cq = u0 * wq

    #Do FINUFFT
    aspec = finufft.nufft2d1(dk*xq, dk*yq, cq, (ngrid, ngrid), isign=-1, eps=tol)

    #Cleanup
    del cq

    return aspec
