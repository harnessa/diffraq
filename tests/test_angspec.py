"""
test_angspec.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-05-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test basics of angular spectrum method

"""

import diffraq
import numpy as np
import finufft
import diffraq.utils.image_util as image_util
from diffraq.quadrature import polar_quad
from scipy.special import j1

class Test_Angspec(object):

    tol = 1e-2
    fft_tol = 1e-9

    def run_all_tests(self):
        for tt in ['diffraction', 'focusing']:
            getattr(self, f'test_{tt}')()

############################################

    def test_diffraction(self):
        #Input
        num_pts = 1024
        wave = 0.6e-6
        width = 12.5e-3
        zz = 50

        #Derived
        dx = width/num_pts
        zcrit = 2*num_pts*dx**2/wave
        kk = 2*np.pi/wave

        #Input field
        u0 = np.ones((num_pts, num_pts))
        u0, _ = image_util.round_aperture(u0)

        #Source coordinates
        xx = (np.arange(num_pts) - num_pts/2) * dx
        xx = np.tile(xx, (num_pts, 1))
        yy = xx.T.flatten()
        xx = xx.flatten()

        #Calculate bandwidth
        if zz < zcrit:
            bf = 1/dx
        elif zz >= 3*zcrit:
            bf = np.sqrt(2*num_pts/(wave*zz))
        else:
            bf = 2*num_pts*dx/(wave*zz)

        #Get gaussian quad
        fx, fy, wq = polar_quad(lambda t: np.ones_like(t)*bf/2, num_pts, num_pts)

        #Get transfer function
        fz2 = 1. - (wave*fx)**2 - (wave*fy)**2
        evind = fz2 < 0
        Hn = np.exp(1j* kk * zz * np.sqrt(np.abs(fz2)))
        Hn[evind] = 0

        #scale factor
        scl = 2*np.pi

        for nutype in ['12', '33']:

            if nutype == '12':

                #Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
                angspec = finufft.nufft2d2(fx*scl*dx, fy*scl*dx, u0, isign=-1, eps=self.fft_tol)

                #Get solution with inverse NUFFT (nonuniform -> uniform)
                uu = finufft.nufft2d1(fx*scl*dx,  fy*scl*dx, angspec*Hn*wq, \
                    n_modes=(num_pts, num_pts), isign=1, eps=self.fft_tol)

            else:

                #Calculate angspectrum of input with NUFFT (nonuniform -> nonuniform)
                angspec = finufft.nufft2d3(xx, yy, u0.flatten(), fx*scl, fy*scl, \
                    isign=-1, eps=self.fft_tol)

                #Get solution with inverse NUFFT (nonuniform -> nonuniform)
                uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, xx, yy, \
                    isign=1, eps=self.fft_tol)
                uu = uu.reshape(u0.shape)

            #Normalize
            uu *= dx**2

            #Trim
            uu = uu[len(uu)//2]

            #Calculate analytic solution
            xpts = (np.arange(num_pts) - num_pts/2) * dx
            utru = diffraq.utils.solution_util.calculate_circle_solution(xpts, \
                wave, zz, 1e19, width/2, False)

            #Assert true
            assert(abs(uu - utru).max() < self.tol)

        #Cleanup
        del fx, fy, wq, xx, yy, u0, uu, utru, xpts, Hn, angspec, fz2, evind

############################################

    def test_focusing(self):

        #Input
        num_pts = 512
        wave = 0.6e-6
        width = 5e-3
        focal_length = 400e-3

        zz = focal_length

        #Derived
        dx = width/num_pts
        zcrit = 2*num_pts*dx**2/wave
        kk = 2*np.pi/wave

        #Input field
        u0 = np.ones((num_pts, num_pts)) + 0j
        u0, _ = image_util.round_aperture(u0)

        #Source coordinates
        xx = (np.arange(num_pts) - num_pts/2) * dx

        #Add lens phase
        u0 *= np.exp(-1j*kk/(2*focal_length)*(xx**2 + xx[:,None]**2))

        #Target coordinates
        fov = wave/width * zz * 10
        ox = (np.arange(num_pts)/num_pts - 1/2) * fov
        dox = ox[1]-ox[0]
        ox = np.tile(ox, (num_pts, 1))
        oy = ox.T.flatten()
        ox = ox.flatten()

        #Calculate bandwidth
        if zz < zcrit:
            bf = 1/dx
        elif zz >= 3*zcrit:
            bf = np.sqrt(2*num_pts/(wave*zz))
        else:
            bf = 2*num_pts*dx/(wave*zz)

        #Get gaussian quad
        fx, fy, wq = polar_quad(lambda t: np.ones_like(t)*bf/2, num_pts, num_pts)

        #Get transfer function
        fz2 = 1. - (wave*fx)**2 - (wave*fy)**2
        evind = fz2 < 0
        Hn = np.exp(1j* kk * zz * np.sqrt(np.abs(fz2)))
        Hn[evind] = 0

        #scale factor
        scl = 2*np.pi

        #Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
        angspec = finufft.nufft2d2(fx*scl*dx, fy*scl*dx, u0, isign=-1, eps=self.fft_tol)

        #Get solution with inverse NUFFT (nonuniform -> nonuniform)
        uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, ox, oy, isign=1, eps=self.fft_tol)
        uu = uu.reshape(u0.shape)

        #Normalize
        uu *= dx**2

        #Turn into intensity
        uu = np.real(uu.conj()*uu)

        #Theoretical Airy Disk
        xa = kk*width/2*np.hypot(ox, oy)/zz
        xa[np.isclose(xa,0)] = 1e-16
        area = np.pi*width**2/4
        I0 = area**2/wave**2/zz**2
        airy = I0*(2*j1(xa)/xa)**2
        airy = airy.reshape(u0.shape)

        #Assert true
        assert(abs(uu - airy).max()/I0 < self.tol)

        #Cleanup
        del fx, fy, wq, ox, oy, u0, uu, airy, xa, Hn, angspec, fz2, evind

############################################

if __name__ == '__main__':

    ta = Test_Angspec()
    ta.run_all_tests()
