"""
phase_screen.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-29-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to generate random phase screens for simulation of atmosphere
    or random mask thickness.
"""

import numpy as np
from diffraq.utils import image_util
import finufft

class Phase_Screen(object):

    def __init__(self, params, parent):
        self.parent = parent            #Beam
        self.set_parameters(params)

    def set_parameters(self, params):
        def_pms = {
            'kind':             'atmosphere',
            'spectrum':         'kolmogorov',
            'fried_r0':         0.1,               #Fried parameter [m] at 0.5e-6
            'thick_amplitude':  0,                 #Amplitude (one-sided) of thickness range
            'thick_scale':      0,                 #Size scale over which to blur thickness
        }

        for k, v in def_pms.items():
            setattr(self, k, v)

        for k, v in params.items():
            setattr(self, k, v)

        #Create stored map
        self.stored_map = None

############################################
#####  Main Script #####
############################################

    def get_field(self, xq, yq, wq, wave):

        #Get field
        fld = getattr(self, f'{self.kind}_field')(xq, yq, wq, wave)

        return fld

############################################
############################################

############################################
#####  Atmosphere #####
############################################

    def kolmogorov_spectrum(self, r0, hbf):
        return 0.49 * r0**(-5/3.) * \
            (np.hypot(self.parent.fx, self.parent.fy)*hbf)**(-11/3)

    def atmosphere_field(self, xq, yq, wq, wave):

        #Get half-bandwidth
        dx = 2*max(xq.max(), yq.max())/np.sqrt(xq.size)
        hbf = self.get_angspec_bandwidth(dx, wave)

        #Get Fried parameter
        r0 = self.fried_r0 * (wave/0.5e-6)**(6/5)

        #Get turbulence spectrum
        spec = np.sqrt(getattr(self, f'{self.spectrum}_spectrum')(r0, hbf))

        #Get random values
        rand_real = self.parent.sim.rng.normal(0, 1, spec.shape)
        rand_imag = self.parent.sim.rng.normal(0, 1, spec.shape)

        #Create random realization of amplitude (sqrt(power)) spectrum
        kernel = spec*(rand_real + 1j*rand_imag)

        #Cleanup
        del spec, rand_real, rand_imag

        #FFT scale factor
        scl = 2*np.pi*hbf

        #Take inverse NUFFT to go from spectrum to spatial (nonuniform -> nonuniform)
        fft_ans = finufft.nufft2d3(scl*self.parent.fx, scl*self.parent.fy,
            kernel*self.parent.fw*hbf**2, xq, yq, isign=1, eps=self.parent.sim.fft_tol)

        #Get field map from real solution (imaginary is also valid)
        scn_fld = np.exp(1j*fft_ans.real)

        # import matplotlib.pyplot as plt;plt.ion()
        #
        # # plt.figure(); plt.colorbar(plt.imshow(fft_ans.real.reshape((256,256))))
        # # plt.figure(); plt.colorbar(plt.imshow(np.angle(scn_fld).reshape((256,256))))
        # plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=fft_ans.real, s=1))
        # plt.figure(); plt.colorbar(plt.scatter(xq, yq, c=np.angle(scn_fld), s=1))
        # breakpoint()

        #Cleanup
        del kernel, fft_ans

        return scn_fld

############################################
############################################

############################################
#####  Thickness #####
############################################

    def thickness_field(self, xq, yq, wq, wave):

        #If not run yet, build map of thicknesses
        if self.stored_map is None:

            #Build random map of thicknesses
            thk = self.parent.sim.rng.uniform(-self.thick_amplitude, \
                self.thick_amplitude, size=xq.shape)

            # old = thk.copy() #TODO: remove

            #Get maximum frequency
            hbf = 2/self.thick_scale

            #FFT scale factor
            scl = 2*np.pi*hbf

            #Get NUFFT of thickness map (nonuniform -> nonuniform)
            thk_fft = finufft.nufft2d3(xq, yq, thk*wq, \
                scl*self.parent.fx, scl*self.parent.fy, isign=-1, eps=self.parent.sim.fft_tol)

            #Multiply by gaussian in frequency space
            thk_fft *= np.exp(-2*np.pi**2*self.thick_scale**2 * \
                (hbf*np.hypot(self.parent.fx, self.parent.fy))**2)
            # thk_fft *= 2*np.pi*self.thick_scale**2

            #Compute convolution via inverse NUFFT (nonuniform -> nonuniform)
            thk = finufft.nufft2d3(scl*self.parent.fx, scl*self.parent.fy,
                thk_fft*self.parent.fw*hbf**2, xq, yq, isign=1, eps=self.parent.sim.fft_tol).real

            #TODO: fix normalization
            #Constrain to be within thick_amplitude
            thk *= self.thick_amplitude / np.percentile(abs(thk), 99.7)

            #Store thickness
            self.stored_map = thk.copy()

            #Cleanup
            del thk, thk_fft

        #Build field from stored thickness
        fld = np.exp(1j*2*np.pi/wave * self.stored_map)


        # import matplotlib.pyplot as plt;plt.ion()   #TODO: remove
        # print(thk.min(), thk.max())
        # old = old.reshape((128,128))
        # thk = thk.reshape((128,128))
        #
        # plt.figure()
        # plt.imshow(old)
        # plt.figure()
        # plt.imshow(thk, extent=[xq.min(),xq.max(),yq.max(),yq.min()])
        #
        # plt.figure()
        # plt.hist(old.flatten(),bins=20)
        # plt.hist(thk.flatten(),bins=20,alpha=0.8)
        #
        # # plt.colorbar(plt.scatter(self.parent.fx, self.parent.fy, c=kernel, s=1))
        # breakpoint()

        return fld

############################################
############################################

############################################
#####  Gaussian #####
############################################

    def gaussian_field(self, xq, yq, wave):

        breakpoint()

############################################
############################################

############################################
#####  Misc #####
############################################

    def get_angspec_bandwidth(self, dx, wave):

        #Assume propagation distance and number of points
        zz = self.parent.sim.zz
        num_pts = int(np.sqrt(self.parent.fx.size))

        #Critical distance
        zcrit = 2*num_pts*dx**2/wave

        #Calculate bandwidth
        if zz < zcrit:
            bf = 1/dx
        elif zz >= 3*zcrit:
            bf = np.sqrt(2*num_pts/(wave*zz))
        else:
            bf = 2*num_pts*dx/(wave*zz)

        #Divide by two because radius of ang spec quadrature
        bf /= 2

        return bf

############################################
############################################
