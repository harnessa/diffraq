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
import diffraq.utils.image_util as image_util

class Test_Angspec(object):

    tol = 1e-2

    num_pts = 2**8
    num_pad = 2**3
    wave = 0.6e-6

    def run_all_tests(self):
        for tt in ['diffraction', 'focusing'][1:]:
            getattr(self, f'test_{tt}')()

############################################

    def test_diffraction(self):

        dz = 50
        radius = 12e-3

        #Setup and get initial field
        U0, kz, xx = self.setup(radius)

        #Propagate
        fn = np.fft.fftshift(np.fft.fft2(U0))
        Ue = np.fft.ifft2(np.fft.ifftshift(fn * np.exp(1j*kz*dz)))

        #Crop
        Ue = image_util.crop_image(Ue, None, self.num_pts//2)
        Ue = Ue[len(Ue)//2]

        #Calculate analytic solution
        utru = diffraq.utils.solution_util.calculate_circle_solution(xx, \
            self.wave, dz, 1e19, radius, False)

        import matplotlib.pyplot as plt;plt.ion()
        plt.plot(xx, abs(Ue))
        plt.plot(xx, abs(utru), '--')
        plt.figure()
        plt.plot(xx, np.angle(Ue))
        plt.plot(xx, np.angle(utru), '--')

        print(abs(Ue - utru).mean())
        breakpoint()

        #Assert true
        assert(abs(Ue - utru).mean() < self.tol)


############################################

    def test_focusing(self):

        dz = 0.5
        radius = 2.5e-3
        focal_length = dz

        #Setup and get initial field
        U0, kz, xx = self.setup(radius)

        #Add lens
        xxp = (np.arange(U0.shape[0]) - U0.shape[0]/2)*(xx[1]-xx[0])
        U0 *= np.exp(-1j*self.kk*(xxp[:,None]**2 + xxp**2)/(2*focal_length))

        #Propagate
        fn = np.fft.fftshift(np.fft.fft2(U0))
        Ue = np.fft.ifft2(np.fft.ifftshift(fn * np.exp(1j*kz*dz)))

        #Crop
        Ue = image_util.crop_image(Ue, None, self.num_pts//2)
        Ue = Ue[len(Ue)//2]

        import matplotlib.pyplot as plt;plt.ion()
        plt.plot(xx, abs(Ue))

        breakpoint()

############################################

    def setup(self, radius):

        #Setup numerics
        NN = int(self.num_pts*self.num_pad)
        dx = 2*radius / self.num_pts
        xx = (np.arange(self.num_pts) - self.num_pts/2)*dx

        #Calculate frequencies
        self.kk = 2.*np.pi/self.wave
        kx = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(NN, d=dx))
        kz2 = self.kk**2. - (kx[:,None]**2. + kx**2.)

        #Propagation wavenumber
        kz = np.sqrt(np.abs(kz2)) + 0j
        kz[kz2 < 0] *= 1j

        #Cleanup
        del kz2

        #Build init field
        U0 = np.ones((self.num_pts, self.num_pts)) + 0j
        nn = (np.arange(self.num_pts) - self.num_pts/2)
        rho = np.hypot(nn, nn[:,None])
        U0[rho >= self.num_pts/2] = 0
        U0 = image_util.pad_array(U0, NN)

        return U0, kz, xx

############################################

if __name__ == '__main__':

    ta = Test_Angspec()
    ta.run_all_tests()
