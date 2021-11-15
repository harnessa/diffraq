"""
test_angspec.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-10-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Test basics of non-unifrom angular spectrum calculation.

"""

import diffraq
import numpy as np

class Test_angspec(object):

    radius = 12.e-3
    zz = 50.
    z0 = 27.5
    wave = 0.6e-6
    n = 1000
    m = 200            #Number of quadrature points
    fft_tol = 1e-12
    tol = 1e-3
    num_pts = 512
    tel_diameter = 5e-3

    def run_all_tests(self):
        for tt in ['diffraction']:
            getattr(self, f'test_{tt}')()

############################################

    def test_diffraction(self):

        # #Get FFT solution #Not used
        # apts, grid_pts, tru_asp, tru_uu = self.get_fft_soln()

        #Get quadrature
        xq, yq, wq = diffraq.quadrature.polar_quad(lambda t: \
            np.ones_like(t)*self.radius, self.m, self.n)

        #Get spatial points
        xpts = diffraq.utils.image_util.get_grid_points(self.num_pts, self.tel_diameter)

        #Get sampling requirements
        Dmax = 2*self.radius
        ovr = 4
        D = (Dmax/2 + self.tel_diameter/2)
        dang = 1/(2*Dmax) / ovr
        amax = max(D/np.hypot(self.zz,D)/self.wave, Dmax/(self.wave*self.zz)) * ovr
        nang = int(np.ceil(2*amax/dang/2)) * 2

        #Get angular spectrum points
        apts = (np.arange(nang) - nang/2)*dang * self.wave

        #Loop through plane and spherical wave
        for z0 in [1e19, self.z0]:

            #Get initial field
            u0 = np.exp(1j*2*np.pi/(2*self.wave*z0)*(xq**2 + yq**2))

            #Calculate angular spectrum
            Aspec = diffraq.diffraction.calculate_angspec(\
                xq, yq, wq, u0, self.wave, apts, self.fft_tol)

            #Calculate diffraction
            uans1 = diffraq.diffraction.diffract_from_angspec(Aspec, self.wave, self.zz, \
                apts, xpts, self.fft_tol)
            uans1 = uans1[len(uans1)//2]

            #Calculate straight diffraction
            uans2 = diffraq.diffraction.diffract_angspec(xq, yq, wq, u0, Dmax, \
                self.wave, self.zz, xpts, self.fft_tol)
            uans2 = uans2[len(uans2)//2]

            #Calculate analytic solution
            utru = diffraq.utils.solution_util.calculate_circle_solution(xpts, \
                self.wave, self.zz, z0, self.radius, False)

            #Assert close to theoretical
            assert(abs(utru - uans1).max() < self.tol)
            assert(abs(utru - uans2).max() < self.tol)

############################################

    def get_fft_soln(self):
        """ Not used """
        #Setup numerics
        NN = int(self.num_pts*self.num_pad)
        dx = 2*self.radius / self.num_pts
        xx = (np.arange(self.num_pts) - self.num_pts/2)*dx

        #Calculate frequencies
        kk = 2.*np.pi/self.wave
        kx = np.fft.fftshift(np.fft.fftfreq(NN, d=dx))
        kz2 = kk**2. - (kx[:,None]**2. + kx**2.)*(2*np.pi)**2

        #Propagation wavenumber
        kz = np.sqrt(np.abs(kz2)) + 0j
        kz[kz2 < 0] *= 1j

        nn = (np.arange(self.num_pts) - self.num_pts/2)
        rho = np.hypot(nn, nn[:,None])

        #Build init field
        U0 = np.ones((self.num_pts, self.num_pts)) + 0j
        # U0 = np.exp(1j*kk/(2*self.z0)*rho**2)

        U0[rho >= self.num_pts/2] = 0
        U0 = diffraq.utils.image_util.pad_array(U0, NN)

        #Get angspec
        fn = np.fft.fftshift(np.fft.fft2(U0))

        #Propagate
        Ue = np.fft.ifft2(np.fft.ifftshift(fn * np.exp(1j*kz*self.zz)))

        #Crop
        Ue = diffraq.utils.image_util.crop_image(Ue, None, self.num_pts//2)

        #Normalize
        fn *= dx**2

        #Cleanup
        del U0, kz2, kz

        return kx, xx, fn, Ue

############################################

if __name__ == '__main__':

    ta = Test_angspec()
    ta.run_all_tests()
