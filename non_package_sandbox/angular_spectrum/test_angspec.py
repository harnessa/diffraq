import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq.utils.image_util as image_util

class Test_AngSpec(object):

    def __init__(self):
        self.num_pts = 2**9
        self.width = 25e-3
        self.wave = 0.6e-6
        self.num_pad = 2
        self.zz = 10
        self.fft_pad = 4
        self.fl = 0.5

    def test(self):
        #Setup
        self.setup_angspec()

        #Build init field
        U0 = np.ones((self.num_pts, self.num_pts)) + 0j
        U0, dum = image_util.round_aperture(U0)
        U0_uncrop = U0.copy()
        U0 = image_util.pad_array(U0, self.NN)

        #Propagate ang spec
        Ua = self.propagate(U0.copy(), self.zz)
        plt.imshow(abs(Ua))

        #Propagate fft
        Uf = self.propagate_fft(U0_uncrop.copy(), self.zz)
        plt.figure()
        plt.imshow(abs(Uf))

        breakpoint()

    def setup_angspec(self):

        self.NN = int(self.num_pts*self.num_pad)

        #Get radii values
        self.dx = self.width / self.num_pts

        #Calculate frequencies
        self.kk = 2.*np.pi/self.wave
        self.ky = 2*np.pi*np.fft.fftshift(np.fft.fftfreq(self.NN, d=self.dx))
        kz2 = self.kk**2. - (self.ky[:,None]**2. + self.ky**2.)

        #Propagation wavenumber
        self.kz = np.sqrt(np.abs(kz2)) + 0j
        self.kz[kz2 < 0] *= 1j

        #Cleanup
        del kz2

    def propagate(self,E_in,dz):
        #Return if dz is zero
        if dz == 0.: return E_in


        #FFT initial electric field + ifftshift
        fn = np.fft.fftshift(np.fft.fft2(E_in))

        #Multiply by transfer function + fftshift
        # fm = np.fft.fftshift(fn*np.exp(1j*self.kz*dz))
        # del fn

        #Take inverse FFT + ifftshift to turn back to electric field
        # E_out = np.fft.ifftshift(np.fft.ifft2(fm))
        # del fm
        E_out = np.fft.ifft2(np.fft.ifftshift(fn * np.exp(1j*self.kz*dz)))

        return E_out

    def propagate_fft(self, U0, dz):

        #Pad
        NN = self.num_pts * self.fft_pad
        U0 = image_util.pad_array(U0, NN)

        #Get fresnel kernel
        xx = (np.arange(NN) - NN/2)*self.dx
        U0 *= np.exp(1j*self.kk*(xx**2 + xx[:,None]**2)/(2*dz))
        # U0 *= np.exp(1j*self.kk*(xx**2 + xx[:,None]**2)/(2*-self.fl))

        #Do FFT
        UU = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(U0)))

        return UU

if __name__ == '__main__':

    ta = Test_AngSpec()
    ta.test()
