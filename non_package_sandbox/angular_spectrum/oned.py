import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq.utils.image_util as image_util


N0 = 512
wave = 0.6e-6
width = 5e-3
zz = 10e-3

def get_uu(num_pts, num_pad):

    dx = width/num_pts
    kk = 2.*np.pi/wave

    xx = (np.arange(num_pts) - num_pts//2)*dx

    NN = num_pts * num_pad

    zcrit = num_pad*num_pts * dx**2/wave
    print(num_pts, num_pad, zcrit)

    #Calculate frequencies
    fx = np.fft.fftshift(np.fft.fftfreq(NN, d=dx))
    fz2 = 1. - (wave*fx)**2

    #Propagation wavenumber
    kz = kk * np.sqrt(np.abs(fz2)) + 0j
    evind = fz2 < 0
    Hn = np.exp(1j*kz*zz)
    Hn[evind] = 0

    u0 = np.pad(np.ones(num_pts), num_pts*(num_pad-1)//2)

    fn = np.fft.fftshift(np.fft.fft(u0))
    uu = np.fft.ifft(np.fft.ifftshift(fn * Hn))

    uu = uu[NN//2-num_pts//2:NN//2+num_pts//2]

    return uu, xx

u1p, x1p = get_uu(N0, 2)
u2p, x2p = get_uu(N0, 4)

u1n, x1n = get_uu(N0, 2)
u2n, x2n = get_uu(N0*2, 2)

fig, axes = plt.subplots(2, 2, figsize=(9,9), sharex=True)

axes[0,0].plot(x1p, abs(u1p))
axes[0,0].plot(x2p, abs(u2p), '--')
axes[0,1].plot(x1p, np.angle(u1p))
axes[0,1].plot(x2p, np.angle(u2p), '--')

axes[1,0].plot(x1n, abs(u1n))
axes[1,0].plot(x2n, abs(u2n), '--')
axes[1,1].plot(x1n, np.angle(u1n))
axes[1,1].plot(x2n, np.angle(u2n), '--')


plt.figure()
for m in range(8, 16)[::-1]:
    uu, xx = get_uu(2**m, 2)
    plt.plot(xx, abs(uu), label=m)

plt.legend()

breakpoint()
