import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import finufft
import time

#Input
num_pts = 1024
wave = 0.5e-6
# width = 5e-3
width = num_pts*1e-6
zz = 300e-3
maper = 0.8
tol=1e-9

#Derived
dx = width/num_pts
zcrit = 2*num_pts*dx**2/wave
kk = 2*np.pi/wave
xx = (np.arange(num_pts) - num_pts/2) * dx

zz = zcrit *4

#Input field
u0 = np.ones(num_pts)
u0[abs(xx) >= width*maper/2] = 0

#Calculate bandwidth
if zz < zcrit:
    bf = 1/dx
elif zz >= 3*zcrit:
    bf = np.sqrt(2*num_pts/(wave*zz))
else:
    bf = 2*num_pts*dx/(wave*zz)

#Get gaussian quad
fx, wq = np.polynomial.legendre.leggauss(num_pts*2)

#Linear map from [-1,1] to [a,b]
fx = (-bf/2*(1-fx) + bf/2*(1+fx))/2
wq *= bf * dx / 2                       #/2 for gauss quad; 

# fx = (np.arange(num_pts)/num_pts - 0.5) * bf
# wq = np.ones(num_pts)/num_pts

tik = time.perf_counter()

#Rescale fx to [-pi, pi)
ffx = fx*dx*2*np.pi

#Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
angspec = finufft.nufft1d2(ffx, u0, eps=tol)
angspec5 = finufft.nufft1d3(xx, u0, fx*2*np.pi, isign=-1, eps=tol)

#Get transfer function
fz2 = 1. - (wave*fx)**2
evind = fz2 < 0
Hn = np.exp(1j* kk * zz * np.sqrt(np.abs(fz2)))
Hn[evind] = 0

#Band-limited
df = bf/num_pts
fband = 1/wave/np.sqrt((2*df*zz)**2 + 1)
# Hn[abs(fx) > fband] = 0

#Get solution with inverse NUFFT (nonuniform -> uniform)
uu = finufft.nufft1d1(ffx, angspec*Hn*wq, n_modes=num_pts, isign=1, eps=tol)
uu5 = finufft.nufft1d3(fx*2*np.pi, angspec5*Hn*wq, xx, isign=1, eps=tol)

tok = time.perf_counter()
print(f'NUFFT Time: {tok-tik:.2f} s')

#Uniform FFT solution
num_pad = 1
fx2 = np.fft.fftshift(np.fft.fftfreq(num_pts*num_pad, d=dx))
fz2 = 1. - (wave*fx2)**2
evind = fz2 < 0
Hn2 = np.exp(1j*kk*zz*np.sqrt(np.abs(fz2)) + 0j)
Hn2[evind] = 0
u02 = np.pad(u0, num_pts*(num_pad-1)//2)
fn2 = np.fft.fftshift(np.fft.fft(u02))
uu2 = np.fft.ifft(np.fft.ifftshift(fn2 * Hn2))
uu2 = uu2[num_pts*num_pad//2-num_pts//2:num_pts*num_pad//2+num_pts//2]

tik = time.perf_counter()
num_pad *= 20
fx3 = np.fft.fftshift(np.fft.fftfreq(num_pts*num_pad, d=dx))
fz3 = 1. - (wave*fx3)**2
evind = fz3 < 0
Hn3 = np.exp(1j*kk*zz*np.sqrt(np.abs(fz3)) + 0j)
Hn3[evind] = 0
u03 = np.pad(u0, num_pts*(num_pad-1)//2)
fn3 = np.fft.fftshift(np.fft.fft(u03))
uu3 = np.fft.ifft(np.fft.ifftshift(fn3 * Hn3))
uu3 = uu3[num_pts*num_pad//2-num_pts//2:num_pts*num_pad//2+num_pts//2]
tok = time.perf_counter()
print(f'FFT Time: {tok-tik:.2f} s')

print(f'{abs(uu3 - uu5).max():.3e}, {abs(uu3 - uu).max():.3e}')
# print(abs(uu3).max(), abs(uu).max(), abs(uu5).max())

plt.figure()
plt.plot(xx, abs(uu))
plt.plot(xx, abs(uu5))
plt.plot(xx, abs(uu3), '--')

# plt.figure()
# plt.plot(xx, np.angle(uu))
# plt.plot(xx, np.angle(uu5))
# plt.plot(xx, np.angle(uu3), '--')


breakpoint()
