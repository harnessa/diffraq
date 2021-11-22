import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()
import finufft
import time
import diffraq.utils.image_util as image_util
from diffraq.quadrature import polar_quad
from scipy.special import j1

name = ['AC508-150-A-ML', 'AC127-019-A-ML', 'AC064-015-A-ML', \
    'EO_35996-150', 'EO_84328-015'][-1]

lens = diffraq.diffraction.Lens(name)

#Input
num_pts = 512
wave = 0.68e-6
width = lens.diameter /10
tol = 1e-9
focal_length = lens.efl
defocus = 0
# defocus = -0.5e-3

zz = focal_length + defocus

#Derived
dx = width/num_pts
zcrit = 2*num_pts*dx**2/wave
kk = 2*np.pi/wave

#Input field
u0 = np.ones((num_pts, num_pts)) + 0j
u0 = image_util.round_aperture(u0)

#Source coordinates
xx = (np.arange(num_pts) - num_pts/2) * dx

#Add lens phase
rads = np.hypot(xx, xx[:,None])
lens_phs = np.exp(1j*kk*lens.opd_func(rads))
u0 *= lens_phs
phs2 = np.exp(-1j*kk/(2*focal_length)*(xx**2 + xx[:,None]**2))
# u0 *= phs2


# plt.plot(np.angle(lens_phs[len(lens_phs)//2]))
# plt.plot(np.angle(phs2[len(phs2)//2]), '--')
# breakpoint()

#Target coordinates
# fov = wave/width * zz * 100
# ox1d = (np.arange(num_pts)/num_pts - 1/2) * fov
ox1d = (np.arange(num_pts)- num_pts/2) * 13e-6
dox = ox1d[1]-ox1d[0]
ox = np.tile(ox1d, (num_pts, 1))
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
angspec = finufft.nufft2d2(fx*scl*dx, fy*scl*dx, u0, isign=-1, eps=tol)

#Get solution with inverse NUFFT (nonuniform -> nonuniform)
uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, ox, oy, isign=1, eps=tol)
uu = uu.reshape(u0.shape)

#Normalize
uu *= dx**2
uu = abs(uu)**2

#Theoretical Airy Disk
xa = kk*width/2*np.hypot(ox, oy)/zz
xa[np.isclose(xa,0)] = 1e-16
area = np.pi*width**2/4
I0 = area**2/wave**2/zz**2
airy = I0*(2*j1(xa)/xa)**2
airy = airy.reshape(u0.shape)

print(abs(airy -uu).max()/I0)

# plt.imshow(abs(uu)**2)
plt.figure()
plt.semilogy(ox1d, uu[len(uu)//2])
plt.semilogy(ox1d, airy[len(airy)//2], '--')


breakpoint()
