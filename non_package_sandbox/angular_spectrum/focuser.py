import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import finufft
import time
import diffraq.utils.image_util as image_util
from diffraq.quadrature import polar_quad
from scipy.special import j1
import diffraq

#Input
num_pts = 512
wave = 0.68e-6
width = 5e-3
tol = 1e-9
focal_length = 50e-3
defocus = 0

#TODO: why doesn't it work for large apertures?
# width = 2.4
# focal_length = width*80

zz = focal_length + defocus
# zz = 140.5e-3

# zz = 124.217e-3

#Derived
dx = width/num_pts
zcrit = 2*num_pts*dx**2/wave
kk = 2*np.pi/wave

# zz = zcrit
# focal_length = zz

#Input field
u0 = np.ones((num_pts, num_pts)) + 0j
u0 = image_util.round_aperture(u0)

#Source coordinates
xx = (np.arange(num_pts) - num_pts/2) * dx

#Add lens phase
# u0 *= np.exp(-1j*kk/(2*focal_length)*(xx**2 + xx[:,None]**2))
# l1 = diffraq.diffraction.Lens_Element(\
#     {'lens_name':'AC508-150-A-ML', 'diameter':width}, num_pts)
# u0 *= np.exp(1j*kk*l1.lens_func(np.hypot(xx, xx[:,None])))

focal_length = 125e-3
R1 = 128.21e-3
thk = 3.260e-3
rr = np.hypot(xx, xx[:,None])
n1 = 1.514
opd = (R1 - np.sqrt(R1**2 - rr**2))*(1 - n1)*2 + n1*thk
# opd = -rr**2/(2*focal_length)
u0 *= np.exp(1j*kk*opd)

zback = 124.217e-3
zpp = focal_length*(n1 - 1)*thk/R1/n1
zz = zback + zpp/2

# focal_length = 50e-3
# R1 = 50.6e-3
# thk = 5.24e-3
# rr = np.hypot(xx, xx[:,None])
# n1 = 1.514
# opd = (R1 - np.sqrt(R1**2 - rr**2))*(1 - n1)*2 + n1*thk
# # opd = -rr**2/(2*focal_length)
# u0 *= np.exp(1j*kk*opd)
#
# zback = 48.373e-3
# zpp = focal_length*(n1 - 1)*thk/R1/n1
# zz = zback + zpp/2


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

print(abs(uu - airy).max())


# plt.imshow(abs(uu)**2)
plt.figure()
plt.semilogy(uu[len(uu)//2])
plt.semilogy(airy[len(airy)//2], '--')


breakpoint()
