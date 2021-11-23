import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()
import finufft
import time
import diffraq.utils.image_util as image_util
from diffraq.quadrature import polar_quad
from scipy.special import j1


#Input
num_pts = 512
wave = 0.641e-6
width = 5e-3
tol = 1e-9
focal_length = 498e-3


#Distances between lenses
# z12 = 167.1e-3
# z2f = 51.4e-3
# z12 = 167.84e-3 + 0.5e-3
# z2f = 59.85e-3 - 10e-3

z12 = 159.430e-3
z2f = 63.034e-3

z12 += 7.e-3
z2f += 1.5e-3

#Lenses
l1 = diffraq.diffraction.Lens_Element(\
    {'lens_name':'AC508-150-A-ML', 'diameter':width}, num_pts)
l2 = diffraq.diffraction.Lens_Element(\
    {'lens_name':'AC064-015-A-ML', 'diameter':width/2}, num_pts)

# f1 = l1.focal_length
# fb1 = 140.4e-3
# zpp1 = f1 - fb1
# z12 += zpp1#/2
#
# f2 = l2.focal_length
# fb2 = 13e-3
# zpp2 = f2 - fb2
# z2f += zpp2#/2



# o1 = 77.455
# f1 = l1.focal_length
# f2 = l2.focal_length
# # f1b = 140.4e-3
# # f2b = 13.6e-3
# i1 = o1*f1/(o1 - f1)
# o2 = z12 - i1
# i2 = o2*f2/(o2 - f2)
# # z2f = i2
#
# # l1 = diffraq.diffraction.Lens({'lens_name':'EO_35996-150'})
# # l2 = diffraq.diffraction.Lens({'lens_name':'EO_84328-015'})
# # z12 = 168.95e-3
# # z2f = 56.8e-3

#Derived
kk = 2*np.pi/wave
dx1 = l1.dx
dx2 = l2.dx
zcrit1 = 2*num_pts*dx1**2/wave
zcrit2 = 2*num_pts*dx2**2/wave

#Source coordinates
x1 = (np.arange(num_pts) - num_pts/2) * dx1

#Input field
u0 = np.ones((num_pts, num_pts)) + 0j
# u0 = np.exp(1j*2*np.pi/wave/2/77.455*(x1**2 + x1[:,None]**2))


#########################
### First Propagation ###
#########################

#Apply first lens
rr1 = np.hypot(x1, x1[:,None])
lens_phs = np.exp(1j*kk*l1.lens_func(rr1))
# lens_phs = np.exp(-1j*kk*rr1**2/2/0.15)
u0 *= lens_phs
u0 = image_util.round_aperture(u0)

#Target coordinates
ox2_1D = (np.arange(num_pts) - num_pts/2) * dx2
ox2 = np.tile(ox2_1D, (num_pts, 1))
oy2 = ox2.T.flatten()
ox2 = ox2.flatten()

#Calculate bandwidth
if z12 < zcrit1:
    bf1 = 1/dx1
elif z12 >= 3*zcrit1:
    bf1 = np.sqrt(2*num_pts/(wave*z12))
else:
    bf1 = 2*num_pts*dx1/(wave*z12)

#Get gaussian quad
fx, fy, wq = polar_quad(lambda t: np.ones_like(t)*bf1/2, num_pts, num_pts)

#Get transfer function
fz2 = 1. - (wave*fx)**2 - (wave*fy)**2
evind = fz2 < 0
Hn = np.exp(1j* kk * z12 * np.sqrt(np.abs(fz2)))
Hn[evind] = 0

#scale factor
scl = 2*np.pi

#Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
angspec = finufft.nufft2d2(fx*scl*dx1, fy*scl*dx1, u0, isign=-1, eps=tol)

#Get solution with inverse NUFFT (nonuniform -> nonuniform)
u2 = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, ox2, oy2, isign=1, eps=tol)
u2 = u2.reshape(u0.shape)

#Normalize
u2 *= dx1**2

#########################
### Second Propagation ###
#########################

#Apply second lens
rr2 = np.hypot(ox2_1D, ox2_1D[:,None])
lens_phs = np.exp(1j*kk*l2.lens_func(rr2))
# lens_phs = np.exp(-1j*kk*rr2**2/2/0.015)
u2 *= lens_phs
u2 = image_util.round_aperture(u2)

#Target coordinates
fov = wave/width * focal_length * 10
# oxf_1d = (np.arange(num_pts)/num_pts - 1/2) * fov
# oxf_1d = (np.arange(num_pts) - num_pts/2) * 13e-6
oxf_1d = (np.arange(74) - 74/2) * 13e-6
doxf = oxf_1d[1] - oxf_1d[0]
oxf = np.tile(oxf_1d, (len(oxf_1d), 1))
oyf = oxf.T.flatten()
oxf = oxf.flatten()

#Calculate bandwidth
if z2f < zcrit2:
    bf2 = 1/dx2
elif z2f >= 3*zcrit2:
    bf2 = np.sqrt(2*num_pts/(wave*z2f))
else:
    bf2 = 2*num_pts*dx2/(wave*z2f)

#Get gaussian quad
fx, fy, wq = polar_quad(lambda t: np.ones_like(t)*bf2/2, num_pts, num_pts)

#Get transfer function
fz2 = 1. - (wave*fx)**2 - (wave*fy)**2
evind = fz2 < 0
Hn = np.exp(1j* kk * z2f * np.sqrt(np.abs(fz2)))
Hn[evind] = 0

#scale factor
scl = 2*np.pi

#Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
angspec = finufft.nufft2d2(fx*scl*dx2, fy*scl*dx2, u2, isign=-1, eps=tol)

#Get solution with inverse NUFFT (nonuniform -> nonuniform)
uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, oxf, oyf, isign=1, eps=tol)
uu = uu.reshape((len(oxf_1d),)*2)

#Normalize
uu *= dx2**2
uu = abs(uu)**2

#Theoretical Airy Disk
xa = kk*width/2*np.hypot(oxf, oyf)/focal_length
xa[np.isclose(xa,0)] = 1e-16
area = np.pi*width**2/4
I0 = area**2/wave**2/focal_length**2
airy = I0*(2*j1(xa)/xa)**2
airy = airy.reshape((len(oxf_1d),)*2)

#Plot
# fig, axes = plt.subplots(1, 2, figsize=(8,5))
# axes[0].imshow(abs(u2)**2)
# axes[1].imshow(uu)

plt.figure()
plt.semilogy(oxf_1d/13e-6, uu[len(uu)//2])
plt.semilogy(oxf_1d/13e-6, airy[len(airy)//2], '--')


breakpoint()
