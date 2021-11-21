import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import finufft
import time
import diffraq.utils.image_util as image_util
from diffraq.quadrature import polar_quad
import diffraq

#Input
num_pts = 1024
wave = 0.5e-6
width = 12.5e-3
zz = 10
tol=1e-9

nutype = ['12', '33'][0]

#Derived
dx = width/num_pts
zcrit = 2*num_pts*dx**2/wave
kk = 2*np.pi/wave

# zz = zcrit

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

tik = time.perf_counter()

if nutype == '12':

    #Calculate angspectrum of input with NUFFT (uniform -> nonuniform)
    angspec = finufft.nufft2d2(fx*scl*dx, fy*scl*dx, u0, isign=-1, eps=tol)

    #Get solution with inverse NUFFT (nonuniform -> uniform)
    uu = finufft.nufft2d1(fx*scl*dx,  fy*scl*dx, angspec*Hn*wq, \
        n_modes=(num_pts, num_pts), isign=1, eps=tol)

else:

    #Calculate angspectrum of input with NUFFT (nonuniform -> nonuniform)
    angspec = finufft.nufft2d3(xx, yy, u0.flatten(), fx*scl, fy*scl, isign=-1, eps=tol)

    #Get solution with inverse NUFFT (nonuniform -> nonuniform)
    uu = finufft.nufft2d3(fx*scl, fy*scl, angspec*Hn*wq, xx, yy, isign=1, eps=tol)
    uu = uu.reshape(u0.shape)

#Normalize
uu *= dx**2

tok = time.perf_counter()
print(f'NUFFT {nutype} Time: {tok-tik:.2f} s')

#Calculate analytic solution
xpts = (np.arange(num_pts) - num_pts/2) * dx
utru = diffraq.utils.solution_util.calculate_circle_solution(xpts, \
    wave, zz, 1e19, width/2, False)

# plt.imshow(abs(uu))
# plt.figure()
plt.plot(xpts, abs(uu)[len(uu)//2])
plt.plot(xpts, abs(utru), '--')

breakpoint()
