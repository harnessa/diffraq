import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()
import finufft
import time
import diffraq.utils.image_util as image_util
from diffraq.quadrature import polar_quad
from scipy.special import j1

#Distances between lenses
# z12 = 155e-3
# z2f = 60e-3

#Lenses
l1 = diffraq.diffraction.Lens('AC508-150-A-ML')
l2 = diffraq.diffraction.Lens('AC064-015-A-ML')
z120 = 167.84e-3
z2f0 = 59.85e-3

# l1 = diffraq.diffraction.Lens({'name':'EO_35996-150'})
# l2 = diffraq.diffraction.Lens({'name':'EO_84328-015'})
# z120 = 168.95e-3
# z2f0 = 56.8e-3

dz = 0.2e-3
z12s = np.linspace(z120-dz, z120+dz, 20)
z2f = z2f0

# z12 = z120
# z2fs = np.linspace(z2f0-dz, z2f0+dz, 20)

z2f = z2f0
z12 = z120

allu = []
# for z2f in z2fs:
# for z12 in z12s:
for i in [0]:

    #Input
    num_pts = 512
    wave = 0.641e-6
    width = 5e-3
    tol = 1e-9
    focal_length = 500e-3
    # focal_length = 450e-3

    #Derived
    kk = 2*np.pi/wave
    dx1 = width/num_pts
    dx2 = l2.diameter/num_pts
    zcrit1 = 2*num_pts*dx1**2/wave
    zcrit2 = 2*num_pts*dx2**2/wave

    #Input field
    u0 = np.ones((num_pts, num_pts)) + 0j

    #Source coordinates
    x1 = (np.arange(num_pts) - num_pts/2) * dx1

    #########################
    ### First Propagation ###
    #########################

    #Apply first lens
    rr1 = np.hypot(x1, x1[:,None])
    lens_phs = np.exp(1j*kk*l1.opd_func(rr1))
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
    lens_phs = np.exp(1j*kk*l2.opd_func(rr2))
    u2 *= lens_phs
    u2 = image_util.round_aperture(u2)

    #Target coordinates
    # fov = wave/width * focal_length * 10
    # oxf = (np.arange(num_pts)/num_pts - 1/2) * fov
    oxf = (np.arange(num_pts) - num_pts/2) * 13e-6
    oxf = np.tile(oxf, (num_pts, 1))
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
    uu = uu.reshape(u0.shape)

    #Normalize
    uu *= dx2**2
    uu = abs(uu)**2

    allu.append(uu[len(uu)//2])
allu = np.array(allu)

#Theoretical Airy Disk
xa = kk*width/2*np.hypot(oxf, oyf)/focal_length
xa[np.isclose(xa,0)] = 1e-16
area = np.pi*width**2/4
I0 = area**2/wave**2/focal_length**2
airy = I0*(2*j1(xa)/xa)**2
airy = airy.reshape(u0.shape)


fig, axes = plt.subplots(1, 2, figsize=(8,5))
axes[0].imshow(abs(u2)**2)
axes[1].imshow(uu)

plt.figure()
plt.semilogy(uu[len(uu)//2])
plt.semilogy(airy[len(airy)//2], '--')


# diff = abs(allu - airy[len(airy)//2]).sum(1)
# plt.semilogy((z12s- z120)*1e3, diff, 'o')
# # plt.semilogy((z2fs- z2f0)*1e3, diff, 'o')
#
#
# plt.figure()
# for u in allu:
#     plt.semilogy(u)
# plt.semilogy(airy[len(airy)//2], '--')
#


breakpoint()
