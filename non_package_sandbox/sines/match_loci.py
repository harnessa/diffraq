import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from utilities import util

loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

use_inn = [False, True][0]

if use_inn:
    pet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
    ncycles = 4
    pet_num = 3
else:
    pet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')
    ncycles = 5
    pet_num = -3

nom[:,1] *= -1
pet[:,1] *= -1

old_nom = nom.copy()
old_pet = pet.copy()

amp = 2e-6
rads = np.hypot(*nom.T)

use = nom.copy()

#Build sine wave
dd = np.hypot(*(pet - nom).T)
the = np.zeros(len(dd))
# inds = np.where(dd != 0)[0]
inds = np.where(abs(dd)>1e-8)[0]

if use_inn:
    inds = inds[:-2]
else:
    # inds = inds[:-7]
    inds = inds[:-8]
npts = len(inds)

use = use[inds]
pet = pet[inds]
nom = nom[inds]
rads = np.hypot(*nom.T)

num_quad = npts//ncycles//2
import diffraq.quadrature as quad
# pa, wa = quad.lgwt(num_quad, 0, 1)
pa = np.linspace(0, 1, num_quad, endpoint=False)[::-1]
pa = np.concatenate((pa[::-1], 1+pa[::-1]))
pa = (pa + 2*np.arange(ncycles)[:,None]).ravel()
the = np.pi*pa

#Scale factor 16 -> 12 petals
scale = 12/16

#Build new old edge w/ 16 petals
a0 = np.arctan2(use[:,1], use[:,0]) * scale
pet0 = np.hypot(use[:,0], use[:,1])[:,None] * \
    np.stack((np.cos(a0), np.sin(a0)),1)

#Build normal from this edge
normal = (np.roll(pet0, 1, axis=0) - pet0)[:,::-1] * np.array([-1,1])
normal /= np.hypot(normal[:,0], normal[:,1])[:,None]
#Fix normals on end pieces
normal[0] = normal[1]
normal[-1] = normal[-2]

sine = amp*np.sin(the[:,None])

new0 = pet0 + normal*sine

#Convert to polar cooords
anch = np.arctan2(new0[:,1], new0[:,0]) / scale

#Scale and go back to cart coords
nch = np.hypot(new0[:,0], new0[:,1])[:,None] * \
    np.stack((np.cos(anch), np.sin(anch)), 1)

#New edge
new = scale*nch + use*(1-scale)

#Get frequency [1/m]
freq = ncycles/rads.ptp()

#Get start point
xy0 = pet[0].copy()

#Rotate
ang = 2.*np.pi/12 * pet_num
rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
xy0 = xy0.dot(rot)

print(f"\n{['Out', 'Inn'][use_inn]} - Ncycles: {ncycles}, XY0: {xy0[0]*1e3:.4f}, " + \
    f"{xy0[1]*1e3:.4f} [mm]; Amp: {amp*1e6:.2f} [um]; Freq: {freq:.2f} [1/m]\n")

rad2 = np.hypot(*use.T)

#Compare
# diff = np.hypot(*(pet - new).T)

plt.figure()
plt.plot(rads, (pet - nom)[:,0], 'b')
plt.plot(rads, (pet - nom)[:,1], 'c')
plt.plot(rad2, (new - nom)[:,0], 'r--')
plt.plot(rad2, (new - nom)[:,1], 'k--')

plt.figure()
plt.plot(*old_nom.T)
plt.plot(*(old_pet.dot(rot)).T, '--')
plt.plot(*xy0, 'o')

breakpoint()
