import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py
from scipy.interpolate import interp1d
from utilities import util

use_inn = [False, True][1]

if use_inn:
    ncycles = 4
    pet_num = 3
else:
    ncycles = 5
    pet_num = -3

def rot(pet, num):
    ang = 2.*np.pi/12*num
    mat = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    return pet.dot(mat)
#######
loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

if use_inn:
    pet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
else:
    pet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')

nom[:,1] *= -1
pet[:,1] *= -1
#######
dd = np.hypot(*(pet - nom).T)

inds = np.where(dd != 0)[0]
if use_inn:
    inds = inds[:-2]
    nquad = 784
else:
    inds = inds[:-7]
    nquad = 574

pet = pet[inds]
nom = nom[inds]
rads = np.hypot(*nom.T)

###########################

norm1 = (np.roll(nom, 1, axis=0) - nom)[:,::-1] * np.array([-1,1])
norm1 /= np.hypot(*norm1.T)[:,None]
norm1[0] = norm1[1]
norm1[-1] = norm1[-2]

norm2 = util.compute_derivative(nom, xaxis=rads)[:,::-1] * np.array([1,-1])
norm2 /= np.hypot(*norm2.T)[:,None]
norm2[0] = norm2[1]
norm2[-1] = norm2[-2]

plt.plot(norm1[:,0], '-')
plt.plot(norm2[:,0], '--')
plt.plot(norm1[:,1], '-')
plt.plot(norm2[:,1], '--')

breakpoint()
