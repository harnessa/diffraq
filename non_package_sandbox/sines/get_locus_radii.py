
import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py
from scipy.interpolate import interp1d
from utilities import util
from scipy.optimize import newton, root_scalar
import time

use_inn = [False, True][1]

if use_inn:
    ncycles = 4
else:
    ncycles = 5

###########################

loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

if use_inn:
    pet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
else:
    pet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')

nom[:,1] *= -1
pet[:,1] *= -1

dd = np.hypot(*(pet - nom).T)
nom = nom[dd != 0]

###########################
scale = 12/16

#Scale to 16 petals
def scale_petal(pet, scl):
    rr = np.hypot(*pet.T)
    aa = np.arctan2(*pet[:,::-1].T) * scl
    new = rr[:,None] * np.stack((np.cos(aa), np.sin(aa)), 1)
    return new

nom12 = nom
nom16 = scale_petal(nom.copy(), scale)


x12 = nom12[:,0]
x16 = nom16[:,0]
r12 = np.hypot(*nom12.T)
r16 = np.hypot(*nom16.T)

print(np.diff(x12).std(), np.diff(x16).std(), np.diff(r12).std(), np.diff(r16).std())

###########################

#this will be rpleaced by func
rad_tru = r12.copy()
the_tru = np.arctan2(nom12[:,1], nom12[:,0])
#
t0 = r12.min()
t1 = r12.max()
ts = np.linspace(t0, t1, len(r12))

#uniformly spaced x's in 16 petal space
x0 = t0*np.cos(np.interp(t0, rad_tru, the_tru)*scale)
x1 = t1*np.cos(np.interp(t1, rad_tru, the_tru)*scale)
xs = np.linspace(x0, x1, len(r12))
#turn xs into rs
func = lambda r, x: r - x/np.cos(np.interp(r, rad_tru, the_tru)*scale)
tik = time.perf_counter()
ans = np.array([newton(func, ts[i], args=(xs[i],)) for i in range(len(xs))])
tok = time.perf_counter()
print(f'time1: {tok-tik:.23}')
tik = time.perf_counter()
ans2 = np.array([root_scalar(func, x0=ts[i], x1=ts[i]+1e-5, args=(xs[i],)).root for i in range(len(xs))])
tok = time.perf_counter()
print(f'time2: {tok-tik:.3f}')

#Or just use ts for close enough
# ans2 = xs/np.cos(np.interp(ts, rad_tru, the_tru)*scale)

plt.figure()
plt.plot(r12, 'x')
plt.plot(ans, '+')
plt.figure()
plt.plot(r12 - ans)
plt.plot(r12 - ans2)
# plt.plot(*nom12.T, 'x-')
# plt.plot(*nom16.T, '+-')

breakpoint()
