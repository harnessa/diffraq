import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from utilities import util

loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'

nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')

use_inn = [False, True][0]

if use_inn:
    pet = np.genfromtxt(f'{loci_dir}/petal4_sine12p.txt', delimiter='  ')
    ncycles = 4
else:
    pet = np.genfromtxt(f'{loci_dir}/petal10_sine12p.txt', delimiter='  ')
    ncycles = 5


#Scale to 16 petals
def scale(pet, scl):
    rr = np.hypot(*pet.T)
    aa = np.arctan2(*pet[:,::-1].T) * scl
    new = rr[:,None] * np.stack((np.cos(aa), np.sin(aa)), 1)
    return new

r0 = np.hypot(*nom.T)
x0 = nom[:,0].copy()
nom = scale(nom, 12/16)
pet = scale(pet, 12/16)


# plt.plot(nom[:,0], pet[:,0] - nom[:,0])
# plt.plot(nom[:,0], pet[:,1] - nom[:,1])
# plt.plot(nom[:,0], np.hypot(*(pet - nom).T), '--')

# plt.figure()
# plt.plot(*nom.T)
# plt.plot(*pet.T)

rads = np.hypot(*nom.T)

breakpoint()
norms = np.diff(nom, axis=0)
norms = np.vstack((norms[0], norms))
# norms = util.compute_derivative(nom)
norms = norms[:,::-1] * np.array([-1.,1.])
norms /= util.norm(norms)[:,None]

#rescale
dout = 16/12*(pet - nom)
# dout = (pet - nom)

dd = np.hypot(*dout.T)

amp = dd.max()
npts = np.count_nonzero(dd)

the = np.zeros(len(dd))

freq = ncycles/nom[:,0][dd!=0].ptp()

# the[dd != 0] = 2.*np.pi*ncycles/npts * np.arange(npts)
the[dd != 0] = 2.*np.pi*ncycles*(nom[:,0][dd!=0]-nom[:,0][dd!=0].min())/nom[:,0][dd!=0].ptp()
the[dd != 0] = 2.*np.pi*freq*(nom[:,0][dd!=0]-nom[:,0][dd!=0].min())
dnew = amp*np.sin(the[:,None])*norms


plt.figure()
plt.plot(nom[:,0], dout[:,0])
plt.plot(nom[:,0], dout[:,1])
plt.plot(nom[:,0], dnew[:,0], '--')
plt.plot(nom[:,0], dnew[:,1], '--')
plt.xlim(nom[:,0][dd!=0].mean() + [-3e-3,3e-3])

plt.figure()
plt.plot(nom[:,0], dd)
plt.plot(nom[:,0], np.hypot(*dnew.T), '--')
plt.xlim(nom[:,0][dd!=0].mean() + [-3e-3,3e-3])

breakpoint()
