import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py
from scipy.interpolate import interp1d
from utilities import util

### Load Data ###

ncycles = 4
xy0 = np.array([9.3391, -0.0992]) * 1e-3
freq = 1379.94
amp = 2e-6

loci_dir = '/home/aharness/repos/Milestone_2/diffraq_analysis/modeling/M12P9/make_shape/Loci'
nom = np.genfromtxt(f'{loci_dir}/nompetal12p.txt', delimiter='  ')
nom[:,1] *= -1

### Trim ###

rads = np.hypot(*nom.T)
ind0 = np.argmin(np.hypot(*(nom - xy0).T))
ind1 = np.argmin(np.abs(rads - (rads[ind0] + ncycles/freq)))

nom = nom[ind0:ind1]
rads = rads[ind0:ind1]

# plt.plot(*nom.T)
# plt.plot(*nom[ind0],'o')
# plt.plot(*nom[ind1],'s')
# plt.plot(*xy0,'d')

### Normals ###

def get_normal(pet):
    norm = (np.roll(pet, 1, axis=0) - pet)[:,::-1] * np.array([-1,1])
    norm /= np.hypot(*norm.T)[:,None]
    norm[0] = norm[1]
    norm[-1] = norm[-2]
    return norm

norm = get_normal(nom)

### Build sines ###

xx1 = 2*ncycles * np.linspace(0,1, len(rads))
sin1 = amp * np.sin(np.pi*xx1)[:,None]
new1 = nom + norm * sin1

xx2 = np.load('./xx2.npy')
# np.save('./xx2', 2*ncycles * diffraq.quadrature.lgwt(len(rads), 0, 1)[0][::-1])
# xx2 = xx1.copy()
# sin2 = amp * np.sin(np.pi*xx2)[:,None]
rr2 = xx2/(2*ncycles) * rads.ptp() + rads.min()
sin2 = np.interp(rr2, rads, sin1[:,0])[:,None]

# xx3 = np.interp(rr2, rads, xx1)
# sin3 = amp*np.sin(np.pi*xx3)

pa0 = np.load('./xx2.npy')          #quad nodes [cycles]
t0 = np.hypot(*xy0)                 #start radius
ts = pa0/freq/2 + t0                #radius at quad nodes [m]
##
#I need a kluge to get radius points from original
ue = nom.copy() #Load this from file?
ur = np.hypot(*ue.T)    #Or this
##
uxx = 2*ncycles*np.linspace(0, 1, len(ts))   #uniform cycle nodes
pa = np.interp(ts, ur, uxx)                  #interpolated cycle nodes
sin3 = amp*np.sin(np.pi*pa)[:,None]          #sine wave at nodes

nom3x = np.interp(ts, rads, nom[:,0])       #This will be cart_func call
nom3y = np.interp(ts, rads, nom[:,1])
nom3 = np.stack((nom3x, nom3y),1)
norm3 = get_normal(nom3)

plt.figure()
plt.plot(rads, sin1)
plt.plot(rr2, sin2,'--')
plt.plot(ts, sin3,':')

plt.figure()
plt.plot(rads, sin1, 'x')
plt.plot(ts, sin3,'+')


new1 = nom + norm*sin1
new3 = nom3 + norm3*sin3

#reinterpolate with common x-axis
newy1 = np.interp(nom[:,0], new1[:,0], new1[:,1])
newy3 = np.interp(nom[:,0], new3[:,0], new3[:,1])

plt.figure()
plt.plot(nom[:,0], newy1 - nom[:,1])
plt.plot(nom[:,0], newy3 - nom[:,1], ':')

breakpoint()
#
# nom2x = np.interp(rr2, rads, nom[:,0])
# nom2y = np.interp(rr2, rads, nom[:,1])
# nom2 = np.stack((nom2x, nom2y),1)
# norm2 = get_normal(nom2)
#
# norm3x = np.interp(rr2, rads, norm[:,0])
# norm3y = np.interp(rr2, rads, norm[:,1])
# norm3 = np.stack((norm3x, norm3y), 1)
#
# new2 = nom2 + norm2 * sin2
# new3 = nom2 + norm3 * sin2
#
# #reinterpolate with common x-axis
# newy1 = np.interp(nom[:,0], new1[:,0], new1[:,1])
# newy2 = np.interp(nom[:,0], new2[:,0], new2[:,1])
# newy3 = np.interp(nom[:,0], new3[:,0], new3[:,1])
#
# plt.figure()
# plt.plot(nom[:,0], newy1 - nom[:,1])
# plt.plot(nom[:,0], newy2 - nom[:,1], '--')
# plt.plot(nom[:,0], newy3 - nom[:,1], ':')
#
# # plt.figure()
# # plt.plot(rads, (new1 - nom)[:,0])
# # plt.plot(rads, (new1 - nom)[:,1])
# # plt.plot(rads, (new2 - nom)[:,0], '--')
# # plt.plot(rads, (new2 - nom)[:,1], '--')
#
# # plt.figure()
# # plt.plot(rads, -sin1)
# # plt.plot(rads, -sin2, '--')
#
# breakpoint()
