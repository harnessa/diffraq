import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq
from scipy.interpolate import InterpolatedUnivariateSpline

fname = f'{diffraq.apod_dir}/bb_2017.h5'

num_petals = 12

#Load data
with h5py.File(fname, 'r') as f:
    data = f['data'][()]

#Interpolate
fcn_func = InterpolatedUnivariateSpline(data[:,0], data[:,1], k=2, ext=3)
fcn_diff = fcn_func.derivative(1)
fcn_diff_2nd = fcn_func.derivative(2)

#Get one petal
rr = np.linspace(data[:,0].min(), data[:,0].max(), 2000)

# fig, axes = plt.subplots(1, 2, figsize=(10,10))

for ip in range(num_petals):
    for tl in [1,-1]:

        func = fcn_func(rr)*tl + 2*ip
        diff = fcn_diff(rr)*tl
        diff_2nd = fcn_diff_2nd(rr)*tl

        #Cartesian
        pang = np.pi/num_petals
        cf = np.cos(func*pang)
        sf = np.sin(func*pang)
        cart_func = rr[:,None]*np.stack((cf, sf), func.ndim).squeeze()
        cart_diff = np.stack((cf - rr*sf*diff*pang, sf + rr*cf*diff*pang), diff.ndim).squeeze()
        shr1 = -pang * diff**2 * rr
        shr2 = 2*diff + rr*diff_2nd
        cart_diff_2nd = pang*np.stack((shr1*cf - sf*shr2, shr1*sf + cf*shr2), func.ndim).squeeze()

        #Normal angle
        # beta = -np.sign(cart_diff[...,0]*cart_diff_2nd[...,1] - cart_diff[...,1]*cart_diff_2nd[...,0])
        # beta += (beta == 0)
        # nq = np.arctan2(beta*cart_diff[...,0], -beta*cart_diff[...,1])

        beta = tl
        nq = np.arctan2(beta*cart_diff[...,0], -beta*cart_diff[...,1])

        nqd = np.degrees(nq) % 360

        # axes[0].cla()
        # axes[1].cla()
        #
        # # axes[0].plot(diff_2nd,'x')
        #
        # if ip == 0 and tl == 1:
        #     cbar = plt.colorbar(axes[0].scatter(cart_func[:,0], cart_func[:,1], c=nqd, s=1))
        # else:
        #     axes[0].scatter(cart_func[:,0], cart_func[:,1], c=nqd, s=1)
        #
        # etch = 10e-6
        # axes[1].plot(cart_func[:,0], cart_func[:,1])
        # axes[1].plot(cart_func[:,0]+np.cos(nq)*etch, cart_func[:,1]+np.sin(nq)*etch, '--')

        plt.figure()
        plt.colorbar(plt.scatter(cart_func[:,0], cart_func[:,1], c=nqd, s=1))
        breakpoint()
