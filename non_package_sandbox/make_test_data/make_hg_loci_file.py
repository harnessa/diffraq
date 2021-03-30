import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py

rmin, rmax = 5, 13
hga = rmin
hgb = (rmax - rmin)*0.6
hgn = 6
ss_Afunc = lambda r: np.exp(-((r-hga)/hgb)**hgn)

num_pts = 10000
npet = 16

rr = np.linspace(rmin, rmax, num_pts)
aa = ss_Afunc(rr)

#Trailing edge
xx = rr*np.cos(aa*np.pi/npet)
yy = rr*np.sin(aa*np.pi/npet)

#Leading edge
xx = np.concatenate((xx,  xx[::-1]))
yy = np.concatenate((yy, -yy[::-1]))

pet_ang = 2.*np.pi/npet

loci = []
for i in range(npet):
    xnew =  xx*np.cos(i*pet_ang) + yy*np.sin(i*pet_ang)
    ynew = -xx*np.sin(i*pet_ang) + yy*np.cos(i*pet_ang)
    loci.extend(np.stack((xnew, ynew),1))
loci = np.array(loci)

#Flip to run CCW
loci = loci[::-1]

plt.plot(*loci.T)

with h5py.File(f'{diffraq.int_data_dir}/Test_Data/hg_loci_file.h5', 'w') as f:
    f.create_dataset('hg__a_b_n', data = np.array([hga, hgb, hgn]))
    f.create_dataset('note', data='set key: hg__a_b_n')
    f.create_dataset('header', data='x [m], y [m]')
    f.create_dataset('loci', data=loci)

breakpoint()
