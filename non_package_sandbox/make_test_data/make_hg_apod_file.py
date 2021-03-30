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

rr = np.linspace(rmin, rmax, num_pts)
aa = ss_Afunc(rr)

data = np.stack((rr, aa), 1)

with h5py.File(f'{diffraq.int_data_dir}/Test_Data/hg_apod_file.h5', 'w') as f:
    f.create_dataset('hg__a_b_n', data=np.array([hga, hgb, hgn]))
    f.create_dataset('data', data=data)
    f.create_dataset('header', data='radius [m], apodization value')

plt.plot(rr, aa)
breakpoint()
