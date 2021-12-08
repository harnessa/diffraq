import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq

x0 = 8e-3
x1 = 13e-3
num_pet = 12
gap = 22e-6 *  0

npts = 10000

xx = np.linspace(x0, x1, npts)
y0 = x0 * np.tan(np.pi/num_pet) - gap/2
y1 = gap/2
mm = (y0 - y1) / (x0 - x1)

yy = mm * (xx - x0) + y0

rr = np.hypot(xx, yy)
aa = num_pet/np.pi * np.arctan2(yy,xx)

data = np.stack((rr, aa), 1)


plt.plot(*data.T)

#Write out full apod
with h5py.File(f'{diffraq.int_data_dir}/Test_Data/star_apod_file.h5', 'w') as f:
    f.create_dataset('note', data='Test Starshape apodization function')
    f.create_dataset('x0_x1_np_gap', data=np.array([x0,x1,num_pet,gap]))
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=data)

breakpoint()
