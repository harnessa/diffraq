import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py

#Load bb_2017
with h5py.File(f'{diffraq.apod_dir}/bb_2017.h5', 'r') as f:
    data = f['data'][()]
rads = data[:,0]
apod = data[:,1]

rmin = rads.min()
max_apod = apod.max()

#Get inner (inverse)
inn_cut = np.where(apod >= max_apod)[0][0]
data_inn = np.stack((rads, 1 - apod), 1)[:inn_cut]

#Get outer
out_cut = np.where(apod >= max_apod)[0][-1]
data_out = data[out_cut:]

#Write out full apod
with h5py.File(f'{diffraq.int_data_dir}/Test_Data/inv_apod_file.h5', 'w') as f:
    f.create_dataset('note', data='Test Inverse apodization function (bb_2017)')
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=data)

#Write out split apod
with h5py.File(f'{diffraq.int_data_dir}/Test_Data/inv_apod__inner.h5', 'w') as f:
    f.create_dataset('note', data='Inner apodization of inverse apod')
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=data_inn)

with h5py.File(f'{diffraq.int_data_dir}/Test_Data/inv_apod__outer.h5', 'w') as f:
    f.create_dataset('note', data='Outer apodization of inverse apod')
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=data_out)

#Apod with max value = 1
a1_data = data.copy()
a1_data[:,1] /= a1_data[:,1].max()
with h5py.File(f'{diffraq.int_data_dir}/Test_Data/inv_apod_A1_file.h5', 'w') as f:
    f.create_dataset('note', data='Test Inverse apodization function (bb_2017) with max A=1')
    f.create_dataset('header', data='radius [m], apodization value')
    f.create_dataset('data', data=a1_data)
