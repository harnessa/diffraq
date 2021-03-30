import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq
import h5py

num_pts = 10000

is_circle =  False

if is_circle:
    name = 'circle'
    xfunc = lambda t: 12*np.cos(t)
    yfunc = lambda t: 12*np.sin(t)
    head = '#Test circle of radius = 12'
else:
    name = 'kite'
    xfunc = lambda t: 0.5*np.cos(t) + 0.5*np.cos(2*t)
    yfunc = lambda t: np.sin(t)
    head = '#Test of kite function x(t)=0.5cos(t) + 0.5cos(2t), y(t) = sin(t)'

the = np.linspace(0, 2*np.pi, num_pts)

xx = xfunc(the)
yy = yfunc(the)

loci = np.stack((xx,yy), 1)

plt.plot(xx, yy, 'x-')

with h5py.File(f'{diffraq.int_data_dir}/Test_Data/{name}_loci_file.h5', 'w') as f:
    f.create_dataset('note', data=head)
    f.create_dataset('header', data='x [m], y [m]')
    f.create_dataset('loci', data=loci)

breakpoint()
