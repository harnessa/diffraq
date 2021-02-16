import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

rmin, rcut, rstt, rmax = 8e-3, 13e-3, 18e-3, 25e-3
max_apod = 0.9

afunc_inn = lambda r: np.exp(-((r-rmin)/(rcut-rmin)/0.6)**6)
afunc_out = lambda r: np.exp(-((r-rmax)/(rstt-rmax)/0.6)**6)

num_pts = 50000

rads = np.linspace(rmin, rmax, num_pts)
apod = np.zeros_like(rads)

#Load bb_2017
rads, apod = np.genfromtxt(f'{diffraq.apod_dir}/bb_2017.txt', delimiter=',').T
breakpoint()

#Build apod
apod[rads <= rcut] = afunc_inn(rads[rads <= rcut])
apod[rads >= rstt] = afunc_out(rads[rads >= rstt])

#Add struts
apod *= max_apod

#Take inverse
apod = max_apod - apod

#Get inner (inverse)
ainn0 = 1 - apod[rads <= rcut].copy()
rinn0 = rads[rads <= rcut].copy()

#Get outer
aout0 = apod[rads >= rcut].copy()
rout0 = rads[rads >= rcut].copy()

#Resample
rinn = np.linspace(rinn0.min(), rinn0.max(), num_pts)
ainn = np.interp(rinn, rinn0, ainn0)
rout = np.linspace(rout0.min(), rout0.max(), num_pts)
aout = np.interp(rout, rout0, aout0)


#Write out full apod
with open(f'{diffraq.int_data_dir}/Test_Data/inv_apod_file.txt', 'w') as f:
    f.write(f'#Test Inverse apodization function with radii [in mm]: ' + \
        f'rmin={rmin*1e3:0f}, rcut={rcut*1e3:0f}, rstt={rstt*1e3:0f}, rmax={rmax*1e3:0f} ,n=6\n')
    for i in range(len(rads)):
        f.write(f'{rads[i]},{apod[i]}\n')

#Write out split apod
with open(f'{diffraq.int_data_dir}/Test_Data/inv_apod__inner.txt', 'w') as f:
    f.write(f'#Inner apodization of inverse apod')
    for i in range(len(rinn)):
        f.write(f'{rinn[i]},{ainn[i]}\n')

with open(f'{diffraq.int_data_dir}/Test_Data/inv_apod__outer.txt', 'w') as f:
    f.write(f'#Outer apodization of inverse apod')
    for i in range(len(rout)):
        f.write(f'{rout[i]},{aout[i]}\n')


plt.plot(rads, apod)

plt.plot(rinn, ainn, '--')
plt.plot(rout, aout, '--')
breakpoint()
