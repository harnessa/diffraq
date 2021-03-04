import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

#Load bb_2017
rads, apod = np.genfromtxt(f'{diffraq.apod_dir}/bb_2017.txt', delimiter=',').T
rmin = rads.min()

max_apod = apod.max()

#Get inner (inverse)
inn_cut = np.where(apod >= max_apod)[0][0]
ainn = 1 - apod[:inn_cut]
rinn = rads[:inn_cut]

#Get outer
out_cut = np.where(apod >= max_apod)[0][-1]
aout = apod[out_cut:]
rout = rads[out_cut:]

#Write out full apod
with open(f'{diffraq.int_data_dir}/Test_Data/inv_apod_file.txt', 'w') as f:
    f.write(f'#Test Inverse apodization function (bb_2017)\n')
    for i in range(len(rads)):
        f.write(f'{rads[i]},{apod[i]}\n')

#Write out split apod
with open(f'{diffraq.int_data_dir}/Test_Data/inv_apod__inner.txt', 'w') as f:
    f.write(f'#Inner apodization of inverse apod\n')
    #Write out last radius
    f.write(f'#,{rinn[-1]}\n')
    for i in range(len(rinn)):
        f.write(f'{rinn[i]},{ainn[i]}\n')

with open(f'{diffraq.int_data_dir}/Test_Data/inv_apod__outer.txt', 'w') as f:
    f.write(f'#Outer apodization of inverse apod\n')
    #Write out first radius
    f.write(f'#,{rout[0]}\n')
    for i in range(len(rout)):
        f.write(f'{rout[i]},{aout[i]}\n')

# plt.plot(rads, apod, '-')
#
# plt.plot(rinn, ainn, '--')
# plt.plot(rinn, 1-ainn, '+-')
# plt.plot(rout, aout, 'x-')
# breakpoint()
