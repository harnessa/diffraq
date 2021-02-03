import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

apod_name = 'wfirst_ni2'

orig_apod = f'/home/aharness/repos/lotus/External_Data/Apodization_Profiles/{apod_name}.txt'

data = np.genfromtxt(orig_apod, delimiter=',')
data[:,0] *= 1e-6
# data[:,1] *= 0.9

with open(f'{diffraq.ext_data_dir}/Apodization_Profiles/{apod_name}.txt', 'w') as f:
    for i in range(len(data)):
        f.write(f'{data[i][0]},{data[i][1]}\n')

plt.plot(*data.T)
breakpoint()
