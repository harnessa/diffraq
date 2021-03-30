import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py

pre = 'M12P2'
ext = 'al_stoic_ta5_sd225'

waves = [641, 660, 699, 725]

load_dir = '/home/aharness/repos/lotus/External_Data/Vector_Edges'
save_dir = '/home/aharness/repos/diffraq/External_Data/Vector_Edges'

#Get data
sflds, pflds, sxxs, pxxs = [], [], [], []
for wv in waves:
    with h5py.File(f'{load_dir}/edge_{pre}_{wv}_{ext}_p.h5', 'r') as f:
        #Trim last 10 microns
        ind = np.where(f['xx'][()] < f['xx'][-1] - 10)[0].max()
        pxxs.append(f['xx'][:ind])
        pflds.append(f['fld_drv'][:ind])
    with h5py.File(f'{load_dir}/edge_{pre}_{wv}_{ext}_s.h5', 'r') as f:
        ind = np.where(f['xx'][()] < f['xx'][-1] - 10)[0].max()
        sflds.append(f['fld_drv'][:ind])
        sxxs.append(f['xx'][:ind])

sflds, pflds, sxxs, pxxs = np.array(sflds), np.array(pflds), np.array(sxxs), np.array(pxxs)

for i in range(len(waves)):
    plt.plot(sxxs[0], abs(sflds[i]))
    plt.plot(sxxs[0], abs(pflds[i]), '--')

#Save in new data
if [False, True][0]:

    with h5py.File(f'{save_dir}/edge_{pre}_{ext}.h5', 'w') as f:
        f.create_dataset('waves', data=np.array(waves)*1e-9)
        f.create_dataset('xx', data=sxxs[0]*1e-6)
        for i in range(len(waves)):
            f.create_dataset(f'{waves[i]:.0f}_s', data=sflds[i])
            f.create_dataset(f'{waves[i]:.0f}_p', data=pflds[i])


breakpoint()
