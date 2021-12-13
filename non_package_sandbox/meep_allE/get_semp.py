import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import semp
import os

do_save = [False, True][0]

load_ext = ''
session = 'test_all2'

save_file = './all_flds'

params = {
    # 'session':      f'{session}/{load_ext}',
    'session':      'final_model/m12px_newfinal',
    'base_dir':     '/scratch/network/aharness/Semp_Results',
    'obs_distance': 0.0,
}

#Load analyzer
alz = semp.analysis.Analyzer(params)
waves = alz.prop.msim.waves

#Get x index
xind = alz.get_xind()

dirs = ['y','z']
flds = ['e','h']

#Loop through waves and get data
all_data = []
for iw in range(len(waves)):

    #Get wave index
    wind = alz.get_wind(waves[iw])

    data = {}
    for ff in flds:
        for dd in dirs:
            comp = ff + dd
            data[comp] = alz.get_data(comp, wave=waves[0], ind=xind, is_bbek=True)

    all_data.append(data)

fig, axes = plt.subplots(3,2)
for iw in range(len(waves)):
    for i in range(len(dirs)):
        for j in range(len(flds)):
            cn = flds[j] + dirs[i]
            axes[i,j].plot(abs(all_data[iw][cn]))
            axes[i,j].set_title(cn)

for fld in flds:
    with h5py.File(f'{save_file}_{fld.upper()}.h5', 'w') as f:
        f.create_dataset('waves', data=np.array(waves)*1e-6)
        for i in range(len(waves)):
            #Write out edges
            f.create_dataset(f'{waves[i]*1e3:.0f}_x', data=alz.yy*1e-6)
            for cn in dirs:
                f.create_dataset(f'{waves[i]*1e3:.0f}_{fld}{cn}', data=all_data[iw][fld+cn])


breakpoint()
