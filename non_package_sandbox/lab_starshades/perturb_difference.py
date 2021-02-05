import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py

loci_dir = '/home/aharness/repos/lotus/External_Data/Special_Loci/old'

name = 'M12P2'

if [False, True][0]:
    with open(f'{loci_dir}/{name}.txt', 'r') as f:
        lines = f.readlines()

    data, tmp, = [], []
    for ln in lines:
        if ln.startswith('@@'):
            continue
        elif ln.startswith('$$'):
            continue
        elif ln.startswith('$X'):
            continue
        elif ln.startswith('#'):
            data.append(np.array(tmp))
            tmp = []
            continue

        #Get data
        tmp.append(np.array(ln.split(',')).astype(float))

    #Get last shape
    if len(tmp) > 0:
        data.append(np.array(tmp))

    data = np.array(data)

    with h5py.File(f'./saves/{name}.h5', 'w') as f:
        f.create_dataset('mask', data=data)

    del tmp, lines, data

with h5py.File(f'./saves/{name}.h5', 'r') as f:
    nomocc = f['mask'][0]
    defocc = f['mask'][2]

ang = 2.*np.pi/12. * 3
newx =  defocc[:,0]*np.cos(ang) + defocc[:,1]*np.sin(ang)
newy = -defocc[:,0]*np.sin(ang) + defocc[:,1]*np.cos(ang)
defocc = np.dstack((newx, newy)).squeeze()
del newx, newy

# plt.plot(*nomocc.T)
# plt.plot(*defocc.T)

diff = defocc - nomocc

pert = defocc[np.hypot(*diff.T) > 0]
notpert = nomocc[np.hypot(*diff.T) > 0]

plt.plot(*pert.T, '-')
plt.plot(*notpert.T, '--')


breakpoint()
