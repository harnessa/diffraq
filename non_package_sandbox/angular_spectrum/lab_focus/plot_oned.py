import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py

pre = '2'
suf = 'dz2'

with h5py.File(f'./saves/{pre}lab_focus_{suf}.h5', 'r') as f:
    dzs = f['dzs'][()]
    z10 = f['z10'][()]
    z20 = f['z20'][()]
    maxs = f['maxs'][()]
    imgs = f['imgs'][()]

mind = np.argmax(maxs)

z1 = z10
z2 = z20 + dzs[mind]

print(z1*1e3, z2*1e3, mind, dzs[mind]*1e3)

plt.plot(dzs, maxs, 'o-')

plt.figure()
plt.imshow(imgs[mind])

breakpoint()
