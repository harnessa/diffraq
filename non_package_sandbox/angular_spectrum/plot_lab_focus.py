import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py

with h5py.File('./saves/lab_focus3.h5', 'r') as f:
    dzs = f['dzs'][()]
    z10 = f['z10'][()]
    z20 = f['z20'][()]
    maxs = f['maxs'][()]

wmx = maxs.sum(-1)

waves = np.array([641, 660, 699, 725])

plt.imshow(wmx)

mind = np.unravel_index(np.argmax(wmx), wmx.shape)

z1 = z10 + dzs[mind[0]]
z2 = z20 + dzs[mind[1]]

print(z1, z2, mind, dzs[mind[0]]*1e3, dzs[mind[1]]*1e3)

# fig, axes = plt.subplots(2,2, figsize=(8,8))
# for i in range(4):
#     axes[i//2,i%2].imshow(maxs[...,i])
#     axes[i//2,i%2].set_title(waves[i])



breakpoint()
