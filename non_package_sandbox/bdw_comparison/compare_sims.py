import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq

bdw_run = 'bdw'
dif_run = 'diffraq'

session  = 'wfirst'
load_dir_base = f'{diffraq.results_dir}/bdw_compare'

#BDW
with h5py.File(f'{load_dir_base}/{session}/pupil_{bdw_run}.h5', 'r') as f:
    bdw = f['pupil_Ec'][()]
    bxx = f['pupil_xx'][()]

#DIFFRAQ
dif_params = {
    'load_dir_base':    load_dir_base,
    'session':          session,
    'load_ext':         dif_run,
}
dif_alz = diffraq.Analyzer(dif_params)
dfq = dif_alz.pupil[0]
dxx = dif_alz.pupil_xx

# dfq *= np.exp(-1j*2*np.pi/dif_alz.sim.waves[0] * dif_alz.sim.zz/2)
print(np.abs(bdw - dfq).max()**2)

#Phase / amp
bphs = np.angle(bdw)
bdw = np.abs(bdw)**2
dphs = np.angle(dfq)
dfq = np.abs(dfq)**2


# fig, axes = plt.subplots(2)
# axes[0].imshow(bdw)
# axes[1].imshow(dfq)

fig, axes = plt.subplots(2, figsize=(9,9))
axes[0].semilogy(bxx, bdw[len(bdw)//2], '-' , label='BDW')
axes[0].semilogy(dxx, dfq[len(dfq)//2], '--', label='DIFFRAQ')
axes[0].legend()

axes[1].plot(bxx, bphs[len(bphs)//2])
axes[1].plot(dxx, dphs[len(dphs)//2], '--')
breakpoint()
