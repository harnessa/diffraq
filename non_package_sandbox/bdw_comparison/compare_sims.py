import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq

if [False,True][0]:
    bdw_run = 'bdw_2k'
    dif_run = 'diffraq_pw'
    session  = 'wfirst'
else:
    bdw_run = 'bdw_4k'
    # dif_run = 'diffraq_joint_pw'
    dif_run = 'diffraq3'
    session  = 'bb_2017'

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

print(np.abs(bdw - dfq).max()**2)
print(np.angle(bdw)[128,128] - np.angle(dfq)[128,128])

# fig, axes = plt.subplots(2)
# axes[0].imshow(bdw)
# axes[1].imshow(dfq)

fig, axes = plt.subplots(2, figsize=(9,9))
axes[0].semilogy(bxx, np.abs(bdw)[len(bdw)//2]**2, '-' , label='BDW')
axes[0].semilogy(dxx, np.abs(dfq)[len(dfq)//2]**2, '--', label='DIFFRAQ')
axes[0].legend()

axes[1].plot(bxx, np.angle(bdw)[len(bdw)//2])
axes[1].plot(dxx, np.angle(dfq)[len(dfq)//2], '--')
breakpoint()
