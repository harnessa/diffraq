import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq

session = ['wfirst', 'bb_2017', 'M12P2', 'M12P6'][-1]

bdw_run = 'bdw'
dif_run = 'diffraq6'


load_dir_base = f'{diffraq.results_dir}/bdw_compare'

#BDW
with h5py.File(f'{load_dir_base}/{session}/pupil_{bdw_run}.h5', 'r') as f:
    bdw = f['pupil_Ec'][()]
    bxx = f['pupil_xx'][()]

#FIXME: check coords
#take transpose
bdw = bdw.T

#DIFFRAQ
dif_params = {
    'load_dir_base':    load_dir_base,
    'session':          session,
    'load_ext':         dif_run,
}
alz = diffraq.Analyzer(dif_params)
dfq = alz.pupil[0]
dxx = alz.pupil_xx

print(np.abs(bdw - dfq).max()**2)

# ifig, iaxes = plt.subplots(1,2)
# iaxes[0].imshow(np.abs(bdw)**2)
# iaxes[1].imshow(np.abs(dfq)**2)
# iaxes[0].set_title('BDW')
# iaxes[1].set_title('DIFFRAQ')
#
# fig, axes = plt.subplots(2, figsize=(9,9))
# axes[0].semilogy(bxx, np.abs(bdw)[len(bdw)//2]**2, '-' , label='BDW')
# axes[0].semilogy(dxx, np.abs(dfq)[len(dfq)//2]**2, '--', label='DIFFRAQ')
# axes[0].legend()
#
# axes[1].plot(bxx, np.angle(bdw)[len(bdw)//2])
# axes[1].plot(dxx, np.angle(dfq)[len(dfq)//2], '--')

plt.figure()
# plt.imshow(np.log10(alz.image[0]))
plt.imshow(alz.image[0])
plt.plot(alz.image.shape[-1]//2, alz.image.shape[-1]//2,'r+')
print(alz.image[0].max())

util = diffraq.utils.image_util
bdwi = util.pad_array(util.round_aperture(bdw.copy())[0], 2048)
bdwi = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(bdwi)))
bdwi = np.real(bdwi.conj()*bdwi)

plt.figure()
plt.imshow(bdwi)
breakpoint()
