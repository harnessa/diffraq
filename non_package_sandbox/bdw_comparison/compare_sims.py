import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq

session = ['wfirst', 'bb_2017', 'M12P2', 'M12P6'][-2]

bdw_run = 'bdw_1x'
dif_run = 'diffraq_5'


load_dir_base = f'{diffraq.results_dir}/bdw_compare_new'

#BDW
with h5py.File(f'{load_dir_base}/{session}/pupil_{bdw_run}.h5', 'r') as f:
    bdw = f['pupil_Ec'][()]
    bxx = f['pupil_xx'][()]
    if 'image' in f.keys():
        bimg = f['image'][()]
        bix = f['image_xx'][()]

#FIXME: check coords
if session == 'M12P2':
    bdw = bdw[:,::-1]
    bimg = bimg[:,::-1]

#DIFFRAQ
dif_params = {
    'load_dir_base':    load_dir_base,
    'session':          session,
    'load_ext':         dif_run,
}
alz = diffraq.Analyzer(dif_params)
dfq = alz.pupil[0]
dxx = alz.pupil_xx
dimg = alz.image[0]

print(np.abs(bimg - dimg).max())

# ifig, iaxes = plt.subplots(1,2, sharex=True, sharey=True)
# iaxes[0].imshow(np.abs(bdw)**2)
# iaxes[1].imshow(np.abs(dfq)**2)
# iaxes[0].set_title('BDW')
# iaxes[1].set_title('DIFFRAQ')

# fig, axes = plt.subplots(2, figsize=(9,9))
# axes[0].semilogy(bxx, np.abs(bdw)[len(bdw)//2]**2, '-' , label='BDW')
# axes[0].semilogy(dxx, np.abs(dfq)[len(dfq)//2]**2, '--', label='DIFFRAQ')
# axes[0].legend()
# axes[1].plot(bxx, np.angle(bdw)[len(bdw)//2])
# axes[1].plot(dxx, np.angle(dfq)[len(dfq)//2], '--')

ffig, faxes = plt.subplots(1,2, sharex=True, sharey=True)
if [False,True][0]:
    faxes[0].imshow(np.log10(bimg))
    faxes[1].imshow(np.log10(dimg))
else:
    faxes[0].imshow(bimg)
    faxes[1].imshow(dimg)
faxes[0].plot(bimg.shape[-1]//2, bimg.shape[-1]//2,'r+')
faxes[1].plot(alz.image.shape[-1]//2, alz.image.shape[-1]//2,'r+')
faxes[0].set_title('BDW')
faxes[1].set_title('DIFFRAQ')
print(dimg.max(), bimg.max())
print('Diff/BDW', dimg.max()/bimg.max())

plt.figure()
# plt.imshow(np.log10(np.abs(bimg - dimg)))
# plt.imshow(np.abs(bimg - dimg))
plt.imshow((bimg - dimg))

breakpoint()
