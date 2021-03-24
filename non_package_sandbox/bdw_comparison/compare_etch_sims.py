import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import diffraq

session = 'etch'

bdw_ext = 'n1'
dif_ext = 'n1'

bdw_run = f'bdw__etch_{bdw_ext}'
dif_run = f'diffraq__etch_{dif_ext}'


load_dir_base = f'{diffraq.results_dir}/bdw_compare_new'

#BDW
with h5py.File(f'{load_dir_base}/{session}/pupil_{bdw_run}.h5', 'r') as f:
    if 'image' in f.keys():
        bimg = f['image'][()]

#DIFFRAQ
dif_params = {
    'load_dir_base':    load_dir_base,
    'session':          session,
    'load_ext':         dif_run,
}
alz = diffraq.Analyzer(dif_params)
dimg = alz.image[0]

print(np.abs(bimg - dimg).max())

ffig, faxes = plt.subplots(1,2, sharex=True, sharey=True)
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
