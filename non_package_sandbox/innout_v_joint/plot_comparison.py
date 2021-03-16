import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import diffraq

session = ['bb_2017', 'M12P2'][0]

dif_params = {
    'load_dir_base':    f'{diffraq.results_dir}/innout_v_joint',
    'session':          session,
    'skip_pupil':       True,
}

dif_params['load_ext'] = 'diffraq'
salz = diffraq.Analyzer(dif_params)
simg = salz.image[0]

# dif_params['load_ext'] = 'diffraq_joint'
# dif_params['load_ext'] = 'diffraq_circle2'

dif_params['load_dir_base'] =    f'{diffraq.results_dir}/bdw_compare'
dif_params['session'] = 'M12P6_circle'
dif_params['load_ext'] = 'diffraq_0shft'

jalz = diffraq.Analyzer(dif_params)
jimg = jalz.image[0]

is_log = [False, True][0]

vmax = None

fig, axes = plt.subplots(1,2, sharex=True, sharey=True)
if is_log:
    axes[0].imshow(np.log10(simg))
    axes[1].imshow(np.log10(jimg))
else:
    axes[0].imshow(simg, vmax=vmax)
    axes[1].imshow(jimg, vmax=vmax)
axes[0].set_title('Inn/Out')
axes[1].set_title('Joint')


fig, axes = plt.subplots(1,2)
axes[0].semilogy(salz.image_xx, simg[len(simg)//2], '-',  label='Inn/Out')
axes[0].semilogy(jalz.image_xx, jimg[len(jimg)//2], '--', label='Joint')
axes[1].semilogy(salz.image_xx, simg[:,len(simg)//2], '-')
axes[1].semilogy(jalz.image_xx, jimg[:,len(jimg)//2], '--')
axes[0].legend()


breakpoint()
