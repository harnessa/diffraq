import diffraq
import matplotlib.pyplot as plt;plt.ion()
import numpy as np

#User-input parameters
params = {

    ### Dual Parameters ###
    'load_dir_base':   f'{diffraq.results_dir}/m12p6_test',
    'session':        'allE',

    ### Analyzer Parameters ###
    'cam_analyzer':     ['p','o'][0],
    'wave_ind':         0,
    'skip_pupil':       True,

    ### Contrast Parameters ###
    'is_contrast':      True,
    'max_apod':         0.9,
    'freespace_corr':   {641:0.89, 660:0.83, 699:0.96, 725:0.96},
}

params['load_ext'] = 'nom'
params['calibration_file'] = f'{diffraq.results_dir}/m12p6_test/{params["load_ext"]}_Mcal/image.h5'
alz1 = diffraq.Analyzer(params)

params['load_ext'] = 'eh4'
params['calibration_file'] = f'{diffraq.results_dir}/m12p6_test/{params["load_ext"]}_Mcal/image.h5'
alz2 = diffraq.Analyzer(params)

fig, axes = plt.subplots(1, 3, figsize=(11,8), sharex=True,sharey=True)

#Compare difference
dif = alz1.image - alz2.image
maxper = (1 - alz1.image.max()/alz2.image.max())*100
difper = dif.max()/alz1.image.max()*100

axes[0].imshow(alz1.image)
axes[0].set_title(alz1.load_ext)
axes[1].imshow(alz2.image)
axes[1].set_title(alz2.load_ext)
axes[2].imshow(abs(dif))
print(abs(dif).max(), maxper, difper)
print(alz1.image.max(), alz2.image.max())

breakpoint()
