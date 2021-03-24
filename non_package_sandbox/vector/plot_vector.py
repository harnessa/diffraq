import numpy as np
import diffraq
import matplotlib.pyplot as plt;plt.ion()

load_dir_base = f'{diffraq.results_dir}/vector'
session = 'bb_2017'

def get_data(ext):
    dif_params = {
        'load_dir_base':    load_dir_base,
        'session':          session,
        'load_ext':         ext,
        'cam_analyzer':     [None,'p','o'][2],
    }
    alz = diffraq.Analyzer(dif_params)
    img = alz.image.copy()
    alz.clean_up()
    return img

nom = get_data('nom')
som = get_data('sommer_5')

fig, axes = plt.subplots(1,2, figsize=(9,9))
axes[0].imshow(nom)
axes[0].set_title('Nominal')
axes[1].imshow(som)
axes[1].set_title('Sommerfeld')


# fig, axes = plt.subplots(2, figsize=(9,9))
# axes[0].plot(nom[len(nom)//2])
# axes[0].plot(som[len(som)//2], '--')
# axes[1].plot(nom[:,len(nom)//2])
# axes[1].plot(som[:,len(som)//2], '--')

breakpoint()
