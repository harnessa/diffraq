import diffraq
import numpy as np
import imp

wave = 0.641e-6
tel_diameter = 5e-3
z1 = 50.
z0 = 27.5
num_pts = 256
num_pet = 12
focal_length = 0.498

apod = 'bb_2017'

session = apod
save_dir_base = f'{diffraq.results_dir}/vector_test'

seam_width = 5e-6
mm = 6000
nn = 300

mm = 400
nn = 30

#Params
dif_params = {

    ### World ###
    'num_pts':              num_pts,
    'zz':                   z1,
    'z0':                   z0,
    'tel_diameter':         tel_diameter,
    'waves':                wave,
    'focal_length':         focal_length,
    'image_size':           128,

    ### Numerics ###
    'radial_nodes':         mm,
    'theta_nodes':          nn,
    'seam_radial_nodes':    mm//2,
    'seam_theta_nodes':     nn//2,

    ### Vector ###
    'seam_width':           seam_width,

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        save_dir_base,
    'session':              session,
}

#Load bb_2017
occ_cfg = f'{diffraq.occulter_dir}/{apod}.cfg'
mod = imp.load_source('mask', occ_cfg)
shape = mod.shapes[0]

# #Nominal
# dif_params['do_run_vector'] = False
# dif_params['save_ext'] = 'nom'
# sim = diffraq.Simulator(dif_params, shape)
# sim.run_sim()

# #Sommerfeld
# dif_params['do_run_vector'] = True
# dif_params['is_sommerfeld'] = True
# dif_params['save_ext'] = f'sommer_{seam_width*1e6:.0f}c'
# sim = diffraq.Simulator(dif_params, shape)
# sim.run_sim()

#Lab Edges
vec_edge = 'M12P2_al_stoic_ta5_sd225'
dif_params['do_run_vector'] = True
dif_params['is_sommerfeld'] = False
dif_params['maxwell_file'] =  f'{diffraq.vector_dir}/{vec_edge}'
dif_params['save_ext'] = f'real_{seam_width*1e6:.0f}b'
sim = diffraq.Simulator(dif_params, shape)
sim.run_sim()
