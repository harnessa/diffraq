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
save_dir_base = f'{diffraq.results_dir}/vector'

etch = 1e-6
mm = 6000
nn = 300

# mm = 2000
# nn = 100

#Params
dif_params = {

    ### World ###
    'num_pts':          num_pts,
    'zz':               z1,
    'z0':               z0,
    'tel_diameter':     tel_diameter,
    'waves':            wave,
    'focal_length':     focal_length,
    'image_size':       128,

    ### Numerics ###
    'radial_nodes':     mm,
    'theta_nodes':      nn,

    ### Vector ###
    'seam_width':       2*etch,

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    save_dir_base,
    'session':          session,
}

#Load bb_2017
occ_cfg = f'{diffraq.occulter_dir}/{apod}.cfg'
mod = imp.load_source('mask', occ_cfg)
shapes = mod.shapes[0]

#Add etch
shapes['etch_error'] = etch

#Nominal
dif_params['do_run_vector'] = False
dif_params['save_ext'] = 'etch'
sim = diffraq.Simulator(dif_params, shapes)
sim.run_sim()

#Vector
shapes['etch_error'] = None
maxwell_func = [lambda d, w: np.heaviside(-d, 1)*np.heaviside(etch+d,1)+0j for i in range(2)]
dif_params['do_run_vector'] = True
dif_params['maxwell_func'] = maxwell_func
dif_params['save_ext'] = 'vect'
sim = diffraq.Simulator(dif_params, shapes)
sim.run_sim()
