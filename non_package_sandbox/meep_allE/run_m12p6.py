import diffraq
import numpy as np
import dial2
import imp

params = {

    ### World ###
    'num_pts':              512,
    'zz':                   49.986,
    'z0':                   27.455,
    'tel_diameter':         5e-3,
    'waves':                np.array([641, 660, 699, 725][:1])*1e-9,
    'focal_length':         0.496,
    'defocus':              8e-4,
    'image_size':           128,

    ### Numerics ###
    'radial_nodes':         5000,
    'theta_nodes':          300,
    'seam_radial_nodes':    500,
    'seam_theta_nodes':     500,

    ### Vector ###
    'seam_width':           25e-6,
    'do_run_vector':        True,
    'maxwell_file':         './all_flds',

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        f'{diffraq.results_dir}/m12p6_test',
    'session':              'allE',

    'spin_angle':           20,

}


mask = 'M12P6'

#Load mask + add etch error
mod = imp.load_source('mask', f'{diffraq.occulter_dir}/{mask}.cfg')
starshade = mod.starshade
starshade['etch_error'] = 0

params['save_ext'] = 'eh'

sim = diffraq.Simulator(params, starshade)
sim.run_sim()

# MCAL
if [False, True][1]:
    params['session'] = f'{params["save_ext"]}_Mcal'
    params['save_ext'] = ''

    #Shape
    shape = {'kind':'circle', 'is_opaque':False, 'max_radius':25.086e-3}

    #Load simulator and run sim
    sim = diffraq.Simulator(params, shape)
    sim.run_sim()
