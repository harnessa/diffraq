import diffraq
import numpy as np
import dial2

mask = 'M12P2'

edge_file = 'M12P2_al_stoic_ta5_sd225'

params = {

    ### World ###
    'num_pts':              512,
    'zz':                   49.986,
    'z0':                   27.455,
    'tel_diameter':         5e-3,
    'waves':                np.array([641, 660, 699, 725])*1e-9,
    'focal_length':         0.499,
    'defocus':              3e-3,
    'image_size':           128,

    ### Numerics ###
    'radial_nodes':         3000,
    'theta_nodes':          300,
    'seam_radial_nodes':    500,
    'seam_theta_nodes':     500,

    ### Occulter ###
    'occulter_config':      f'../plain_configs/{mask}.cfg',

    ### Vector ###
    'seam_width':           25e-6,
    'do_run_vector':        True,
    'with_vector_gaps':     False,
    'maxwell_file':         f'{diffraq.vector_dir}/{edge_file}',

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        f'{diffraq.results_dir}/pert_seams',
    'session':              mask,
    'save_ext':             'w_seam',
}

#Load simulator and run sim
sim = diffraq.Simulator(params)
sim.run_sim()
