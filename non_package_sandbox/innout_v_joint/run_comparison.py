import diffraq
import numpy as np

wave = 0.641e-6
tel_diameter = 5e-3
z1 = 50.
z0 = 27.455
focal_length = 0.498
num_pts = 256
num_foc = 128
num_pet = 12

apod = ['bb_2017', 'M12P2'][0]

save_dir_base = f'{diffraq.results_dir}/innout_v_joint'

mm = 6000
nn = 300

# for jnt in ['', '_joint'][:]:
for jnt in ['', '_circle'][1:]:
    params = {

        ### World ###
        'num_pts':          num_pts,
        'zz':               z1,
        'z0':               z0,
        'tel_diameter':     tel_diameter,
        'waves':            wave,
        'focal_length':     focal_length,
        'image_size':       num_foc,

        ### Numerics ###
        'radial_nodes':     mm,
        'theta_nodes':      nn,

        ### Occulter ###
        'occulter_config':  f'{diffraq.occulter_dir}/{apod}{jnt}.cfg',
        'is_finite':        True,

        ### Saving ###
        'do_save':          True,
        'save_dir_base':    save_dir_base,
        'session':          apod,
        'save_ext':         f'diffraq{jnt}2',
    }

    sim = diffraq.Simulator(params)
    sim.run_sim()
