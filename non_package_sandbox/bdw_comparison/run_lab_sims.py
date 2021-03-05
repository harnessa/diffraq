from bdw import BDW
import diffraq
import numpy as np

wave = 0.641e-6
tel_diameter = 5e-3
z1 = 50.
z0 = 27.5
num_pts = 256
num_pet = 12

apod = 'bb_2017'

session = apod
save_dir_base = f'{diffraq.results_dir}/bdw_compare'

bdw_params = {
        'wave':             wave,
        'z0':               z0,
        'z1':               z1,
        'tel_diameter':     tel_diameter,
        'num_tel_pts':      num_pts,
        'num_petals':       num_pet,
        'num_occ_pts':      4000,
        'image_pad':        0,
        'apod_name':        apod,
        'do_save':          True,
        'save_dir_base':    save_dir_base,
        'session':          session,
        'save_ext':         'bdw_4k',
}

mm = 6000
nn = 300

dif_params = {

    ### World ###
    'num_pts':          num_pts,
    'zz':               z1,
    'z0':               z0,
    'tel_diameter':     tel_diameter,
    'waves':            wave,

    ### Numerics ###
    'radial_nodes':     mm,
    'theta_nodes':      nn,

    ### Occulter ###
    'occulter_config':  f'{diffraq.occulter_dir}/{apod}.cfg',
    'is_finite':        True,

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    save_dir_base,
    'session':          session,
    'save_ext':         f'diffraq',
    'skip_image':       True,
}

if [False, True][1]:
    sim = diffraq.Simulator(dif_params)
    sim.run_sim()

if [False, True][0]:
    bdw = BDW(bdw_params)
    bdw.run_sim()
