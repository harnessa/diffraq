from bdw import BDW
import diffraq
import numpy as np

wave = 0.641e-6
tel_diameter = 5e-3
z1 = 50.
z0 = 27.5
num_pts = 256
num_pet = 12

apod = 'M12P6'

session = apod
save_dir_base = f'{diffraq.results_dir}/bdw_compare'

dec_ext = '_4x'
bdw_params = {
        'wave':             wave,
        'z0':               z0,
        'z1':               z1,
        'tel_diameter':     tel_diameter,
        'num_tel_pts':      num_pts,
        'num_petals':       num_pet,
        'num_occ_pts':      2000,
        'image_pad':        0,
        # 'apod_name':        apod,             #For bb_2017
        'loci_file':        apod + dec_ext,
        'do_save':          True,
        'save_dir_base':    save_dir_base,
        'session':          session,
        'save_ext':         'bdw' + dec_ext,
}

mm = 6000
nn = 300

# mm = 100
# nn = 30

dif_params = {

    ### World ###
    'num_pts':          num_pts,
    'zz':               z1,
    'z0':               z0,
    'tel_diameter':     tel_diameter,
    'waves':            wave,
    'focal_length':     0.498,
    'image_size':       128,

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
    'save_ext':         f'diffraq7',
    # 'skip_image':       True,
}

if [False, True][1]:
    sim = diffraq.Simulator(dif_params)
    sim.run_sim()

if [False, True][0]:
    bdw = BDW(bdw_params)
    bdw.run_sim()
