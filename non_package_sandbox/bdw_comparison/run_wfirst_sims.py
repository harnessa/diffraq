from bdw import BDW
import diffraq
import numpy as np

wave = 0.641e-6
tel_diameter = 2.4
z1 = 37.24e6 * 552./641.
z0 = 1e19
focal_length = 240
num_pts = 256
num_foc = 128
num_pet = 22

session = 'wfirst'
save_dir_base = f'{diffraq.results_dir}/bdw_compare'

bdw_params = {
        'wave':             wave,
        'z0':               z0,
        'z1':               z1,
        'tel_diameter':     tel_diameter,
        'num_tel_pts':      num_pts,
        'num_petals':       num_pet,
        'image_pad':        0,
        'apod_name':        'wfirst',
        'is_connected':     True,
        'do_save':          True,
        'save_dir_base':    save_dir_base,
        'session':          session,
        'save_ext':         'bdw',
}

mm = 6000
nn = 300

shape = {'kind':'starshade', 'is_opaque':True,  \
    'num_petals':num_pet,'edge_file':f'{diffraq.apod_dir}/wfirst_ni2.txt'}

dif_params = {

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

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    save_dir_base,
    'session':          session,
    'save_ext':         f'diffraq',
    'skip_image':       True,
}

if [False, True][0]:
    sim = diffraq.Simulator(dif_params, shape)
    sim.run_sim()

if [False, True][1]:
    bdw = BDW(bdw_params)
    bdw.run_sim()
