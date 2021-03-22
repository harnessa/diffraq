from bdw import BDW
import diffraq
import numpy as np

wave = 0.641e-6
tel_diameter = 5e-3
z1 = 50.
z0 = 27.5
num_pts = 256
num_pet = 12
focal_length = 0.498

apod = 'bb_2017'
etch = -1e-6

session = 'etch'
save_dir_base = f'{diffraq.results_dir}/bdw_compare_new'

sgn = {1:'p', -1:'n'}[np.sign(etch)]
ext = f'__etch_{sgn}{abs(etch*1e6):.0f}'

bdw_params = {
        'wave':             wave,
        'z0':               z0,
        'z1':               z1,
        'tel_diameter':     tel_diameter,
        'focal_length':     focal_length,
        'num_tel_pts':      num_pts,
        'num_petals':       num_pet,
        'num_occ_pts':      2000,
        'image_pad':        0,
        'loci_file':        apod + ext,
        'do_save':          True,
        'save_dir_base':    save_dir_base,
        'session':          session,
        'save_ext':         f'bdw{ext}',
}

mm = 6000
nn = 300

num_pet = 12
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'
shape = {'kind':'starshade', 'is_opaque':False, 'num_petals':num_pet, \
    'edge_file':f'{apod_dir}/{apod}.txt', 'has_center':False, 'etch_error':etch}

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

    ### Occulter ###
    'is_finite':        True,

    ### Saving ###
    'do_save':          True,
    'save_dir_base':    save_dir_base,
    'session':          session,
    'save_ext':         f'diffraq{ext}',
}

if [False, True][1]:
    sim = diffraq.Simulator(dif_params, shape)
    sim.run_sim()

if [False, True][0]:
    bdw = BDW(bdw_params)
    bdw.run_sim()
