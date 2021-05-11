import diffraq
import numpy as np
import h5py

params = {

    ### World ###
    'num_pts':              512,
    'zz':                   49.986,
    'z0':                   27.455,
    'tel_diameter':         5e-3,
    'waves':                641e-9,
    'focal_length':         0.499,
    'image_size':           128,

    ### Numerics ###
    'radial_nodes':         5000,
    'theta_nodes':          300,
    'occulter_is_finite':   True,

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        f'{diffraq.results_dir}/sandbox',
    'session':              'new_petal_shift',
    'save_ext':             'old_pet2',
}

##########################################################################

#Shift amounts and angles of shifted petals
# petal_shifts = {2: 7.5e-6, 6: 7.5e-6, 10: 10.5e-6}
petal_shifts = {2: 7.5e-6}

num_pet = 12
max_apod = 0.9
occulter_is_finite = True
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

#Inner starshade
inn_starshade = {'kind':'starshade', 'is_opaque':True, 'num_petals':num_pet, \
    'edge_file':f'{apod_dir}/bb_2017__inner.h5', 'is_clocked':True}

#Outer starshade
out_starshade = {'kind':'starshade', 'is_opaque':False, 'num_petals':num_pet, \
    'edge_file':f'{apod_dir}/bb_2017__outer.h5'}

#Read max/min radii for struts
with h5py.File(inn_starshade['edge_file'], 'r') as f:
    strut_rmin = f['data'][-1,0]

with h5py.File(out_starshade['edge_file'], 'r') as f:
    strut_rmax = f['data'][0,0]

#Build perturbations
inn_perts = []
for pet_num, shift in petal_shifts.items():

    #Angles between petals
    angles = np.pi/num_pet * np.array([2*(pet_num-1), 2*pet_num])

    #Build shifted petal perturbation
    pert_n = {'kind':'shifted_petal', 'angles':angles, 'shift':shift, 'num_quad':50}

    #Append
    inn_perts.append(pert_n)

#Get starshades and struts from base occulter and add perturbations
inn_starshade['perturbations'] = inn_perts

#Build modify struts with clipped minimum radius
struts = []
for i in range(num_pet):
    #Shift minimum radius outwards
    if i+1 in petal_shifts.keys():
        min_rad = strut_rmin + petal_shifts[i+1]
    else:
        min_rad = strut_rmin

    #Build new strut
    strut_n = {'kind':'petal', 'is_opaque':True, 'num_petals':1, 'has_center':False,
        'edge_func':lambda t: np.ones_like(t)*(1-max_apod)/num_pet,
        'min_radius':min_rad, 'max_radius':strut_rmax, 'radial_nodes':50,
        'theta_nodes':50, 'rotation': 2*np.pi/num_pet*(i+0.5)}
    struts.append(strut_n)

#Build shape list
shapes = [inn_starshade, out_starshade, *struts]

##########################################################################

#Load simulator and run sim
sim = diffraq.Simulator(params, shapes)
sim.run_sim()
