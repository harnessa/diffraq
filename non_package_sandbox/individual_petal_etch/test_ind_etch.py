import diffraq
import numpy as np

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
    'radial_nodes':         3000,
    'theta_nodes':          200,

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        f'{diffraq.results_dir}/new_ind_etch_test',
    'session':              'first',

}

num_pet = 12
occulter_is_finite = True
apod_dir = '/home/aharness/repos/diffraq/External_Data/Apodization_Profiles'

etch_error = 70e-9      #Positive means underetched, or more material

#Starshade
starshade = {'kind':'petal', 'is_opaque':False, 'num_petals':num_pet,
    'edge_file':f'{apod_dir}/m12p8.h5', 'has_center':False}

manf_pert = {'xy0':[16.944e-3, 4.067e-3], 'height':40.2e-6, 'width':30.0e-6,
    'kind':'notch', 'direction':1, 'local_norm':False, 'num_quad':50}
starshade['perturbations'] = [manf_pert]

if [False, True][0]:

    #Run single etch
    starshade['etch_error'] = etch_error
    params['save_ext'] = 'single'
    sim = diffraq.Simulator(params, starshade)
    sim.run_sim()

    #Run separate etches
    starshade['kind'] = 'uniquePetal'
    starshade['etch_error'] = etch_error*np.ones(num_pet)
    params['save_ext'] = 'separate'
    sim = diffraq.Simulator(params, starshade)
    sim.run_sim()

    starshade['kind'] = 'uniquePetal'
    starshade['etch_error'] = np.linspace(0, etch_error, num_pet)
    params['save_ext'] = 'distinct'
    sim = diffraq.Simulator(params, starshade)
    sim.run_sim()

else:

    import matplotlib.pyplot as plt;plt.ion()

    #User-input parameters
    params = {

        ### Dual Parameters ###
        'load_dir_base_1':   params['save_dir_base'],
        'session_1':         params['session'],
        'load_ext_1':       'single',
        'load_ext_2':       'separate',
        # 'load_ext_2':       'distinct',
    }

    #Load analysis class and plot results
    duo = diffraq.Dual_Analyzer(params)
    duo.show_results()

    #Compare difference
    dif = duo.alz1.image - duo.alz2.image
    maxper = (1 - duo.alz1.image.max()/duo.alz2.image.max())*100
    difper = dif.max()/duo.alz1.image.max()*100

    plt.figure()
    plt.imshow(dif)
    print(dif.max(), maxper, difper)
    print(duo.alz1.image.max(), duo.alz2.image.max())

    breakpoint()
