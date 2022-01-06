import diffraq
import numpy as np
import dial2
import imp

params = {

    ### World ###
    'num_pts':              512,
    'zz':                   49.986,
    'tel_diameter':         5e-3,
    'waves':                np.array([641, 660, 699, 725])*1e-9,
    'focal_length':         0.496,
    'defocus':              8e-4,
    'image_size':           128,
    'focus_point':          'occulter',

    ### Numerics ###
    'radial_nodes':         5000,
    'theta_nodes':          300,

    ### Saving ###
    'do_save':              True,
    'save_dir_base':        f'{diffraq.results_dir}/beam_test',
    'session':              'neg_sphere',

}

shape = {'kind':'circle', 'is_opaque':False,'max_radius':25e-3}

if [False, True][0]:

    #Plane wave
    params['save_ext'] = 'plane'

    sim = diffraq.Simulator(params, shape)
    sim.run_sim()

    #Spherical beam
    z0 = 27.5
    params['z0'] = z0
    params['save_ext'] = 'neg_sphere'

    def beam(xq, yq, wv):
        return np.exp(-1j*2*np.pi/wv*(xq**2 + yq**2)/(2*z0))

    params['beam_function'] = beam

    sim = diffraq.Simulator(params, shape)
    sim.run_sim()

else:

    import matplotlib.pyplot as plt;plt.ion()

    params = {

        ### Dual Parameters ###
        'load_dir_base_1':   f'{diffraq.results_dir}/beam_test',
        'session_1':        'neg_sphere',
        'load_ext_1':       'plane',
        'load_ext_2':       'neg_sphere',

        ### Analyzer Parameters ###
        'cam_analyzer':     ['p','o'][0],
        'wave_ind':         3,

    }

    duo = diffraq.Dual_Analyzer(params)
    duo.show_results()

    #Compare difference
    dif = duo.alz1.image - duo.alz2.image
    maxper = (1 - duo.alz1.image.max()/duo.alz2.image.max())*100
    difper = dif.max()/duo.alz1.image.max()*100

    plt.figure()
    plt.imshow(dif)
    print(abs(dif).max(), maxper, difper)
    print(duo.alz1.image.max(), duo.alz2.image.max())



    breakpoint()
