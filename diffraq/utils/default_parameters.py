"""
default_parameters.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Default parameters for DIFFRAQ simulations.

"""

""" All units in [m] unless specificed others """

############################################
####	Default SIMULATION Parameters ####
############################################

def_sim_params = {

    ### Simulator ###
    'skip_image':           False,      #Skip image calculation?
    'do_load_pupil':        False,      #Load pupil instead of calculating?
    'pupil_load_ext':       None,       #Extension to load pupil field from (if different from saving ext)

    ### Saving ###
    'do_save':              False,      #Save data?
    'save_dir_base':        None,       #Base directory to save data
    'session':              '',         #Session name (next level folder)
    'save_ext':             '',         #Save extension
    'verbose':              True,       #Print to standard output?
    'with_log':             True,       #Record output to log?
    'free_on_end':          True,       #Delete results after simulation run?

    ### Occulter ###
    'occulter_shape':       'circle',   #Shape of occulter. Options: [circle, polar, starshade]
    'circle_rad':           12,         #Circle occulter radius
    'apod_file':            None,       #Filename that holds numerical apodization function as a function of radius. Supercedes apod_func
    'apod_func':            None,       #Lambda function (accepts radius as argument) defining apodization function
    'is_babinet':           False,      #Using Babinet principle to calculate?

    ### Starshades ###
    'num_petals':           16,         #Number of starshade petals
    'ss_rmin':              5,          #Minimum starshade radius
    'ss_rmax':              13,         #Maximum starshade radius

    ### Observation ###
    'waves':                0.6e-6,     #Wavelengths
    'z0':                   1e19,       #Source - Occulter distance
    'zz':                   15e6,       #Occulter - Telescope distance
    'tel_diameter':         2.4,        #Telescope diameter
    'num_pts':              256,        #Number of points across telescope

    ### Focuser ###
    'focal_length':         240,        #Focal length of optics
    'focus_point':          'source',   #Which plane to focus on. Options: [source, occulter]
    'pixel_size':           13e-6,      #Width of square pixels
    'defocus':              0.,         #Amount of defocus (can be + or -)
    'image_size':           128,        #Width (# pixels) of square image

    ### Numerics ###
    'radial_nodes':         20,
    'theta_nodes':          20,
    'fft_tol':              1e-9,       #Tolerance to feed to NUFFT

}

############################################
############################################

############################################
####	Default ANALYSIS Parameters ####
############################################

def_alz_params = {

    ### Loading ###
    'load_dir_base':        None,       #Base directory to load data from
    'session':              '',         #Session name (next level folder)
    'load_ext':             '',         #Load extension

    ### Analysis to run ###
    'skip_pupil':           False,      #Skip loading + showing pupil data
    'skip_image':           False,      #Skip loading + showing image data

    ### Analysis ###
    'is_normalized':        True,       #Is normalized by unblocked simulation
    'wave_ind':             0,          #Wavelength index to show,

}

############################################
############################################
