"""
default_parameters.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Default parameters for DIFFRAQ simulations and analyses.

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
    'do_run_vector':        False,      #Will also run non-scalar calculation via Braunbek method

    ### Saving ###
    'do_save':              False,      #Save data?
    'save_dir_base':        None,       #Base directory to save data
    'session':              '',         #Session name (next level folder)
    'save_ext':             '',         #Save extension
    'verbose':              True,       #Print to standard output?
    'with_log':             True,       #Record output to log?
    'free_on_end':          True,       #Delete results after simulation run?

    ### Occulter ###
    'is_finite':            True,       #Diffraction integral does not extend to infinity? Necessary only for multiple shapes
    'occulter_config':      None,       #Configuration file to point to file containing pre-specified shape dictionaries
    'spin_angle':           0,          #Spin angle of the occulter

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
    'radial_nodes':         50,         #Number of radial quadrature nodes
    'theta_nodes':          50,         #Number of azimuthal quadrature nodes
    'fft_tol':              1e-9,       #Tolerance to feed to NUFFT

    ### Polarization ###
    'seam_width':           5e-6,       #Half-width of Braunbek seam
    'is_sommerfeld':        False,      #Use Sommerfeld half-plane solution to calculate edge effect
    'maxwell_func':         None,       #List of lambda functions (for s + p) that hold solution to Maxwell's Eqns. for the differential field near the edge
    'maxwell_file':         None,       #Filename that holds solution to Maxwell's Eqns. for the differential field near the edge
    'polarization_state':   'linear',   #State of polarization. Options: linear, left_circular, right_circular, stokes"
    'polarization_angle':   0,          #Angle of linear polarization relative to Lab frame horizontal [degrees]
    'stokes_parameters':    None,       #Stokes' parameters [I,Q,U,V] describing state of polarization
    'analyzer_angle':       0,          #Angle of camera polarizing analyzer relative to Lab frame horizontal [degrees]
}

############################################
############################################

############################################
####	Default SHAPE Parameters ####
############################################

def_shape_params = {

    ### Outline Function ###
    'kind':                 'polar',    #Functional type of outline
    'edge_func':            None,       #lambda function describing the shape's edge; f(theta) [polar, cart] or f(r) [petal]
    'edge_diff':            None,       #lambda function describing the derivative of the shapes edge; f'(theta) [polar, cart] or f'(r) [petal]
    'edge_data':            None,       #Filename that holds the (x,y) coordinates describing the occulter edge
    'edge_file':            None,       #Data that holds numerical apodization function as a function of radius. Supercedes edge_func and edge_file
    'loci_file':            None,       #Filename that holds numerical apodization function as a function of radius. Supercedes edge_func

    ### Shape Details ###
    'is_opaque':            False,      #Shape is opaque?
    'num_petals':           16,         #Number of petals
    'min_radius':           0,          #Minimum radius
    'max_radius':           12,         #Maximum radius
    'is_clocked':           False,      #Shape is clocked by half a petal (for petal/starshade only)
    'has_center':           True,       #Has central disk? (for petal/starshade only)
    'rotation':             0,          #Angle to rotate individual shape by [radians]
    'perturbations':        [],         #List of dictionaries describing perturbations to be added to the shape
    'etch_error':           None,       #Uniform edge etching error [m]. < 0: removal of material, > 0: extra material.

    ### Numerics ###
    'radial_nodes':         None,       #Number of radial quadrature nodes OR (if < 1) fraction of parent's nodes to use
    'theta_nodes':          None,       #Number of theta quadrature nodes OR (if < 1) fraction of parent's nodes to use

}

############################################
############################################

############################################
####	Default THICK_SCREEN Parameters ####
############################################

def_screen_params = {

    ### Vector solution ###

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
    'is_normalized':        False,      #Is normalized by unblocked simulation
    'wave_ind':             0,          #Wavelength index to show,
    'max_apod':             1.,         #Maximum Apodization value, to convert to contrast
}

############################################
############################################
