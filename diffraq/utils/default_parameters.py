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
    'with_log':             False,      #Record output to log?
    'free_on_end':          True,       #Delete results after simulation run?

    ### Occulter ###
    'occulter_is_finite':   True,       #Diffraction integral does not extend to infinity? Necessary only for multiple shapes
    'occulter_config':      None,       #Configuration file to point to file containing pre-specified shape dictionaries
    'occulter_shift':       None,       #Shift (x,y) of occulter position relative to telescope-source line. [m]
    'spin_angle':           0,          #Spin (in-plane) angle of the occulter. [degrees]
    'tilt_angle':           [0,0],      #Pitch (out-of-plane vertical) and roll (out-of-plane horizontal) angles of the occulter. [degrees]

    ### Observation ###
    'z0':                   1e19,       #Source - Occulter distance
    'zz':                   15e6,       #Occulter - Telescope distance
    'tel_diameter':         2.4,        #Telescope diameter
    'num_pts':              256,        #Number of points across telescope
    'waves':                0.6e-6,     #Wavelengths
    'target_center':        [0,0],      #Center of target grid

    ### Laser Beam ###
    'beam_function':        None,       #lambda function that calculates beam's illumination pattern for each input x,y

    ### Focuser ###
    'focal_length':         240,        #Focal length of optics
    'lens_system':          None,       #Supply a lens system to use for lens OPD
    'focus_point':          'source',   #Which plane to focus on. Options: [source, occulter]
    'pixel_size':           13e-6,      #Width of square pixels
    'defocus':              0.,         #Amount of defocus (can be + or -)
    'image_size':           128,        #Width (# pixels) of square image

    ### Numerics ###
    'radial_nodes':         200,        #Number of radial quadrature nodes
    'theta_nodes':          200,        #Number of azimuthal quadrature nodes
    'fft_tol':              1e-9,       #Tolerance to feed to NUFFT
    'seam_radial_nodes':    None,       #Number of radial quadrature nodes in seam. If None, use shape's
    'seam_theta_nodes':     None,       #Number of theta quadrature nodes in seam. If None, use shape's
    'angspec_radial_nodes': 500,        #Number of radial quadrature nodes for ang spec focuser
    'angspec_theta_nodes':  500,        #Number of azimuthal quadrature nodes for ang spec focuser

    ### Polarization ###
    'seam_width':           25e-6,      #Half-width of Braunbek seam
    'is_sommerfeld':        False,      #Use Sommerfeld half-plane solution to calculate edge effect
    'maxwell_func':         None,       #List of lambda functions (for s + p) that hold solution to Maxwell's Eqns. for the differential field near the edge
    'maxwell_file':         None,       #Filename that holds solution to Maxwell's Eqns. for the differential field near the edge
    'polarization_state':   'linear',   #State of polarization. Options: linear, left_circular, right_circular, stokes"
    'polarization_angle':   0,          #Angle of linear polarization relative to Lab frame horizontal [degrees]
    'stokes_parameters':    None,       #Stokes' parameters [I,Q,U,V] describing state of polarization
    'analyzer_angle':       0,          #Angle of camera polarizing analyzer relative to Lab frame horizontal [degrees]
    'with_vector_gaps':     False,      #Use different solution for small gaps between petals?

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
    'edge_data':            None,       #Data that holds numerical apodization function as a function of radius. Supercedes edge_func and edge_file
    'edge_file':            None,       #Filename that holds numerical apodization function as a function of radius. Supercedes edge_func
    'loci_file':            None,       #Filename that holds the (x,y) coordinates describing the occulter edge

    ### Shape Details ###
    'is_opaque':            False,      #Shape is opaque?
    'num_petals':           1,          #Number of petals
    'min_radius':           0,          #Minimum radius
    'max_radius':           12,         #Maximum radius
    'is_clocked':           False,      #Shape is clocked by half a petal (for petal/starshade only)
    'has_center':           True,       #Has central disk? (for petal/starshade only)
    'rotation':             0,          #Angle to rotate individual shape by [radians]
    'perturbations':        [],         #List of dictionaries describing perturbations to be added to the shape
    'unique_edges':         None,       #Separate edge files for unique petals. {edge file: [edge_numbers]}
    'etch_error':           None,       #Uniform edge etching error [m]. < 0: removal of material, > 0: extra material.
    'unique_edge_data':     None,       #Separate edge data for unique petals. {edge data name: edge data} (data name pointed to by edge_file in unique edges)

    ### Numerics ###
    'radial_nodes':         None,       #Number of radial quadrature nodes OR (if < 1) fraction of parent's nodes to use
    'theta_nodes':          None,       #Number of theta quadrature nodes OR (if < 1) fraction of parent's nodes to use

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

    ### Skip data ###
    'skip_pupil':           False,      #Skip loading + showing pupil data
    'skip_image':           False,      #Skip loading + showing image data

    ### Analysis ###
    'cam_analyzer':         None,       #Camera's polarized analyzer mode. Options: [None:unpolarized, 'p':primary polarization, 'o':orthogonal polarization]
    'wave_ind':             0,          #Wavelength index to show,

    ### Plotting ###
    'image_vmin':           None,       #Min value to show in image plot
    'image_vmax':           None,       #Max value to show in image plot

    ### Contrast ###
    'is_contrast':          False,      #Should we normalize as a contrast measurement?
    'calibration_file':     None,       #File pointing to calibration image for which to normalize image to contrast
    'max_apod':             1.,         #Maximum Apodization value, to convert to contrast
    'freespace_corr':       1.,         #Freespace correction for contrast measurement. If wavelength dependent, supply dictionary with {wave [nm]: correction}
    'fit_airy':             False,      #Fit Airy pattern to calibration image to get peak?
}

############################################
############################################
