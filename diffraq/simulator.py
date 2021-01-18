"""
simulator.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Main class to control simulations for the DIFFRAQ python package.
    #TODO: ADD MORE

"""

import diffraq
import numpy as np
import atexit

class Simulator(object):

    def __init__(self, params={}, do_start_up=False, is_analysis=False):
        # self.is_analysis = is_analysis
        #Set parameters
        self.set_parameters(params)
        #Load children
        self.load_children()
        # #Start up?
        if do_start_up:
            self.open_up_shop()

############################################
####	Initialize  ####
############################################

    def set_parameters(self, params):
        #Set parameters
        diffraq.utils.set_default_params(self, params, diffraq.utils.def_params)
        self.params = diffraq.utils.deepcopy(params)

        ### Derived ###
        if self.save_dir_base is None:
            self.save_dir_base = f"{diffraq.pkg_home_dir}/Results"
        self.waves = np.atleast_1d(self.waves)
        #Effective separation for diverging beam
        self.zeff = self.z0 * self.z1 / (self.z0 + self.z1)

    def load_children(self):
        self.children = ['logger']#, 'occulter']
        self.logger = diffraq.utils.Logger(self)
        # self.occulter = diffraq.occulter.Occulter(self)
        self.shop_is_open = False

############################################
############################################

############################################
##### Open/Close Up Shop #####
############################################

    def open_up_shop(self):
        #Reset if already started up
        if self.shop_is_open:
            self.load_children()

        #Run start ups
        for child in self.children:
            getattr(self, child).start_up()

        #Open flag
        self.shop_is_open = True

        #Don't forget to close up
        atexit.register(self.close_up_shop)

    def close_up_shop(self):
        if not self.shop_is_open:
            return

        #Run close ups
        for child in self.children:
            getattr(self, child).close_up()

        #Close flag
        self.shop_is_open = False

############################################
############################################

############################################
####	Main Script  ####
############################################

    def run_sim(self):

        #Open shop
        self.open_up_shop()

        #Calculate pupil field
        self.run_pupil_field()

        breakpoint()

############################################
############################################

############################################
####	Pupil Field  ####
############################################

    def run_pupil_field(self):
        #TODO: add loading pupil field
        self.calc_pupil_field()

    def calc_pupil_field(self):
        #Build target
        grid_pts = diffraq.world.get_grid_points(self.num_pts, self.tel_diameter)

        #Build AQ
        self.occulter.build_AQ()
        # #TODO: load from occulter
        # gfunc = lambda t: 1 + 0.3*np.cos(3*t)
        # xq, yq, wq = diffraq.quad.polar_quad(gfunc, 120, 350)
        

        #Run diffraction calculation over wavelength
        pupil = np.empty((0, self.num_pts, self.num_pts)) + 0j
        for wav in self.waves:
            #lambda * z
            lamz = wav * self.zeff

            #Calculate diffraction
            uu = diffraq.diff.diffract_grid(self.occulter.xq, self.occulter.yq, \
                self.occulter.wq, lamz, grid_pts, self.fft_tol)

            #Concatenate
            pupil = np.concatenate((pupil, [uu]))

        return pupil

############################################
############################################
