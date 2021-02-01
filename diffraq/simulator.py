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
import gc

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
        diffraq.utils.misc_util.set_default_params(self, params, diffraq.utils.def_params)
        self.params = diffraq.utils.misc_util.deepcopy(params)

        ### Derived ###

        if self.save_dir_base is None:
            self.save_dir_base = f"{diffraq.pkg_home_dir}/Results"
        self.waves = np.atleast_1d(self.waves)
        #Effective separation for diverging beam
        self.zeff = self.z0 * self.zz / (self.z0 + self.zz)

    def load_children(self):
        self.logger = diffraq.utils.Logger(self)
        self.occulter = diffraq.world.Occulter(self)
        self.focuser = diffraq.world.Focuser(self)
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
        self.logger.start_up()

        #Open flag
        self.shop_is_open = True

        #Don't forget to close up
        atexit.register(self.close_up_shop)

    def close_up_shop(self):
        if not self.shop_is_open:
            return

        #Run close ups
        self.logger.close_up()

        #Empty trash
        self.clean_up()

        #Close flag
        self.shop_is_open = False

    def clean_up(self):
        #Return if not freeing
        if not self.free_on_end:
            return

        #Names
        trash_dict = {self:['pupil', 'focal'], 'occulter':['xq', 'yq', 'wq']}

        #Loop through objects and attributes and delete
        for obj, att_list in trash_dict.items():
            for att in att_list:
                if hasattr(obj, att):
                    delattr(obj, att)

        #Collect
        n = gc.collect()
        breakpoint()

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

        #Calculate image field
        self.run_image_field()

        #Close shop
        self.close_up_shop()

############################################
############################################

############################################
####	Pupil Field  ####
############################################

    def run_pupil_field(self):
        #TODO: add loading pupil field
        pupil = self.calc_pupil_field()

        #Save + store
        #FIXME: add save
        self.pupil = pupil

    def calc_pupil_field(self):
        #Build target
        grid_pts = diffraq.world.get_grid_points(self.num_pts, self.tel_diameter)

        #Build Area Quadrature
        self.occulter.build_quadrature()

        #Create empty pupil field array
        pupil = np.empty((len(self.waves), self.num_pts, self.num_pts)) + 0j

        #Run diffraction calculation over wavelength
        for iw in range(len(self.waves)):

            #lambda * z
            lamz = self.waves[iw] * self.zeff

            #Calculate diffraction
            uu = diffraq.diffraction.diffract_grid(self.occulter.xq, \
                self.occulter.yq, self.occulter.wq, lamz, grid_pts, self.fft_tol)

            #Store
            pupil[iw] = uu

        return pupil

############################################
############################################

############################################
####	Image Field  ####
############################################

    def run_image_field(self):
        #Return immediately if running pupil only
        if self.skip_image:
            return

        #Calculate image
        focal = self.focuser.calculate_image(self.pupil)

############################################
############################################
