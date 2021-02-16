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

class Simulator(object):

    def __init__(self, params={}, shapes=[]):
        #Set parameters
        self.set_parameters(params)

        #Load children
        self.load_children(shapes)

############################################
####	Initialize  ####
############################################

    def set_parameters(self, params):
        #Set parameters
        diffraq.utils.misc_util.set_default_params(self, params, diffraq.utils.def_sim_params)
        self.params = diffraq.utils.misc_util.deepcopy(params)

        ### Derived ###
        if self.save_dir_base is None:
            self.save_dir_base = f"{diffraq.pkg_home_dir}/Results"

        self.waves = np.atleast_1d(self.waves)

    def load_children(self, shapes):
        #Logging + Saving
        self.logger = diffraq.utils.Logger(self)

        #Occulter
        self.occulter = diffraq.geometry.Occulter(self, shapes)

        #Open flag
        self.shop_is_open = False

    def load_focuser(self):
        #Load focuser child
        self.focuser = diffraq.diffraction.Focuser(self)

############################################
############################################

############################################
##### Open/Close Up Shop #####
############################################

    def open_up_shop(self):
        #Run start ups
        self.logger.start_up()

        #Print
        self.logger.print_start_message()

        #Open flag
        self.shop_is_open = True

    def close_up_shop(self):
        if not self.shop_is_open:
            return

        #Print
        self.logger.print_end_message()

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
        trash_dict = {self:['pupil', 'image'], self.occulter:['xq', 'yq', 'wq']}

        #Loop through objects and attributes and delete
        for obj, att_list in trash_dict.items():
            for att in att_list:
                #Delete if object has attribute
                if hasattr(obj, att):
                    delattr(obj, att)

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
        #Load pupil instead of calculating?
        pupil_loaded = self.load_pupil_field()

        #If pupil loaded, return
        if pupil_loaded:
            return

        #Calculate pupil field
        self.pupil, grid_pts = self.calc_pupil_field()

        #Save pupil field
        self.logger.save_pupil_field(self.pupil, grid_pts)

    ###########################

    def load_pupil_field(self):
        #Return immediately if not loading
        if not self.do_load_pupil:
            return False

        #Check if pupil file exists
        if not self.logger.pupil_file_exists():
            return False

        #Load pupil
        self.pupil, grid_pts = self.logger.load_pupil_field()

        #Return True
        return True

    ###########################

    def calc_pupil_field(self):
        #Build target
        grid_pts = diffraq.utils.image_util.get_grid_points(self.num_pts, self.tel_diameter)

        #Build Area Quadrature
        self.occulter.build_quadrature()

        #Create empty pupil field array
        pupil = np.empty((len(self.waves), self.num_pts, self.num_pts)) + 0j

        #Run diffraction calculation over wavelength
        for iw in range(len(self.waves)):

            #lambda * z
            lamzz = self.waves[iw] * self.zz
            lamz0 = self.waves[iw] * self.z0

            #Calculate diffraction
            uu = diffraq.diffraction.diffract_grid(self.occulter.xq, \
                self.occulter.yq, self.occulter.wq, lamzz, grid_pts, self.fft_tol,
                is_babinet=self.occulter.is_babinet, lamz0=lamz0)

            #Store
            pupil[iw] = uu

        return pupil, grid_pts

############################################
############################################

############################################
####	Image Field  ####
############################################

    def run_image_field(self):
        #Return immediately if running pupil only
        if self.skip_image:
            return

        #Load focuser child
        self.load_focuser()

        #Calculate image
        self.image, grid_pts = self.focuser.calculate_image(self.pupil)

        #Save image
        self.logger.save_image_field(self.image, grid_pts)

############################################
############################################
