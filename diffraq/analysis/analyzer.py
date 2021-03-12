"""
analyzer.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-01-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to analyze results of DIFFRAQ simulation.
"""

import numpy as np
from diffraq.utils import image_util, misc_util, def_alz_params, Logger
from diffraq import pkg_home_dir, Simulator

class Analyzer(object):

    def __init__(self, params={}):
        #Set parameters
        self.set_parameters(params)

        #Load data
        self.load_data()

############################################
####	Start up ####
############################################

    def set_parameters(self, params):
        #Set default parameters
        misc_util.set_default_params(self, params, def_alz_params)

        ### Derived ###
        if self.load_dir_base is None:
            self.load_dir_base = f"{pkg_home_dir}/Results"
        self.load_dir = f'{self.load_dir_base}/{self.session}'

    def load_data(self):
        #Load parameters (user-input + default)
        sim_params = Logger.load_parameters(None, self.load_dir, self.load_ext)

        #Overwrite analysis parameters
        sim_params = self.overwrite_analysis_params(sim_params)

        #Create SIM instance
        self.sim = Simulator(sim_params)

        #Load pupil data
        self.load_pupil_data()

        #Load image data
        self.load_image_data()

    def overwrite_analysis_params(self, sim_params):
        #Overwrite values
        ovr_dict = {'verbose':False, 'do_save':False}

        for k, v in ovr_dict.items():
            sim_params[k] = v

        #Shared values
        shr_dict = {'save_dir_base':'load_dir_base', 'save_ext':'load_ext', \
            'session': 'session'}

        for sk, ak in shr_dict.items():
            sim_params[sk] = getattr(self, ak)

        return sim_params

    def clean_up(self):
        att_list = ['pupil', 'image', 'sim']

        #Loop through attributes and delete
        for att in att_list:
            #Delete if self has attribute
            if hasattr(obj, att):
                delattr(obj, att)

############################################
############################################

############################################
####	Load Data ####
############################################

    def load_pupil_data(self):
        #Return if skipping
        if self.skip_pupil:
            return

        #Load pupil data
        self.pupil, self.pupil_xx = self.sim.logger.load_pupil_field()

        #Normalize with calibration data
        # self.calibrate_pupil()

        #Get coordinates

        # breakpoint()

    def load_image_data(self):
        #Return if skipping or sim didn't run image
        if self.skip_image or self.sim.skip_image:
            return

        #Load image data
        self.image, self.image_xx = self.sim.logger.load_image_field()

        #Normalize with calibration data
        self.calibrate_image()

############################################
############################################

############################################
####	Calibration Values ####
############################################

    def calibrate_pupil(self):
        # #FIXME: load open data
        # if self.is_normalized:
        #
        #     breakpoint()
        # else:
        #
        #     breakpoint()
        return

    def calibrate_image(self):
        # #FIXME: load open data
        # if self.is_normalized:
        #
        #     breakpoint()
        # else:
        #     self.image *= (self.sim.z0/(self.sim.z0 + self.sim.zz))**2
        #
        # #Convert to contrast by dividing by blocked apodization area
        # self.image /= self.max_apod**2.
        return

############################################
############################################

############################################
####	Main Script ####
############################################

    def show_results(self):

        import matplotlib.pyplot as plt;plt.ion()
        plt.imshow(self.image[0])
        print(self.image[0].max())
        breakpoint()

############################################
############################################
