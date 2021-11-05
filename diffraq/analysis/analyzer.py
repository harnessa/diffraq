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
from diffraq.polarization import VectorSim
import h5py
import matplotlib.pyplot as plt

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
        att_list = ['sim', 'pupil', 'image', 'pupil_waves', 'image_waves',\
            'pupil_xx', 'image_xx', 'pupil_image']

        #Cleanup sim
        if hasattr(self, 'sim'):
            self.sim.clean_up()

        #Loop through attributes and delete
        for att in att_list:
            #Delete if self has attribute
            if hasattr(self, att):
                delattr(self, att)

############################################
############################################

############################################
####	Properties ####
############################################

    @property
    def pupil_dx(self):
        if hasattr(self, 'pupil_xx'):
            return self.pupil_xx[1] - self.pupil_xx[0]

    @property
    def image_dx(self):
        if hasattr(self, 'image_xx'):
            return self.image_xx[1] - self.image_xx[0]

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
        pupil, self.pupil_xx, vec_pupil, vec_comps, is_polarized = \
            self.sim.logger.load_pupil_field()

        #If polarized, apply camera analyzer
        if is_polarized:
            #Get full polarized field
            field = VectorSim.build_polarized_field(self.sim, pupil, vec_pupil, \
                vec_comps, self.sim.analyzer_angle)
            #Apply analyzer
            self.pupil = self.apply_cam_analyzer(field)
        else:
            self.pupil = pupil

        #TODO: add pupil calibration
        #Normalize with calibration data
        # self.calibrate_pupil()

        #Store all waves
        self.pupil_waves = self.pupil.copy()

        #Take one wavelength only
        self.pupil = self.pupil_waves[self.wave_ind]
        self.pupil_image = np.abs(self.pupil)**2

    ############################################

    def load_image_data(self):
        #Return if skipping or sim didn't run image
        if self.skip_image or self.sim.skip_image:
            return

        #Load image data
        image, self.image_xx, is_polarized = self.sim.logger.load_image_field()

        #If polarized, apply camera analyzer
        if is_polarized:
            self.image = self.apply_cam_analyzer(image)
        else:
            self.image = image

        #Normalize with calibration data
        self.calibrate_image()

############################################
############################################

############################################
####	Main Script ####
############################################

    def show_results(self):

        plt.ion()
        if hasattr(self, 'image'):
            plt.imshow(self.image, vmin=self.image_vmin, vmax=self.image_vmax)
            print(self.image.max())
        else:
            plt.imshow(self.pupil_image)
            print(self.pupil_image.max())
        breakpoint()

############################################
############################################

############################################
####	Calibration ####
############################################

    def calibrate_image(self):
        #Convert to contrast?
        if self.is_contrast:
            self.convert_to_contrast()

        #Store all waves
        self.image_waves = self.image.copy()

        #Take one wavelength only
        self.image = self.image_waves[self.wave_ind]

    def convert_to_contrast(self):

        #Get calibration value
        if self.calibration_file is None:
            cal_val = (self.sim.z0/(self.sim.z0 + self.sim.zz))**2
        else:
            cal_val = self.get_calibration_value()

        #Get freespace correction
        if isinstance(self.freespace_corr, dict):
            fcorr = np.array([self.freespace_corr[np.round(wv*1e9,0)] for wv in self.sim.waves])
        else:
            fcorr = self.freespace_corr

        #Store calibration value
        self.cal_val = fcorr * cal_val * self.max_apod**2.

        #Convert to contrast
        self.image /= self.cal_val[:,None,None]

    def get_calibration_value(self):
        #Load calibration image
        with h5py.File(self.calibration_file, 'r') as f:
            cal_img = f['intensity'][()]
            cal_wvs = f['waves'][()]
            is_polarized = f['is_polarized'][()]

        #Always take parallel polarization for calibration value
        if is_polarized:
            cal_img = cal_img[:,0]

        #Get matching wavelengths
        if len(cal_wvs) != len(self.sim.waves):
            #Get closest wavelength
            wv_inds = [np.argmin(np.abs(cal_wvs - wv)) for wv in self.sim.waves]
            #Throw out if too different
            wv_inds = np.array(wv_inds)[np.abs(self.sim.waves - cal_wvs[wv_inds]) < 5e-9]
            #keep images at those wavelengths
            cal_img = cal_img[wv_inds]

        #Get calibration value
        if not self.fit_airy:
            #Use max value
            cal_val = cal_img.max((1,2))

        else:
            #Fit airy pattern with astropy
            from astropy.modeling import models, fitting

            cal_val = np.array([])
            for iw in range(cal_img.shape[0]):
                #Get sub image
                wid = int(self.sim.waves[iw]/self.sim.tel_diameter * \
                    misc_util.rad2arcsec/self.image_dx) * 10
                sub_img = image_util.crop_image(cal_img[iw], None, wid)

                #Fit Airy 2D
                gg_init = models.AiryDisk2D(sub_img.max(),len(sub_img)//2,len(sub_img)//2,5)
                fitter = fitting.LevMarLSQFitter()
                y,x = np.indices(sub_img.shape)
                weights = np.sqrt(sub_img - sub_img.min())
                weights[weights == 0.] = 1e-12
                gg_fit = fitter(gg_init, x, y, sub_img, weights=weights)

                #Get peak value
                cal_val = np.concatenate((cal_val, [gg_fit.amplitude.value]))

        return cal_val

############################################
############################################

############################################
####	Polarization ####
############################################

    def apply_cam_analyzer(self, image):
        #Unpolarized
        if self.cam_analyzer is None:
            image = image.sum(1)
        #Parallel polarization
        elif self.cam_analyzer.startswith('p'):
            image = image[:,0]
        #Orthogonal polarization
        elif self.cam_analyzer.startswith('o'):
            image = image[:,1]

        return image

############################################
############################################
