"""
logger.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to log and record results of DIFFRAQ calculations.

"""

from diffraq.utils import misc_util, def_sim_params
import numpy as np
from datetime import datetime
import h5py
import time
import sys
import os

class Logger(object):

    def __init__(self, sim):
        self.sim = sim
        self.copy_params()
        self.log_is_open = False

############################################
#####  Start/Finish #####
############################################

    def copy_params(self):
        pms = ['do_save', 'verbose', 'save_ext', 'with_log']
        for k in pms:
            setattr(self, k, getattr(self.sim, k))

        self.save_dir = f"{self.sim.save_dir_base}/{self.sim.session}"

    def start_up(self):
        #Create save directory
        if self.do_save:
            misc_util.create_directory(self.save_dir)

        #Start
        self.start_time = time.perf_counter()
        self.open_log_file()
        self.save_parameters()
        self.log_is_open = True

    def close_up(self):
        #Finish
        if not self.log_is_open:
            return

        self.end_time = time.perf_counter()
        self.close_log_file()
        self.log_is_open = False

    def print_start_message(self):
        self.write(is_brk=True)
        self.write(f'Running DIFFRAQ with {self.sim.num_pts} x {self.sim.num_pts} Grid ' + \
            f'and {len(self.sim.waves)} wavelengths')
        self.write(f'Shape: {self.sim.occulter_shape}', is_time=False)
        self.write(f'Saved at: {self.save_dir}', is_time=False)
        self.write(f'Save ext: {self.save_ext}', is_time=False)

    def print_end_message(self):
        self.write(is_brk=True)
        self.write(f'Completed in: {time.perf_counter() - self.start_time:.2f} seconds')
        self.write(is_brk=True)

############################################
############################################

############################################
#####  File Functions #####
############################################

    def filename(self, base_name, file_type, ext=None):
        if ext is None:
            ext = self.save_ext
        return misc_util.get_filename(self.save_dir, base_name, ext, file_type)

    def open_log_file(self):
        #Return immediately if not saving
        if not self.do_save or not self.with_log:
            return

        #Open file and register it to close at exit
        self.log_file = open(self.filename('logfile','txt'), 'w')

    def close_log_file(self):
        #Return immediately if not saving
        if not self.do_save or not self.with_log:
            return

        #Close file if still open
        if not self.log_file.closed:
            self.log_file.close()

############################################
############################################

############################################
#####  Writing Functions #####
############################################

    def write(self, txt='',is_brk=False,n_strs=2,is_time=True,is_err=False, is_warn=False):
        #Build message
        if is_brk:
            new_str = '*'*20 + '\n'
        else:
            if is_time:
                txt += f' ({datetime.utcnow()})'
            if is_err:
                txt = misc_util.color_string(f'Error! {txt}', misc_util.bad_color)
            elif is_warn:
                txt = misc_util.color_string(f'Warning! {txt}', misc_util.neutral_color)
            str_str = '*'*int(n_strs)
            new_str = f'{str_str} {txt} {str_str}\n'

        if self.do_save and self.log_is_open and self.with_log:
            self.log_file.write(new_str)

        if self.verbose:
            print(new_str)

    def error(self, txt, is_warning=False):
        self.write(txt=txt, is_err=not is_warning, is_warn=is_warning)
        if not is_warning:
            sys.exit(0)
        else:
            breakpoint()

############################################
############################################

############################################
#####   Saving functions #####
############################################

    def save_parameters(self):
        #Return immediately if not saving
        if not self.do_save:
            return
        #Dump parameters into a json file
        misc_util.json_dump(self.sim.params, self.filename('parameters','json'))
        #Save default parameters as well (in case these get changed)
        misc_util.json_dump(def_sim_params, self.filename('def_params','json'))

    ###########################

    def save_pupil_field(self, pupil, grid_pts):
        #Return immediately if not saving
        if not self.do_save:
            return

        #Save to HDF5
        with h5py.File(self.filename('pupil','h5'), 'w') as f:
            f.create_dataset('field', data=pupil)
            f.create_dataset('grid_pts', data=grid_pts)

    ###########################

    def save_image_field(self, image, grid_pts):
        #Return immediately if not saving
        if not self.do_save:
            return

        #Save to HDF5
        with h5py.File(self.filename('image','h5'), 'w') as f:
            f.create_dataset('intensity', data=image)
            f.create_dataset('grid_pts', data=grid_pts)

############################################
############################################

############################################
#####   Loading functions #####
############################################

    def load_parameters(self, load_dir, load_ext):
        #Get filenames
        usr_file = misc_util.get_filename(load_dir, 'parameters', load_ext, 'json')
        def_file = misc_util.get_filename(load_dir, 'def_params', load_ext, 'json')

        #Load json files
        usr_pms = misc_util.json_load(usr_file)
        def_pms = misc_util.json_load(def_file)

        #Mix parameters
        params = {**def_pms, **usr_pms}

        return params

    ###########################

    def pupil_file_exists(self):
        return os.path.exists(self.filename('pupil','h5', ext=self.sim.pupil_load_ext))

    def load_pupil_field(self):
        #Filename with pupil load extension
        filename = self.filename('pupil','h5', ext=self.sim.pupil_load_ext)

        #Load from HDF5
        with h5py.File(filename, 'r') as f:
            pupil = f['field'][()]
            grid_pts = f['grid_pts'][()]

        return pupil, grid_pts

    ###########################

    def load_image_field(self):
        #Load from HDF5
        with h5py.File(self.filename('image','h5'), 'r') as f:
            image = f['intensity'][()]
            grid_pts = f['grid_pts'][()]

        return image, grid_pts

############################################
############################################
