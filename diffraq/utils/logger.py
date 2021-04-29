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

############################################
#####  Start/Finish #####
############################################

    def copy_params(self):
        pms = ['do_save', 'verbose', 'save_ext', 'with_log']
        for k in pms:
            setattr(self, k, getattr(self.sim, k))

        self.save_dir = f"{self.sim.save_dir_base}/{self.sim.session}"
        self.log_needs_dump = False

    def start_up(self):
        #Create save directory
        if self.do_save:
            misc_util.create_directory(self.save_dir)

        #Start
        self.start_time = time.perf_counter()
        self.open_log_file()
        self.save_parameters()

    def close_up(self):
        #Finish
        self.end_time = time.perf_counter()
        self.dump_log_file()

    def print_start_message(self):
        self.write(is_brk=True)
        self.write(f'Running DIFFRAQ with {self.sim.num_pts} x {self.sim.num_pts} Grid ' + \
            f'and {len(self.sim.waves)} wavelengths')
        if self.do_save:
            self.write(f'Saved at: {self.save_dir}', is_time=False)
            self.write(f'Save ext: {self.save_ext}', is_time=False)
        self.write(is_brk=True)

    def print_end_message(self):
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

        #Create container for log file string and register to add to log file
        self.log_file = ''
        self.log_needs_dump = True

    def dump_log_file(self):
        #Return immediately if not writeable (or log file is already dumped)
        if not self.do_save or not self.log_needs_dump:
            return

        #Dump to file
        with open(self.filename('logfile','txt'), 'w') as f:
            f.write(self.log_file)

        #Clear flag
        self.log_needs_dump = False

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

        if self.do_save and self.with_log:
            self.log_file += new_str + '\n'
        if self.verbose:
            print(new_str)

    def error(self, txt, is_warning=False):
        self.write(txt=txt, is_err=not is_warning, is_warn=is_warning)
        if not is_warning:
            sys.exit(0)
        else:
            print(txt)
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

    def save_pupil_field(self, pupil, grid_pts, vec_pupil, vec_comps):
        #Return immediately if not saving
        if not self.do_save:
            return

        #TODO: save wavelengths, basically save all parameters for easy access (dont have to run analyzer)

        #Are we polarized?
        is_polarized = vec_pupil is not None

        #Save to HDF5
        with h5py.File(self.filename('pupil','h5'), 'w') as f:
            f.create_dataset('field', data=pupil)
            f.create_dataset('grid_pts', data=grid_pts)
            f.create_dataset('waves', data=self.sim.waves)
            #Save vector pupil
            f.create_dataset('is_polarized', data=is_polarized)
            if is_polarized:
                f.create_dataset('vector_field', data=vec_pupil)
                f.create_dataset('vector_comps', data=vec_comps)

    ###########################

    def save_image_field(self, image, grid_pts):
        #Return immediately if not saving
        if not self.do_save:
            return
        #TODO: save wavelengths, basically save all parameters for easy access (dont have to run analyzer)

        #Are we polarized
        is_polarized = image.ndim == 4

        #Save to HDF5
        with h5py.File(self.filename('image','h5'), 'w') as f:
            f.create_dataset('intensity', data=image)
            f.create_dataset('grid_pts', data=grid_pts)
            f.create_dataset('waves', data=self.sim.waves)
            f.create_dataset('is_polarized', data=is_polarized)

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
            #Check if polarized key available
            if 'is_polarized' in f.keys():
                is_polarized = f['is_polarized'][()]
            else:
                is_polarized = False
            #If polarized, load vector field
            if is_polarized:
                vec_pupil = f['vector_field'][()]
                vec_comps = f['vector_comps'][()]
            else:
                vec_pupil, vec_comps = None, None

        return pupil, grid_pts, vec_pupil, vec_comps

    ###########################

    def load_image_field(self):
        #Load from HDF5
        with h5py.File(self.filename('image','h5'), 'r') as f:
            image = f['intensity'][()]
            grid_pts = f['grid_pts'][()]
            #Check if polarized key available
            if 'is_polarized' in f.keys():
                is_polarized = f['is_polarized'][()]
            else:
                is_polarized = False

        return image, grid_pts, is_polarized

############################################
############################################
