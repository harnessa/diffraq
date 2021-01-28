"""
logger.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to log and record results of DIFFRAQ calculations.

"""

from diffraq.utils import misc_utils, def_params
import numpy as np
from datetime import datetime
import h5py
import time
import sys

class Logger(object):

    def __init__(self, sim):
        self.sim = sim
        self.copy_params()
        self.log_is_open = False

############################################
#####  Start/Finish #####
############################################

    def copy_params(self):
        pms = ['do_save', 'verbose', 'save_ext']
        for k in pms:
            setattr(self, k, getattr(self.sim, k))

    def start_up(self):
        #Create save directory
        if self.do_save:
            self.save_dir = f"{self.sim.save_dir_base}/{self.sim.session}"
            misc_utils.create_directory(self.save_dir)

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

############################################
############################################

############################################
#####  File Functions #####
############################################

    def filename(self,base_name,file_type,ext=None):
        if ext is None:
            ext = self.save_ext
        return misc_utils.get_filename(self.save_dir, base_name, ext, file_type)

    def open_log_file(self):
        #Return immediately if not saving
        if not self.do_save:
            return

        #Open file and register it to close at exit
        self.log_file = open(self.filename('logfile','txt'), 'w')

    def close_log_file(self):
        #Return immediately if not saving
        if not self.do_save:
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
                txt = misc_utils.color_string(f'Error! {txt}', misc_utils.bad_color)
            elif is_warn:
                txt = misc_utils.color_string(f'Warning! {txt}', misc_utils.neutral_color)
            str_str = '*'*int(n_strs)
            new_str = f'{str_str} {txt} {str_str}\n'

        if self.do_save and self.log_is_open:
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
        misc_utils.json_dump(self.sim.params, self.filename('parameters','json'))
        #Save default parameters as well (in case these get changed)
        misc_utils.json_dump(def_params, self.filename('def_params','json'))

############################################
############################################
