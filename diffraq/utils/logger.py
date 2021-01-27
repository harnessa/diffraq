"""
logger.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to log and record results of DIFFRAQ calculations.

"""

import diffraq.utils as utils
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
            utils.create_directory(self.save_dir)

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
        return utils.get_filename(self.save_dir, base_name, ext, file_type)

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

    def write(self, txt='',is_brk=False,n_strs=2,is_time=True,is_err=False):
        #Build message
        if is_brk:
            new_str = '*'*20 + '\n'
        else:
            if is_time:
                txt += f' ({datetime.utcnow()})'
            if is_err:
                txt = utils.color_string(f'Error! {txt}', utils.bad_color)
            str_str = '*'*int(n_strs)
            new_str = f'{str_str} {txt} {str_str}\n'

        if self.do_save and self.log_is_open:
            self.log_file.write(new_str)

        if self.verbose:
            print(new_str)

    def error(self, txt, do_exit=True):
        self.write(txt=txt, is_err=True)
        if do_exit:
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
        utils.json_dump(self.sim.params, self.filename('parameters','json'))
        #Save default parameters as well (in case these get changed)
        utils.json_dump(utils.def_params, self.filename('def_params','json'))

############################################
############################################
