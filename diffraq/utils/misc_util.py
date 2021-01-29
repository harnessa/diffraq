"""
misc_util.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Miscellaneous utility functions to be used by DIFFRAQ.

"""

import numpy as np
import copy
import os
import json
import base64

#Constants
m2mu = 1e6
mu2m = 1e-6

### Text Colors ###
bad_color = "\x1b[0;30;41m"
good_color = "\x1b[0;30;42m"
off_color = '\x1b[0m'
neutral_color = "\x1b[0;33;40m"

############################################
#####   Misc   #####
############################################

def deepcopy(in_obj):
    return copy.deepcopy(in_obj)

def color_string(string,color):
    return color + string + off_color

############################################
############################################

############################################
#####   Directories   #####
############################################

def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def get_filename(dir_name,base_name,ext,file_type):
    if ext != '':
        ext = '__' + ext
    return rf"{dir_name}/{base_name}{ext}.{file_type}"

############################################
############################################

############################################
#####   Parameter Functions    #####
############################################

def set_default_params(parent, params, in_def_pms):
    bad_str = color_string('!*!', bad_color)

    #Copy defaults
    def_pms = deepcopy(in_def_pms)

    #Set default parameters
    for k,v in def_pms.items():
        setattr(parent, k, v)

    #Overwrite with user-specified parameters
    def_keys = def_pms.keys()
    for k,v in params.items():
        #Error message if unknown parameter supplied
        if k not in def_keys:
            col_str = color_string(k, neutral_color)
            print(f'\n{bad_str} New Parameter not in Defaults: {col_str} {bad_str}\n')
            import sys; sys.exit(0)

        setattr(parent, k, v)

############################################
############################################

############################################
#####   JSON Decoder/Encoder    #####
############################################
"""
Encoder/Decoder code from:
    https://stackoverflow.com/questions/27909658/json-encoder-and-decoder-for-complex-numpy-arrays
"""

class Numpy_Encoder(json.JSONEncoder):
    def default(self, obj):
        """
        if input object is a ndarray it will be converted into a dict holding dtype, shape and the data base64 encoded
        """
        if isinstance(obj, np.ndarray):
            data_b64 = base64.b64encode(np.ascontiguousarray(obj).data).decode()
            return dict(__ndarray__=data_b64, dtype=str(obj.dtype), shape=obj.shape)
        elif callable(obj):
            return 'lambda'
        # Let the base class default method raise the TypeError
        return json.JSONEncoder(default=self, indent=obj)

def json_numpy_obj_hook(dct):
    """
    Decodes a previously encoded numpy ndarray with proper shape and dtype
    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'])
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    return dct

#My JSON wrappers
def json_dump(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, cls=Numpy_Encoder)

def json_load(filename):
    with open(filename, 'r') as f:
        data = json.load(f, object_hook=json_numpy_obj_hook)
    return data

############################################
############################################
