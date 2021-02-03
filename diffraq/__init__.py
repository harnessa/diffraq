"""
__init__.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: __init__ package for the DIFFRAQ python package. Try to load mpi4py
    and set processor rank and size. Try to find Environment Variable. Load modules.

"""

import numpy as np
import os

#########################
#####   Directories #####
#########################

pkg_home_dir = os.getenv("DIFFRAQ")

if pkg_home_dir is None:
    if zero_rank:
        print("\n*** Cannot Find Environment Variable pointing to DIFFRAQ home! ***\n")
        print("* Please set environment variable $DIFFRAQ pointing to directory where diffraq/setup.py is located *")
    import sys
    sys.exit()

ext_data_dir = f"{pkg_home_dir}/External_Data"
int_data_dir = f"{pkg_home_dir}/Internal_Data"
apod_dir = f"{ext_data_dir}/Apodization_Profiles"

#####################
#####   Modules #####
#####################

import diffraq.utils
from diffraq.simulator import Simulator
import diffraq.quadrature
import diffraq.diffraction
import diffraq.world
from diffraq.analysis import Analyzer
