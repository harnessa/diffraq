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

#####################
#####   MPI #####
#####################

try:
    from mpi4py import MPI
    mpi_rank = MPI.COMM_WORLD.rank      # processor ID number, from 0 up to size
    mpi_size = MPI.COMM_WORLD.size      # total number of processors running
    mpi_barrier = MPI.COMM_WORLD.Barrier
    has_mpi = True
except ImportError:
    mpi_rank = 0
    mpi_size = 1
    mpi_barrier = lambda : None
    has_mpi = False
zero_rank = mpi_rank == 0

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

#####################
#####   Modules #####
#####################

import diffraq.utils
util = diffraq.utils.Utilities()
from diffraq.simulator import Simulator
import diffraq.quadrature as quad
