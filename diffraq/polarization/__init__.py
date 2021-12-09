"""
__init__.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: __init__ package for the POLARIZATION module.

"""

from diffraq.polarization.seam import Seam
from diffraq.polarization.thick_screen import ThickScreen
from diffraq.polarization.vector_sim import VectorSim
from diffraq.polarization.quadrature import *
from diffraq.polarization.perturbations import *
from diffraq.polarization.normal_seam_quad import build_normal_quadrature
