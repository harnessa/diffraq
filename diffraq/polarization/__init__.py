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
from diffraq.polarization.seam_cartesian_quad import seam_cartesian_quad, seam_cartesian_edge
from diffraq.polarization.seam_petal_quad import seam_petal_quad, seam_petal_edge
from diffraq.polarization.seam_polar_quad import seam_polar_quad, seam_polar_edge
