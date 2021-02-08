"""
__init__.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: __init__ package for the OCCULTER module.

"""

from diffraq.occulter.occulter import Occulter
from diffraq.occulter.shape_function import Shape_Function
from diffraq.occulter.cartesian_occulter import Cartesian_Occulter
from diffraq.occulter.loci_occulter import Loci_Occulter
from diffraq.occulter.polar_occulter import Polar_Occulter, Circle_Occulter
from diffraq.occulter.starshade_occulter import Starshade_Occulter
import diffraq.occulter.defects
