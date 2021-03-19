"""
__init__.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: __init__ package for the QUADRATURE module.

"""

from diffraq.quadrature.lgwt import lgwt
from diffraq.quadrature.cartesian_quad import cartesian_quad, cartesian_edge
from diffraq.quadrature.loci_quad import loci_quad
from diffraq.quadrature.polar_quad import polar_quad, polar_edge
from diffraq.quadrature.starshade_quad import starshade_quad, starshade_edge
#Seams
from diffraq.quadrature.seam_polar_quad import seam_polar_quad, seam_polar_edge
from diffraq.quadrature.seam_cartesian_quad import seam_cartesian_quad, seam_cartesian_edge
