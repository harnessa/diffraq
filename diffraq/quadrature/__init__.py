"""
__init__.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-08-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: __init__ package for the QUADRATURE module.

"""

from diffraq.quadrature.lgwt import lgwt, fresnaq_lgwt
from diffraq.quadrature.cartesian_quad import cartesian_quad, cartesian_edge
from diffraq.quadrature.loci_quad import loci_quad, loci_edge
from diffraq.quadrature.polar_quad import polar_quad, polar_edge
from diffraq.quadrature.petal_quad import petal_quad, petal_edge
from diffraq.quadrature.triangle_quad import triangle_quad, triangle_edge
from diffraq.quadrature.unique_petal_quad import unique_petal_quad, unique_petal_edge
from diffraq.quadrature.circle_quad import circle_quad, circle_edge
# from diffraq.quadrature.square_quad import square_quad, square_edge
