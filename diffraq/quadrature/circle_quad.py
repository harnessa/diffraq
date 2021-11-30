"""
circle_quad.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 11-29-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: wrapper for circular quadarature.

"""

from diffraq.quadrature import polar_quad, polar_edge

def circle_quad(radius, m, n):
    return polar_quad(lambda t: np.ones_like(t)*radius, m, n)

def circle_edge(radius, n):
    return polar_edge(lambda t: np.ones_like(t)*radius, n)
