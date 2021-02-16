"""
loci_outline.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-15-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class that is the function that describes the shape of a loci
    occulter's edge.

"""

import numpy as np
from diffraq.geometry import Outline

class LociOutline(Outline):

    kind = 'loci'

    def __init__(self, func, diff=None, diff_2nd=-1):
        #FIXME: get this working
        return