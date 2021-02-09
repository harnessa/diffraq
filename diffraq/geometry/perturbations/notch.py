"""
notch.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-09-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class of the notched edge perturbation.

"""

import numpy as np
import diffraq.quadrature as quad
from diffraq.geometry import Perturbation

class Notch(Perturbation):

    kind = 'notch'

    pass
