#!/bin/bash

#Test in order
pytest test_quadrature.py
pytest test_diffraction_grid.py
pytest test_diffraction_points.py
pytest test_occulter.py
pytest test_calc_pupil_field.py
pytest test_focuser.py
pytest test_starshades.py
