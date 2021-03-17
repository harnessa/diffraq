#!/bin/bash

#Test in order
pytest test_quadrature.py
pytest test_diffraction_grid.py
pytest test_diffraction_points.py
pytest test_outline.py
pytest test_cartesian_transforms.py
pytest test_occulter.py
pytest test_occulter_configuration.py
pytest test_calc_pupil_field.py
pytest test_circles.py
pytest test_focuser.py
pytest test_starshades.py
pytest test_annulus.py
pytest test_perturbations.py
pytest test_etching_error.py
