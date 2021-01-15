# diffraq
------------
Diffraction with Areal Quadrature, or _diffraq_, is a fast nonuniform Fourier
method to calculate Fresnel diffraction via areal quadrature. This is a translation
of the code _fresnaq_ (https://github.com/ahbarnett/fresnaq) into Python.

Getting started
---------------------
To clone a copy and install: ::

    git clone https://github.com/harnessa/diffraq
    cd diffraq
    python setup.py install

You'll need to set an environment variable which points to the _diffraq_ install directory, e.g. (in bash):

    export DIFFRAQ=${HOME}/repos/diffraq

Dependencies
--------------------
You will need:

- `numpy <http://www.numpy.org/>`
- `h5py <http://www.h5py.org>`
- `matplotlib <https://pypi.org/project/matplotlib/>`
- `mpi4py <http://mpi4py.scipy.org/>`
- `pytest <https://pypi.org/project/pytest/>`

Documentation
--------------
No public documentation yet :(

Contributors
------------
Primary author: Anthony Harness (Princeton University)
Author of original _fresnaq_ code: Alex Barnett (Flatiron Institute)
