# diffraq
------------
Diffraction with Areal Quadrature, or _diffraq_, is a fast nonuniform Fourier
method to calculate Fresnel diffraction via areal quadrature. This is a translation
of the code _fresnaq_ (https://github.com/ahbarnett/fresnaq) into Python. It
shamelessly takes many algorithms from there.

Getting started
---------------------
To clone a copy and install:

    git clone https://github.com/harnessa/diffraq
    cd diffraq
    python setup.py install

You'll need to set an environment variable which points to the _diffraq_ install directory, e.g. (in bash):

    export DIFFRAQ=${HOME}/repos/diffraq

Dependencies
--------------------
You will need:

- `numpy <http://www.numpy.org/>`
- `finufft <https://github.com/flatironinstitute/finufft>`
- `h5py <http://www.h5py.org>`

And optionally:
- `matplotlib <https://pypi.org/project/matplotlib/>`
- `pytest <https://pypi.org/project/pytest/>`

Testing
---------------------
If _pytest_ is installed, testing can be done through:

    sh $DIFFRAQ/tests/run_all_tests.sh

Documentation
--------------
No public documentation yet :(

Contributors
------------
Primary author: Anthony Harness (Princeton University)
Author of original _fresnaq_ code: Alex Barnett (Flatiron Institute)
