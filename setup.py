#! /usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
import os

setup(
	name='diffraq',
	version='0.1',
    author='Anthony Harness',
    author_email='aharness@princeton.edu',
    description="Diffraction with Areal Quadrature",
    packages=['diffraq'],
    url='https://github.com/harnessa/diffraq',
    license='',
    package_data={},
    requires=['numpy', 'scipy', 'h5py', 'matplotlib', 'mpi4py', 'pytest'],
    )

##
# TELL PEOPLE TO SET ENVIRONMENT VARIABLE
# Code from Jordan Mirocha's ARES code -- https://github.com/mirochaj/ares
##

PKG_name = "DIFFRAQ"
PKG_env = os.getenv(PKG_name)
cwd = os.getcwd()

if not PKG_env:

    import re
    shell = os.getenv('SHELL')

    print("\n")
    print("#" * 78)
    print("It would be in your best interest to set an environment variable")
    print("pointing to this directory.\n")

    if shell:
        if re.search('bash', shell):
            print("Looks like you're using bash, so add the following to your .bashrc:")
            print(f"\n    export {PKG_name}={cwd}")
        elif re.search('csh', shell):
            print("Looks like you're using csh, so add the following to your .cshrc:")
            print(f"\n    setenv {PKG_name} {cwd}")

    print("\nGood luck!")
    print("#" * 78)
    print("\n")

# Print a warning if there's already an environment variable but it's pointing
# somewhere other than the current directory
elif PKG_env != cwd:

    print("\n")
    print("#" * 78)
    print(f"It looks like you've already got an {PKG_name} environment variable " +\
        "set but it's \npointing to a different directory:")
    print(f"\n    {PKG_name}={PKG_env}")

    print(f"\nHowever, we're currently in {cwd}.\n")

    print(f"Is this a different {PKG_name} install (might not cause problems), or " +\
        "perhaps just ")
    print("a typo in your environment variable?")

    print("#" * 78)
    print("\n")
