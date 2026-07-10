## ! DO NOT MANUALLY INVOKE THIS setup.py, USE catkin_make INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['sru_nav_go2'],
    package_dir={'': 'src'},
)

setup(**setup_args)
