import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))

stuff = [package for package in find_packages() if package.startswith('code')]

setup(name='rl-smoothness',
      version='0.0.1',
      description='Test bed for investigating smoothness in Reinforcement Learning',
      author='',
      author_email='',
      packages=[package for package in find_packages() if package.startswith('rlcode')],
      install_requires=['gym', 'numpy', 'noise', 'tensorflow', 'mpi4py'],
)
