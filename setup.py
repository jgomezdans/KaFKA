#!/usr/bin/env python

from setuptools import setup, find_packages

requirements = [
    'pytest',
    'numpy',
    'scipy',
    'gdal',
    # 'BRDF_descriptors', # Not available for automatic installation
    'matplotlib'
]

setup(name='KaFKA',
      description='MULTIPLY KaFKA inference engine',
      author='MULTIPLY Team',
      packages=find_packages(),
      install_requires=requirements
)
