#!/usr/bin/env python

from setuptools import setup

requirements = [
    'pytest',
    'numpy',
    'scipy',
    'gdal',
    'BRDF_descriptors',
    'matplotlib'
]

setup(name='KaFKA',
      description='MULTIPLY KaFKA inference engine',
      author='MULTIPLY Team',
      packages=['kafka'],
      install_requires=requirements
)
