#!/usr/bin/env python

from setuptools import setup

requirements = [
    'pytest',
    'numpy',
    'scipy'
    'gdal'
    'BRDF_descriptors'
]

setup(name='multiply-KaFKA-inference-engine',
      description='MULTIPLY KaFKA inference engine',
      author='MULTIPLY Team',
      packages=['kafka'],
      install_requires=requirements
)
