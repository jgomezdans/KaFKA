#!/usr/bin/env python

from setuptools import setup

requirements = [
    'pytest',
    'numpy',
    'scipy'
    'gdal'
]

setup(name='multiply-KaFKA-inference-engine',
      description='MULTIPLY KaFKA inference engine',
      author='MULTIPLY Team',
      packages=['kafka',
                 'inference',
                 'input_output'],
      install_requires=requirements
)
