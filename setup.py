#!/usr/bin/env python
import codecs
import os
import re

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


requirements = [
    'pytest',
    'numpy',
    'scipy',
    'gp_emulator',
    'BRDF_descriptors',
    'gdal',
    'matplotlib'
]

setup(name='KaFKA',
      version=find_version("kafka", "__init__.py"),
      description='MULTIPLY KaFKA inference engine',
      author='J Gomez-Dans',
      author_email='j.gomez-dans@ucl.ac.uk',
      url='https://github.com/jgomezdans/KaFKA',
      packages=find_packages(),
      install_requires=requirements
)
