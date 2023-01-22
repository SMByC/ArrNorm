# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    README = f.read()
with open(os.path.join(HERE, 'arrnorm', '__init__.py'), encoding='utf-8') as fp:
    VERSION = re.search("__version__ = '([^']+)'", fp.read()).group(1)

setup(
    name='arrnorm',
    version=VERSION,
    description='Automatic relative radiometric normalization',
    long_description=README,
    author='Xavier C. Llano, SMBYC-IDEAM',
    author_email='xavier.corredor.llano@gmail.com, smbyc@ideam.gov.co',
    url='https://github.com/SMByC/ArrNorm',
    license='GPLv3',
    packages=find_packages(),
    install_requires=['gdal',
                      'numpy'],
    platforms=['Windows', 'Linux'],
    scripts=['bin/arrnorm'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: GIS",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Programming Language :: Python",
        "Environment :: Console",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"],
)
