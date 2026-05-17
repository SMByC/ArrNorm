# -*- coding: utf-8 -*-

import os
import re
from setuptools import setup, find_packages

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md'), encoding='utf-8') as f:
    README = f.read()
with open(os.path.join(HERE, 'core', '__init__.py'), encoding='utf-8') as fp:
    VERSION = re.search(r"__version__ = '([^']+)'", fp.read()).group(1)

setup(
    name='arrnorm',
    version=VERSION,
    description='Automatic relative radiometric normalization (IR-MAD)',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Xavier C. Llano, SMBYC-IDEAM',
    author_email='xavier.corredor.llano@gmail.com, smbyc@ideam.gov.co',
    url='https://github.com/SMByC/ArrNorm',
    license='GPLv3',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'gdal',
        'numpy>=1.20',
        'scipy>=1.5',
        'matplotlib',
    ],
    platforms=['Windows', 'Linux', 'Mac OS-X'],
    include_package_data=True,
    zip_safe=False,
    scripts=['arrnorm.py'],
    classifiers=[
        'Topic :: Scientific/Engineering :: GIS',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Environment :: Console',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
